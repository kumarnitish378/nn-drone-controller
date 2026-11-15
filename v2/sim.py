"""
Vectorised multirotor simulator for neural-network flight control experiments.

Frames: world = ENU-style (z up), body = FLU (x forward, y left, z up).
Motor order and spin directions follow ArduPilot's QUAD X / HEXA X layouts.

Network interface
  inputs  (20 + n_motors):  gyro(3) acc(3) roll,pitch(2) heading-error sin,cos(2)
                            position-error body(3) velocity body(3) battery(1)
                            previous motor outputs(n_motors)
                            integrated position error body(3)   <- learned "I term"
  outputs (n_motors):       value in [-1, 1]  ->  PWM = 1500 + 500 * value  (1000..2000 us)
"""
import numpy as np

G = 9.81
I_MAX = 1.5      # m*s, integrator clamp
I_ZONE = 1.0     # m, integrate only when this close to the target (anti-windup)

# angle = degrees clockwise from the nose (ArduPilot convention), spin: +1 = CCW, -1 = CW
FRAMES = {
    "quad_x": dict(angles=[45, -135, -45, 135], spin=[+1, +1, -1, -1]),
    "hex_x": dict(angles=[90, -90, -30, 150, 30, -150], spin=[-1, +1, -1, +1, +1, -1]),
}

DEFAULTS = dict(
    mass=1.0,            # kg
    arm=0.17,            # m
    inertia=(0.010, 0.010, 0.018),  # kg m^2
    thrust_to_weight=2.8,  # all motors at full throttle, nominal battery
    yaw_coeff=0.016,     # m  (reaction torque per N of thrust)
    motor_tau=0.05,      # s  first-order motor lag
    drag=0.25,           # N per m/s
    v_nom=16.0,          # V  (4S)
    ctrl_dt=0.01,        # s  -> 100 Hz control loop
    substeps=2,          # physics at 200 Hz
    err_clip=5.0,        # m  position error fed to the network is limited to this (acts as a speed limit)
)


def quat_rotate(q, v):
    """body -> world"""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    qv = q[:, 1:]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_rotate_inv(q, v):
    """world -> body"""
    qc = q * np.array([1, -1, -1, -1], dtype=q.dtype)
    return quat_rotate(qc, v)


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy], axis=1)


def euler_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MultirotorEnv:
    def __init__(self, n_envs=1, frame="quad_x", randomize=True, noise=True,
                 episode_len=1000, seed=0, **overrides):
        self.cfg = dict(DEFAULTS, **overrides)
        self.N = n_envs
        self.frame = frame
        self.randomize = randomize
        self.noise = noise
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

        f = FRAMES[frame]
        ang = np.deg2rad(np.array(f["angles"], dtype=np.float64))
        self.M = len(ang)
        L = self.cfg["arm"]
        self.mx = L * np.cos(ang)       # forward
        self.my = -L * np.sin(ang)      # left
        self.spin = np.array(f["spin"], dtype=np.float64)
        self.obs_dim = 20 + self.M
        self.act_dim = self.M

        N = self.N
        self.p = np.zeros((N, 3)); self.v = np.zeros((N, 3))
        self.q = np.zeros((N, 4)); self.q[:, 0] = 1
        self.w = np.zeros((N, 3))
        self.mot = np.zeros((N, self.M))
        self.prev_a = np.zeros((N, self.M))
        self.target = np.zeros((N, 3)); self.target_yaw = np.zeros(N)
        self.t = np.zeros(N, dtype=np.int64)
        self.ierr = np.zeros((N, 3))   # integral of position error, world frame (m*s)
        self.acc_b = np.zeros((N, 3))
        self.volt = np.full(N, self.cfg["v_nom"])
        # per-env physical parameters
        self.mass = np.zeros(N); self.inertia = np.zeros((N, 3))
        self.tmax = np.zeros(N); self.tau = np.zeros(N)
        self.v0 = np.zeros(N); self.wind = np.zeros((N, 3))
        self.kyaw = np.zeros(N)
        self.reset_idx(np.arange(N))

    # ------------------------------------------------------------------
    def reset_idx(self, idx, hard=True):
        n = len(idx)
        if n == 0:
            return
        c, r = self.cfg, self.rng
        k = (lambda lo, hi: r.uniform(lo, hi, n)) if self.randomize else (lambda lo, hi: np.ones(n))
        self.mass[idx] = c["mass"] * k(0.8, 1.25)
        self.inertia[idx] = np.array(c["inertia"]) * np.stack([k(0.8, 1.25)] * 3, 1)
        self.tmax[idx] = c["thrust_to_weight"] * c["mass"] * G / self.M * k(0.85, 1.15)
        self.tau[idx] = c["motor_tau"] * k(0.7, 1.4)
        self.kyaw[idx] = c["yaw_coeff"] * k(0.8, 1.25)
        if self.randomize:
            self.v0[idx] = r.uniform(14.4, 16.8, n)
            wdir = r.uniform(-np.pi, np.pi, n); wspd = r.uniform(0, 3.0, n)
            self.wind[idx] = np.stack([wspd * np.cos(wdir), wspd * np.sin(wdir), np.zeros(n)], 1)
        else:
            self.v0[idx] = c["v_nom"]; self.wind[idx] = 0
        self.volt[idx] = self.v0[idx]

        self.p[idx] = 0
        self.v[idx] = r.uniform(-2, 2, (n, 3)) if hard else 0
        if hard:
            self.q[idx] = quat_from_euler(r.uniform(-0.7, 0.7, n), r.uniform(-0.7, 0.7, n),
                                          r.uniform(-np.pi, np.pi, n))
            self.w[idx] = r.uniform(-2, 2, (n, 3))
        else:
            self.q[idx] = np.array([1.0, 0, 0, 0]); self.w[idx] = 0
        hover = np.sqrt(self.mass[idx] * G / (self.M * self.tmax[idx]))
        self.mot[idx] = hover[:, None] * (r.uniform(0.6, 1.2, (n, self.M)) if hard else 1.0)
        self.prev_a[idx] = 2 * self.mot[idx] - 1
        self.t[idx] = 0
        self.ierr[idx] = 0
        self.acc_b[idx] = np.array([0, 0, G])
        self._new_target(idx)

    def _new_target(self, idx):
        n = len(idx); r = self.rng
        off = np.stack([r.uniform(-4, 4, n), r.uniform(-4, 4, n), r.uniform(-2, 2, n)], 1)
        self.target[idx] = self.p[idx] + off
        self.target_yaw[idx] = r.uniform(-np.pi, np.pi, n)

    # ------------------------------------------------------------------
    def _obs(self):
        r = self.rng
        roll, pitch, yaw = euler_from_quat(self.q)
        err_b = quat_rotate_inv(self.q, self.target - self.p)
        vel_b = quat_rotate_inv(self.q, self.v)
        gyro = self.w.copy(); acc = self.acc_b.copy()
        herr = wrap(self.target_yaw - yaw)
        volt = self.volt.copy()
        if self.noise:
            N = self.N
            gyro += r.normal(0, 0.02, (N, 3)); acc += r.normal(0, 0.3, (N, 3))
            roll = roll + r.normal(0, 0.005, N); pitch = pitch + r.normal(0, 0.005, N)
            herr = herr + r.normal(0, 0.01, N)
            err_b += r.normal(0, 0.03, (N, 3)); vel_b += r.normal(0, 0.05, (N, 3))
            volt += r.normal(0, 0.03, N)
        # far targets are clipped to 5 m so the input becomes "fly in this direction"
        d = np.linalg.norm(err_b, axis=1, keepdims=True)
        err_b = err_b * np.minimum(1.0, self.cfg["err_clip"] / np.maximum(d, 1e-6))
        obs = np.concatenate([
            gyro / 8.0, acc / (2 * G),
            (roll / np.pi)[:, None], (pitch / np.pi)[:, None],
            np.sin(herr)[:, None], np.cos(herr)[:, None],
            err_b / 5.0, vel_b / 5.0,
            ((volt - 15.6) / 1.2)[:, None],
            self.prev_a,
            quat_rotate_inv(self.q, self.ierr) / I_MAX,
        ], axis=1)
        return obs.astype(np.float32)

    def observe(self):
        return self._obs()

    # ------------------------------------------------------------------
    def _physics(self, u, dt):
        self.mot += (u - self.mot) * np.minimum(1.0, dt / self.tau)[:, None]
        vscale = (self.volt / self.cfg["v_nom"]) ** 2
        T = self.tmax[:, None] * vscale[:, None] * self.mot ** 2         # N per motor
        Ttot = T.sum(1)
        # battery sag with load
        self.volt = self.v0 - 0.9 * (self.mot ** 2).mean(1)
        tq = np.stack([(T * self.my).sum(1), -(T * self.mx).sum(1),
                       -(T * self.spin).sum(1) * self.kyaw], 1)
        tq -= 0.002 * self.w                                             # rotational damping
        Iw = self.inertia * self.w
        self.w += (tq - np.cross(self.w, Iw)) / self.inertia * dt
        # quaternion integration
        w = self.w; q = self.q
        dq = 0.5 * np.stack([
            -q[:, 1] * w[:, 0] - q[:, 2] * w[:, 1] - q[:, 3] * w[:, 2],
            q[:, 0] * w[:, 0] + q[:, 2] * w[:, 2] - q[:, 3] * w[:, 1],
            q[:, 0] * w[:, 1] - q[:, 1] * w[:, 2] + q[:, 3] * w[:, 0],
            q[:, 0] * w[:, 2] + q[:, 1] * w[:, 1] - q[:, 2] * w[:, 0]], 1)
        q = q + dq * dt
        self.q = q / np.linalg.norm(q, axis=1, keepdims=True)
        # translation
        thrust_b = np.zeros((self.N, 3)); thrust_b[:, 2] = Ttot
        f = quat_rotate(self.q, thrust_b) - self.cfg["drag"] * (self.v - self.wind)
        a = f / self.mass[:, None]
        self.acc_b = quat_rotate_inv(self.q, a)        # accelerometer = specific force
        a[:, 2] -= G
        self.v += a * dt
        self.p += self.v * dt

    def step(self, action, auto_target=True):
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        pwm = 1500.0 + 500.0 * a
        u = (pwm - 1000.0) / 1000.0
        dt = self.cfg["ctrl_dt"] / self.cfg["substeps"]
        for _ in range(self.cfg["substeps"]):
            self._physics(u, dt)
        self.t += 1

        err = self.target - self.p
        dist = np.linalg.norm(err, axis=1)
        _, _, yaw = euler_from_quat(self.q)
        herr = np.abs(wrap(self.target_yaw - yaw))
        up_z = 1 - 2 * (self.q[:, 1] ** 2 + self.q[:, 2] ** 2)   # cos(tilt)
        da = a - self.prev_a
        near = dist < I_ZONE
        self.ierr += err * near[:, None] * self.cfg["ctrl_dt"]
        np.clip(self.ierr, -I_MAX, I_MAX, out=self.ierr)
        rew = (1.0
               + 1.5 * np.exp(-2.0 * dist) + 1.0 * np.exp(-10.0 * dist)
               - 0.25 * np.minimum(dist, 6.0)
               - 0.05 * np.linalg.norm(self.w, axis=1)
               - 0.15 * herr
               - 0.3 * (da ** 2).sum(1)
               - 0.03 * np.maximum(np.linalg.norm(self.v, axis=1) - 4.0, 0.0))
        crashed = (up_z < 0.0) | (dist > 12.0) | (np.linalg.norm(self.v, axis=1) > 20.0)
        rew = np.where(crashed, -10.0, rew)
        timeout = (self.t >= self.episode_len) & ~crashed
        self.prev_a = a.copy()
        info = dict(dist=dist, herr=herr, crashed=crashed, timeout=timeout, pwm=pwm)

        final_obs = None
        if auto_target:
            # occasionally jump to a new waypoint so the policy learns transitions
            sw = np.nonzero((self.rng.random(self.N) < 1 / 400.0) & ~crashed & ~timeout)[0]
            self._new_target(sw)
            if timeout.any():
                final_obs = self._obs()
            done_idx = np.nonzero(crashed | timeout)[0]
            self.reset_idx(done_idx)
        return self._obs(), (rew * 0.1).astype(np.float32), crashed, timeout, final_obs, info
