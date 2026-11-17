"""
Vectorised multirotor simulator for neural-network flight control experiments.

Frames: world = ENU-style (z up), body = FLU (x forward, y left, z up).
Motor order and spin directions follow ArduPilot's QUAD X / HEXA X layouts.

Network interface
  inputs  (20 + n_motors):  gyro(3) acc(3) roll,pitch(2) heading-error sin,cos(2)
                            position-error body(3) velocity body(3) battery(1)
                            previous motor outputs(n_motors)
                            integrated position error body(3)   <- "I term"
                            2 previous frames of gyro, acc, roll/pitch, position error,
                            velocity and motor outputs (2 x (14 + n_motors))
                            vehicle mass class(1): log10(nominal mass in kg) - 1
  outputs (n_motors):       value in [-1, 1]  ->  PWM = 1500 + 500 * value  (1000..2000 us)
"""
import os
import backend
from backend import xp as np      # NumPy, or CuPy when NN_DEVICE=cuda (see backend.py)

G = 9.81
I_MAX = 1.5      # m*s, scale of the integrator input
I_CLAMP = 4.0    # m*s, integrator clamp (wide enough to trim out thin air at 2000 m)
I_ZONE = 1.0     # m, integrate only when this close to the target ...
I_SPEED = 0.5    # m/s ... and nearly settled relative to it (anti-windup)

# angle = degrees clockwise from the nose (ArduPilot convention), spin: +1 = CCW, -1 = CW
FRAMES = {
    "quad_x": dict(angles=[45, -135, -45, 135], spin=[+1, +1, -1, -1]),
    "hex_x": dict(angles=[90, -90, -30, 150, 30, -150], spin=[-1, +1, -1, +1, +1, -1]),
}

DEFAULTS = dict(
    mass=1.0,            # kg  nominal mass; the airframe is scaled from the 1 kg reference (see reset_idx)
    mass_range=None,     # (lo, hi) kg: training samples the nominal mass log-uniformly in this range
    ground=False,        # True: solid ground at z = 0 (the drone rests on it until thrust exceeds weight)
    alt0=0.0,            # m   start altitude above sea level (air density falls with height)
    alt_range=None,      # (lo, hi) m: training samples the start altitude in this range
    arm=0.17,            # m
    inertia=(0.010, 0.010, 0.018),  # kg m^2
    thrust_to_weight=2.8,  # all motors at full throttle, nominal battery
    yaw_coeff=0.016,     # m  (reaction torque per N of thrust)
    motor_tau=0.05,      # s  first-order motor lag
    drag=0.25,           # N per m/s
    v_nom=16.0,          # V  (4S)
    ctrl_dt=0.01,        # s  -> 100 Hz control loop
    substeps=2,          # physics at 200 Hz
    w_da2=1.0, w_da1=0.3, w_tilt=0.5, w_spd=0.05, spd_free=3.0, tilt_free=0.6,   # smoothness penalties
    p_moving=0.4,        # share of training targets that move (lines and arcs)
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
    def __init__(self, n_envs=1, frame=None, randomize=True, noise=True,
                 episode_len=1000, seed=0, **overrides):
        self.cfg = dict(DEFAULTS, **overrides)
        self.N = n_envs
        frame = frame or os.environ.get("NN_FRAME", "quad_x")   # NN_FRAME=hex_x selects the hexacopter
        self.frame = frame
        self.randomize = randomize
        self.noise = noise
        self.episode_len = episode_len
        self.rng = backend.default_rng(seed)

        f = FRAMES[frame]
        ang = np.deg2rad(np.array(f["angles"], dtype=np.float64))
        self.M = len(ang)
        L = self.cfg["arm"]
        self.mx = L * np.cos(ang)       # forward
        self.my = -L * np.sin(ang)      # left
        self.spin = np.array(f["spin"], dtype=np.float64)
        self.hist_dim = 14 + self.M
        self.obs_dim = 21 + self.M + 2 * self.hist_dim
        self.act_dim = self.M

        N = self.N
        self.p = np.zeros((N, 3)); self.v = np.zeros((N, 3))
        self.q = np.zeros((N, 4)); self.q[:, 0] = 1
        self.w = np.zeros((N, 3))
        self.mot = np.zeros((N, self.M))
        self.prev_a = np.zeros((N, self.M))
        self.target = np.zeros((N, 3)); self.target_yaw = np.zeros(N)
        self.t = np.zeros(N, dtype=np.int64)
        self.ierr = np.zeros((N, 3))
        self.tgt_vel = np.zeros((N, 3)); self.tgt_turn = np.zeros(N)
        self.prev_target = np.zeros((N, 3))
        self.h1 = np.zeros((N, self.hist_dim), np.float32); self.h2 = np.zeros((N, self.hist_dim), np.float32)
        self.fresh = np.ones(N, bool); self._last_base = None   # integral of position error, world frame (m*s)
        self.acc_b = np.zeros((N, 3))
        self.volt = np.full(N, self.cfg["v_nom"])
        # per-env physical parameters
        self.mass = np.zeros(N); self.inertia = np.zeros((N, 3))
        self.tmax = np.zeros(N); self.tau = np.zeros(N)
        self.v0 = np.zeros(N); self.wind = np.zeros((N, 3))
        self.kyaw = np.zeros(N)
        self.m_nom = np.ones(N); self.arm_s = np.ones(N); self.drag = np.zeros(N)
        self.rdamp = np.zeros(N); self.alt0 = np.zeros(N); self.dt_env = np.full(N, self.cfg["ctrl_dt"])
        self.time = np.zeros(N)
        self.on_ground = np.zeros(N, bool); self.impact = np.zeros(N)   # impact = hardest touchdown speed, m/s
        self.reset_idx(np.arange(N))

    # ------------------------------------------------------------------
    def reset_idx(self, idx, hard=True):
        n = len(idx)
        if n == 0:
            return
        c, r = self.cfg, self.rng
        k = (lambda lo, hi: r.uniform(lo, hi, n)) if self.randomize else (lambda lo, hi: np.ones(n))
        # Scaling laws from the 1 kg reference airframe (same thrust-to-weight at every size):
        #   length L ~ m^(1/3), inertia ~ m^(5/3), drag area ~ m^(2/3), motor/prop lag ~ sqrt(L)
        # A drone scaled this way behaves like the 1 kg one with lengths x L and time x sqrt(L)
        # (Froude similarity). The network therefore sees size-normalised inputs (see _base) and
        # its control period is 10 ms x sqrt(L): 100 Hz at 1 kg, about 46 Hz at 100 kg.
        if c["mass_range"] is not None and self.randomize:
            lo, hi = np.log10(c["mass_range"][0]), np.log10(c["mass_range"][1])
            mn = 10 ** r.uniform(lo, hi, n)
        else:
            mn = np.full(n, float(c["mass"]))
        sc = mn ** (1 / 3)
        self.m_nom[idx] = mn; self.arm_s[idx] = sc
        self.mass[idx] = mn * k(0.8, 1.25)                      # payload uncertainty
        self.inertia[idx] = np.array(c["inertia"]) * (mn ** (5 / 3))[:, None] * np.stack([k(0.8, 1.25)] * 3, 1)
        self.tmax[idx] = c["thrust_to_weight"] * mn * G / self.M * k(0.85, 1.15)
        self.tau[idx] = c["motor_tau"] * np.sqrt(sc) * k(0.7, 1.4)
        self.dt_env[idx] = c["ctrl_dt"] * np.sqrt(sc)
        self.kyaw[idx] = c["yaw_coeff"] * sc * k(0.8, 1.25)
        self.drag[idx] = c["drag"] * mn ** (2 / 3)
        self.rdamp[idx] = 0.002 * mn ** (5 / 3)
        if c["alt_range"] is not None and self.randomize:
            self.alt0[idx] = r.uniform(c["alt_range"][0], c["alt_range"][1], n)
        else:
            self.alt0[idx] = c["alt0"]
        if self.randomize:
            self.v0[idx] = r.uniform(14.4, 16.8, n)
            wdir = r.uniform(-np.pi, np.pi, n); wspd = r.uniform(0, 3.0, n)
            self.wind[idx] = np.stack([wspd * np.cos(wdir), wspd * np.sin(wdir), np.zeros(n)], 1)
        else:
            self.v0[idx] = c["v_nom"]; self.wind[idx] = 0
        self.volt[idx] = self.v0[idx]

        self.p[idx] = 0
        self.v[idx] = r.uniform(-2, 2, (n, 3)) * np.sqrt(sc)[:, None] if hard else 0
        if hard:
            self.q[idx] = quat_from_euler(r.uniform(-0.7, 0.7, n), r.uniform(-0.7, 0.7, n),
                                          r.uniform(-np.pi, np.pi, n))
            self.w[idx] = r.uniform(-2, 2, (n, 3)) / np.sqrt(sc)[:, None]
        else:
            self.q[idx] = np.array([1.0, 0, 0, 0]); self.w[idx] = 0
        hover = np.sqrt(self.mass[idx] * G / (self.M * self.tmax[idx]))
        self.mot[idx] = hover[:, None] * (r.uniform(0.6, 1.2, (n, self.M)) if hard else 1.0)
        self.prev_a[idx] = 2 * self.mot[idx] - 1
        self.t[idx] = 0
        self.time[idx] = 0
        self.ierr[idx] = 0
        self.fresh[idx] = True
        self.acc_b[idx] = np.array([0, 0, G])
        self._new_target(idx)

    def _new_target(self, idx):
        n = len(idx); r = self.rng
        off = np.stack([r.uniform(-4, 4, n), r.uniform(-4, 4, n), r.uniform(-2, 2, n)], 1)
        L = self.arm_s[idx]
        self.target[idx] = self.p[idx] + off * L[:, None]
        self.target_yaw[idx] = r.uniform(-np.pi, np.pi, n)
        mv = r.random(n) < self.cfg["p_moving"]
        d = r.uniform(-np.pi, np.pi, n); sp = r.uniform(0.5, 2.0, n) * mv * np.sqrt(L)
        self.tgt_vel[idx] = np.stack([sp * np.cos(d), sp * np.sin(d), np.zeros(n)], 1)
        self.tgt_turn[idx] = r.uniform(-0.5, 0.5, n) * mv * (r.random(n) < 0.7) / np.sqrt(L)
        self.prev_target[idx] = self.target[idx]

    # ------------------------------------------------------------------
    def _base(self):
        r = self.rng
        roll, pitch, yaw = euler_from_quat(self.q)
        err_b = quat_rotate_inv(self.q, self.target - self.p)
        vel_b = quat_rotate_inv(self.q, self.v)
        L = self.arm_s[:, None]; rL = np.sqrt(L)       # size normalisation (both 1 for the 1 kg drone)
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
        gyro = gyro * rL; err_b = err_b / L; vel_b = vel_b / rL
        d = np.linalg.norm(err_b, axis=1, keepdims=True)
        err_b = err_b * np.minimum(1.0, self.cfg["err_clip"] / np.maximum(d, 1e-6))
        return np.concatenate([
            gyro / 8.0, acc / (2 * G),
            (roll / np.pi)[:, None], (pitch / np.pi)[:, None],
            np.sin(herr)[:, None], np.cos(herr)[:, None],
            err_b / 5.0, vel_b / 5.0,
            ((volt - 15.6) / 1.2)[:, None],
            self.prev_a,
            quat_rotate_inv(self.q, self.ierr) / I_MAX,
        ], axis=1).astype(np.float32)

    def _core(self, base):
        M = self.M
        return np.concatenate([base[:, 0:8], base[:, 10:16], base[:, 17:17 + M]], axis=1)

    def _obs(self, cache=True):
        base = self._base()
        if cache:
            if self.fresh.any():                      # after a reset the history is the current frame
                c = self._core(base)[self.fresh]
                self.h1[self.fresh] = c; self.h2[self.fresh] = c
                self.fresh[:] = False
            self._last_base = base
        mcls = (np.log10(self.m_nom) - 1.0).astype(np.float32)[:, None]
        return np.concatenate([base, self.h1, self.h2, mcls], axis=1)

    def observe(self):
        return self._obs()

    # ------------------------------------------------------------------
    def _physics(self, u, dt):
        self.mot += (u - self.mot) * np.minimum(1.0, dt / self.tau)[:, None]
        vscale = (self.volt / self.cfg["v_nom"]) ** 2
        # ISA air density ratio: thinner air at altitude gives less thrust (and less drag)
        rho = np.clip(1.0 - 2.2558e-5 * (self.alt0 + self.p[:, 2]), 0.3, 1.1) ** 4.2559
        T = self.tmax[:, None] * (vscale * rho)[:, None] * self.mot ** 2  # N per motor
        Ttot = T.sum(1)
        # battery sag with load
        self.volt = self.v0 - 0.9 * (self.mot ** 2).mean(1)
        tq = np.stack([(T * self.my).sum(1) * self.arm_s, -(T * self.mx).sum(1) * self.arm_s,
                       -(T * self.spin).sum(1) * self.kyaw], 1)
        tq -= self.rdamp[:, None] * self.w                                             # rotational damping
        Iw = self.inertia * self.w
        self.w += (tq - np.cross(self.w, Iw)) / self.inertia * dt[:, None]
        # quaternion integration
        w = self.w; q = self.q
        dq = 0.5 * np.stack([
            -q[:, 1] * w[:, 0] - q[:, 2] * w[:, 1] - q[:, 3] * w[:, 2],
            q[:, 0] * w[:, 0] + q[:, 2] * w[:, 2] - q[:, 3] * w[:, 1],
            q[:, 0] * w[:, 1] - q[:, 1] * w[:, 2] + q[:, 3] * w[:, 0],
            q[:, 0] * w[:, 2] + q[:, 1] * w[:, 1] - q[:, 2] * w[:, 0]], 1)
        q = q + dq * dt[:, None]
        self.q = q / np.linalg.norm(q, axis=1, keepdims=True)
        # translation
        thrust_b = np.zeros((self.N, 3)); thrust_b[:, 2] = Ttot
        f = quat_rotate(self.q, thrust_b) - (self.drag * rho)[:, None] * (self.v - self.wind)
        a = f / self.mass[:, None]
        self.acc_b = quat_rotate_inv(self.q, a)        # accelerometer = specific force
        a[:, 2] -= G
        self.v += a * dt[:, None]
        self.p += self.v * dt[:, None]
        if self.cfg["ground"]:
            g = (self.p[:, 2] <= 0) & (self.v[:, 2] <= 0)
            new = g & ~self.on_ground
            self.impact[new] = np.maximum(self.impact[new], -self.v[new, 2])
            if g.any():                                   # resting on the ground: level, not moving
                _, _, yaw = euler_from_quat(self.q[g])
                ng = int(g.sum())
                self.q[g] = quat_from_euler(np.zeros(ng), np.zeros(ng), yaw)
                self.p[g, 2] = 0; self.v[g] = 0; self.w[g] = 0; self.acc_b[g] = np.array([0, 0, G])
            self.on_ground = g

    def step(self, action, auto_target=True):
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        if self._last_base is not None:               # the frame the network just used becomes history
            self.h2[...] = self.h1; self.h1[...] = self._core(self._last_base)
        if auto_target:                               # moving targets (training only)
            c, s_ = np.cos(self.tgt_turn * self.dt_env), np.sin(self.tgt_turn * self.dt_env)
            vx, vy = self.tgt_vel[:, 0].copy(), self.tgt_vel[:, 1].copy()
            self.tgt_vel[:, 0] = c * vx - s_ * vy; self.tgt_vel[:, 1] = s_ * vx + c * vy
            self.target += self.tgt_vel * self.dt_env[:, None]
            self.target_yaw = wrap(self.target_yaw + self.tgt_turn * self.dt_env)
        pwm = 1500.0 + 500.0 * a
        u = (pwm - 1000.0) / 1000.0
        dt = self.dt_env / self.cfg["substeps"]
        for _ in range(self.cfg["substeps"]):
            self._physics(u, dt)
        self.t += 1
        self.time += self.dt_env

        L = self.arm_s; rL = np.sqrt(L)
        err = self.target - self.p
        dist_m = np.linalg.norm(err, axis=1)               # metres
        dist = dist_m / L                                   # size-normalised from here on
        _, _, yaw = euler_from_quat(self.q)
        herr = np.abs(wrap(self.target_yaw - yaw))
        up_z = 1 - 2 * (self.q[:, 1] ** 2 + self.q[:, 2] ** 2)   # cos(tilt)
        da = a - self.prev_a
        tv = (self.target - self.prev_target) / self.dt_env[:, None]     # target velocity (0 on a waypoint jump)
        tv[np.linalg.norm(tv, axis=1) > 5.0 * rL] = 0.0
        self.prev_target = self.target.copy()
        speed = np.linalg.norm(self.v, axis=1) / rL
        wn = np.linalg.norm(self.w, axis=1) * rL
        # integrate only when close, settled, and the target itself is not moving
        near = (dist < I_ZONE) & (np.linalg.norm(self.v, axis=1) / rL < I_SPEED) & (np.linalg.norm(tv, axis=1) / rL < 0.05)
        self.ierr += (err / L[:, None]) * near[:, None] * self.cfg["ctrl_dt"]
        np.clip(self.ierr, -I_CLAMP, I_CLAMP, out=self.ierr)
        tilt = np.arccos(np.clip(up_z, -1, 1))
        c = self.cfg
        rew = (1.0
               + 1.5 * np.exp(-2.0 * dist) + 1.0 * np.exp(-10.0 * dist)
               - 0.25 * np.minimum(dist, 6.0)
               - 0.05 * wn
               - 0.15 * herr
               - c["w_da2"] * (da ** 2).sum(1) - c["w_da1"] * np.abs(da).sum(1)
               - c["w_spd"] * np.maximum(speed - c["spd_free"], 0.0)
               - c["w_tilt"] * np.maximum(tilt - c["tilt_free"], 0.0))
        crashed = (up_z < 0.0) | (dist > 12.0) | (speed > 20.0)
        rew = np.where(crashed, -10.0, rew)
        timeout = (self.t >= self.episode_len) & ~crashed
        self.prev_a = a.copy()
        info = dict(dist=dist_m, herr=herr, crashed=crashed, timeout=timeout, pwm=pwm)

        final_obs = None
        if auto_target:
            # occasionally jump to a new waypoint so the policy learns transitions
            sw = np.nonzero((self.rng.random(self.N) < 1 / 400.0) & ~crashed & ~timeout)[0]
            self._new_target(sw)
            if timeout.any():
                final_obs = self._obs(cache=False)
            done_idx = np.nonzero(crashed | timeout)[0]
            self.reset_idx(done_idx)
        return self._obs(), (rew * 0.1).astype(np.float32), crashed, timeout, final_obs, info
