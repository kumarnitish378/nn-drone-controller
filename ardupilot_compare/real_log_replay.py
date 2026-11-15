"""Run the quad network against a REAL ArduCopter flight log.

Two things are done with the log:
 1. Shadow replay (open loop): at every instant of the real flight the network is given what the real
    autopilot knew (gyro, accelerometer, attitude, position error to ArduPilot's own target, velocity) and
    its motor PWM is compared with the PWM ArduPilot actually sent (RCOU). The network does not fly the
    drone here, so this shows whether it reacts in the same direction and how strongly, not whether it would fly.
 2. Closed loop in our simulator: the network follows the same target track ArduPilot was following, and
    its tracking error is compared with the real flight's.

Log used: an Erle-Copter quad (X frame), ArduCopter 3.3-dev, 5 Dec 2014, from https://github.com/erlerobot/logs
(file pxf/copter/5-12-2014/218.BIN): ALT_HOLD, LOITER, GUIDED, LOITER, AUTO mission, LAND.

usage: python3 real_log_replay.py /path/to/log.BIN out_dir [mass_kg] [output_name] [label]     (--selftest checks the input builder)
"""
import json, math, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); sys.path.insert(0, root)
from pymavlink import mavutil
from sim import MultirotorEnv, quat_from_euler, quat_rotate_inv, euler_from_quat, wrap, G, I_MAX, I_CLAMP, I_ZONE, I_SPEED
from ppo import PPO

WEIGHTS = os.path.join(root, "quad_x_policy_v4.npz")
MODES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL", 9: "LAND"}


class Shadow:
    """Builds the network's inputs exactly as sim.py does, from externally supplied state (sim axes:
    world x north, y west, z up; body x forward, y left, z up), and keeps the network's own memory."""
    def __init__(self, mass, err_clip=2.5, integrate=True):
        self.integrate = integrate       # in open-loop replay the integral would only wind up, so it is held at zero there
        self.M = 4; self.L = mass ** (1 / 3); self.rL = math.sqrt(self.L); self.dt = 0.01 * self.rL
        self.mass = mass; self.err_clip = err_clip
        self.agent = PPO(21 + self.M + 2 * (14 + self.M), self.M); self.agent.load(WEIGHTS)
        self.reset()
    def reset(self):
        self.prev_a = None; self.h1 = self.h2 = None; self.ierr = np.zeros(3); self.last_target = None; self.prev_target = None
    def step(self, q, w, acc_b, pos, vel, volt, target, target_yaw, applied=None):
        """One network step. `applied` = the motor command (-1..1 per motor) that was really sent on the previous
        step, if known; without it the network's own previous output is used."""
        L, rL = self.L, self.rL
        pos = np.asarray(pos, float); vel = np.asarray(vel, float); target = np.asarray(target, float)
        # integral of position error: sim.py updates it after each physics step, from the state reached and the target just flown
        if self.last_target is not None:
            e_post = self.last_target - pos
            tv = np.zeros(3) if self.prev_target is None else (self.last_target - self.prev_target) / self.dt
            if np.linalg.norm(tv) > 5 * rL: tv = np.zeros(3)
            if self.integrate and np.linalg.norm(e_post) / L < I_ZONE and np.linalg.norm(vel) / rL < I_SPEED and np.linalg.norm(tv) / rL < 0.05:
                self.ierr = np.clip(self.ierr + e_post / L * 0.01, -I_CLAMP, I_CLAMP)
            self.prev_target = self.last_target
        self.last_target = target.copy()
        if applied is not None: self.prev_a = np.asarray(applied, float)
        q = np.asarray(q, float)[None]; roll, pitch, yaw = euler_from_quat(q)
        err = target - pos
        eb = quat_rotate_inv(q, err[None])[0] / L; vb = quat_rotate_inv(q, vel[None])[0] / rL
        d = np.linalg.norm(eb); eb = eb * min(1.0, self.err_clip / max(d, 1e-6))
        herr = wrap(target_yaw - yaw[0])
        if self.prev_a is None: self.prev_a = np.zeros(self.M)
        base = np.concatenate([np.asarray(w) * rL / 8.0, np.asarray(acc_b) / (2 * G), [roll[0] / np.pi, pitch[0] / np.pi, np.sin(herr), np.cos(herr)],
                               eb / 5.0, vb / 5.0, [(volt - 15.6) / 1.2], self.prev_a, quat_rotate_inv(q, self.ierr[None])[0] / I_MAX]).astype(np.float32)
        core = np.concatenate([base[0:8], base[10:16], base[17:17 + self.M]])
        if self.h1 is None: self.h1 = core.copy(); self.h2 = core.copy()
        obs = np.concatenate([base, self.h1, self.h2, [np.log10(self.mass) - 1.0]]).astype(np.float32)[None]
        a = np.clip(self.agent.act(obs, deterministic=True)[0][0], -1, 1).astype(float)
        self.h2 = self.h1; self.h1 = core
        self.prev_a = a
        return 1500.0 + 500.0 * a


def selftest():
    """Feed the Shadow builder states from our own simulator (noise off): its outputs must match the simulator's."""
    env = MultirotorEnv(1, randomize=False, noise=False, mass=1.5, err_clip=2.5, seed=3)
    env.reset_idx(np.arange(1), hard=False); sh = Shadow(1.5); agent = sh.agent; worst = 0.0
    sh.prev_a = env.prev_a[0].copy()
    for k in range(1500):
        tgt = np.array([3.0 * (k > 200), -2.0 * (k > 700), 2.0 + 1.0 * (k > 1100)]); env.target[0] = tgt; env.target_yaw[0] = 0.4 * (k > 500)
        a, _ = agent.act(env.observe(), deterministic=True)
        pwm = sh.step(env.q[0], env.w[0], env.acc_b[0], env.p[0], env.v[0], env.volt[0], tgt, env.target_yaw[0])
        worst = max(worst, float(np.abs(pwm - (1500 + 500 * np.clip(a[0], -1, 1))).max()))
        env.step(a, auto_target=False)
    print(f"self-test: largest PWM difference between the shadow builder and the simulator = {worst:.3f} us")
    return worst


# ------------------------------------------------------------------ log reading (ArduCopter 3.3 message names)
def read_log(path):
    """Returns the same arrays for old logs (ArduCopter 3.3: NTUN/CTUN, TimeMS) and current ones (PSCN/PSCE/PSCD, TimeUS)."""
    m = mavutil.mavlink_connection(path); D = {k: [] for k in ("ATT", "IMU", "NTUN", "CTUN", "RCOU", "GPS")}; modes = []; last_t = 0.0; cmds = []; parm = {}
    psc = {"PSCN": [], "PSCE": [], "PSCD": []}; thr = []
    while True:
        x = m.recv_match()
        if x is None: break
        k = x.get_type(); new = hasattr(x, "TimeUS")
        t = x.TimeUS * 1e-6 if new else (x.TimeMS * 1e-3 if hasattr(x, "TimeMS") and k != "GPS" else None)
        if t is not None: last_t = t
        if k == "ATT": D[k].append([t, x.Roll, x.Pitch, x.Yaw, x.DesRoll, x.DesPitch, x.DesYaw])
        elif k == "IMU":
            if not new or x.I == 0: D[k].append([t, x.GyrX, x.GyrY, x.GyrZ, x.AccX, x.AccY, x.AccZ])
        elif k == "NTUN": D[k].append([t, x.DPosX, x.DPosY, x.PosX, x.PosY, x.VelX, x.VelY])
        elif k == "CTUN":
            if new: thr.append([t, x.ThO * 1000.0])
            else: D[k].append([t, x.DAlt, x.Alt, x.CRt, x.ThrOut])
        elif k == "PSCN": psc[k].append([t, x.TPN, x.PN, x.VN])
        elif k == "PSCE": psc[k].append([t, x.TPE, x.PE, x.VE])
        elif k == "PSCD": psc[k].append([t, x.TPD, x.PD, x.VD])
        elif k == "RCOU": D[k].append([t, x.C1, x.C2, x.C3, x.C4] if new else [t, x.Ch1, x.Ch2, x.Ch3, x.Ch4])
        elif k == "GPS": D[k].append([t if new else x.T * 1e-3, x.NSats, x.HDop, x.Spd])
        elif k == "MODE": modes.append([last_t, int(x.ModeNum) if new else int(x.Mode)])   # old logs have no time in MODE: use the previous record's
        elif k == "CMD": cmds.append([t, x.CNum, x.CId, x.Alt])
        elif k == "PARM": parm[x.Name] = x.Value
    if psc["PSCN"]:                                   # current format: metres, down positive -> the old layout (cm, up positive)
        n, e, d = (np.array(psc[k], float) for k in ("PSCN", "PSCE", "PSCD")); L = min(len(n), len(e), len(d)); n, e, d = n[:L], e[:L], d[:L]
        D["NTUN"] = np.stack([n[:, 0], n[:, 1] * 100, e[:, 1] * 100, n[:, 2] * 100, e[:, 2] * 100, n[:, 3] * 100, e[:, 3] * 100], 1).tolist()
        th = np.array(thr, float)
        D["CTUN"] = np.stack([d[:, 0], -d[:, 1], -d[:, 2], -d[:, 3] * 100, np.interp(d[:, 0], th[:, 0], th[:, 1])], 1).tolist()
    return {k: np.array(v, float) for k, v in D.items()}, modes, cmds, parm


def interp(tt, arr, col, angle=False):
    if angle:
        return np.interp(tt, arr[:, 0], np.unwrap(np.radians(arr[:, col])))
    return np.interp(tt, arr[:, 0], arr[:, col])


def channels(pwm):
    """Split 4 motor PWMs (ArduPilot quad X order: front-right, rear-left, front-left, rear-right) into the four control directions."""
    m1, m2, m3, m4 = pwm[:, 0], pwm[:, 1], pwm[:, 2], pwm[:, 3]
    return dict(collective=(m1 + m2 + m3 + m4) / 4, roll=((m2 + m3) - (m1 + m4)) / 4, pitch=((m1 + m3) - (m2 + m4)) / 4, yaw=((m1 + m2) - (m3 + m4)) / 4)


def main():
    path, out = sys.argv[1], sys.argv[2]; mass = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    global LABEL, TAG
    TAG = sys.argv[4] if len(sys.argv) > 4 else "real_log_replay"
    LABEL = sys.argv[5] if len(sys.argv) > 5 else "erlerobot/logs: pxf/copter/5-12-2014/218.BIN (ArduCopter 3.3-dev, Erle-Copter quad X, real flight)"
    os.makedirs(out, exist_ok=True)
    assert selftest() < 1.0
    D, modes, cmds, parm = read_log(path)
    N, C, A, I, Rr = D["NTUN"], D["CTUN"], D["ATT"], D["IMU"], D["RCOU"]
    # stretches where ArduPilot was controlling position (NTUN is logged) and the drone was flying
    gaps = np.nonzero(np.diff(N[:, 0]) > 1.0)[0]; starts = np.r_[0, gaps + 1]; ends = np.r_[gaps, len(N) - 1]
    def mode_at(t):
        mm = [m for tm, m in modes if tm <= t]; return MODES.get(mm[-1], str(mm[-1])) if mm else "?"
    segs = []
    for s_, e_ in zip(starts, ends):
        cuts = [N[s_, 0]] + [tm for tm, _ in modes if N[s_, 0] < tm < N[e_, 0]] + [N[e_, 0]]
        for t0, t1 in zip(cuts[:-1], cuts[1:]):
            if t1 - t0 < 8: continue
            thr = np.interp(np.linspace(t0, t1, 50), C[:, 0], C[:, 4])
            if np.median(thr) < 200: continue                      # not flying
            segs.append((t0 + 1.0, t1 - 0.5, mode_at(t0 + 0.5)))
    sh = Shadow(mass, integrate=False); dt = sh.dt
    rec = []                                                   # rows: t, seg, ap pwm x4, nn pwm x4, pos err n,e,u
    seg_stats = []; sim_tracks = []
    for si, (t0, t1, mname) in enumerate(segs):
        tt = np.arange(t0, t1, dt)
        roll, pitch, yaw = interp(tt, A, 1, True), interp(tt, A, 2, True), interp(tt, A, 3, True); desyaw = interp(tt, A, 6, True)
        gx, gy, gz = interp(tt, I, 1), interp(tt, I, 2), interp(tt, I, 3); ax, ay, az = interp(tt, I, 4), interp(tt, I, 5), interp(tt, I, 6)
        dpn, dpe, pn, pe = (interp(tt, N, c) / 100 for c in (1, 2, 3, 4)); vn, ve = interp(tt, N, 5) / 100, interp(tt, N, 6) / 100
        dalt, alt, crt = interp(tt, C, 1), interp(tt, C, 2), interp(tt, C, 3) / 100
        ap = np.stack([interp(tt, Rr, c) for c in (1, 2, 3, 4)], 1)
        # ---- 1. shadow replay. Convert ArduPilot's north-east-down / forward-right-down to the simulator's axes.
        sh.reset(); nn = np.zeros((len(tt), 4))
        for k in range(len(tt)):
            q = quat_from_euler(np.array([roll[k]]), np.array([-pitch[k]]), np.array([-yaw[k]]))[0]
            nn[k] = sh.step(q, [gx[k], -gy[k], -gz[k]], [ax[k], -ay[k], -az[k]], [pn[k], -pe[k], alt[k]], [vn[k], -ve[k], crt[k]], 15.6,
                            [dpn[k], -dpe[k], dalt[k]], -desyaw[k])
        desr, desp = interp(tt, A, 4, True), interp(tt, A, 5, True)
        tspd = np.hypot(np.gradient(dpn, tt), np.gradient(dpe, tt)); vspd = np.hypot(vn, ve)
        for k in range(0, len(tt), max(1, int(round(0.1 / dt)))):
            rec.append([tt[k], si, *ap[k], *nn[k], dpn[k] - pn[k], dpe[k] - pe[k], dalt[k] - alt[k],
                        np.degrees(desr[k] - roll[k]), np.degrees(desp[k] - pitch[k]), np.degrees(wrap(desyaw[k] - yaw[k])),
                        float(tspd[k] < 0.3 and vspd[k] < 0.7)])
        # ---- 2. closed loop: the network follows the same target track in our simulator
        env = MultirotorEnv(1, randomize=False, noise=True, mass=mass, err_clip=2.5, seed=5); env.reset_idx(np.arange(1), hard=False)
        env.p[0] = [pn[0], -pe[0], alt[0]]; env.v[0] = [vn[0], -ve[0], crt[0]]
        env.q[0] = quat_from_euler(np.array([0.0]), np.array([0.0]), np.array([-yaw[0]]))[0]
        agent = sh.agent; err_sim = np.zeros((len(tt), 3)); pw_sim = np.zeros((len(tt), 4)); pos_sim = np.zeros((len(tt), 3))
        for k in range(len(tt)):
            env.target[0] = [dpn[k], -dpe[k], dalt[k]]; env.target_yaw[0] = -desyaw[k]
            a, _ = agent.act(env.observe(), deterministic=True); _, _, _, _, _, info = env.step(a, auto_target=False)
            err_sim[k] = env.target[0] - env.p[0]; pw_sim[k] = info["pwm"][0]; pos_sim[k] = env.p[0]
        skip = int(3.0 / dt)                                       # let both settle on the track before measuring
        real_h = np.hypot(dpn - pn, dpe - pe)[skip:]; real_v = np.abs(dalt - alt)[skip:]
        sim_h = np.hypot(err_sim[skip:, 0], err_sim[skip:, 1]); sim_v = np.abs(err_sim[skip:, 2])
        seg_stats.append(dict(mode=mname, start_s=round(float(t0), 1), length_s=round(float(t1 - t0), 1),
                              target_speed_mean_mps=round(float(np.hypot(np.gradient(dpn, tt), np.gradient(dpe, tt)).mean()), 2),
                              real_horizontal_error_mean_m=round(float(real_h.mean()), 3), real_horizontal_error_max_m=round(float(real_h.max()), 3),
                              real_height_error_mean_m=round(float(real_v.mean()), 3),
                              network_sim_horizontal_error_mean_m=round(float(sim_h.mean()), 3), network_sim_horizontal_error_max_m=round(float(sim_h.max()), 3),
                              network_sim_height_error_mean_m=round(float(sim_v.mean()), 3)))
        sim_tracks.append(dict(t=tt, target=np.stack([dpn, dpe, dalt], 1), real=np.stack([pn, pe, alt], 1),
                               sim=np.stack([pos_sim[:, 0], -pos_sim[:, 1], pos_sim[:, 2]], 1), mode=mname, pw_sim=pw_sim))
    rec = np.array(rec); ap, nn = rec[:, 2:6], rec[:, 6:10]
    ca, cn = channels(ap), channels(nn)
    hov = rec[:, 16] > 0.5
    def demean(x, mask):
        return np.concatenate([x[(rec[:, 1] == s_) & mask] - x[(rec[:, 1] == s_) & mask].mean() for s_ in range(len(segs)) if ((rec[:, 1] == s_) & mask).sum() > 20])
    def agree(mask):
        o = {}
        for name in ("collective", "roll", "pitch", "yaw"):
            ad, nd = demean(ca[name], mask), demean(cn[name], mask)          # the moving part: each stretch's average removed
            big = np.abs(ad) > ad.std() * 0.5
            o[name] = dict(correlation=round(float(np.corrcoef(ad, nd)[0, 1]), 2), same_direction_share=round(float((np.sign(ad) == np.sign(nd))[big].mean()), 2),
                           ardupilot_rms_us=round(float(ad.std()), 1), network_rms_us=round(float(nd.std()), 1))
        return o
    allm = np.ones(len(rec), bool)
    shadow = dict(whole_replay=agree(allm), hovering_only=agree(hov), hovering_share=round(float(hov.mean()), 2))
    # does each controller push the way ArduPilot's own attitude demand asks? (roll right when it wants more right roll, and so on)
    dem = {}
    for name, col in (("roll", 13), ("pitch", 14), ("yaw", 15)):
        e_ = demean(rec[:, col], allm)
        dem[name] = dict(ardupilot=round(float(np.corrcoef(e_, demean(ca[name], allm))[0, 1]), 2), network=round(float(np.corrcoef(e_, demean(cn[name], allm))[0, 1]), 2))
    shadow["correlation_with_ardupilots_attitude_demand"] = dem
    res = dict(log=LABEL,
               assumed_mass_kg=mass, network_rate_hz=round(1 / dt, 1), compared_at_hz=10,
               log_facts=dict(flight_s=round(float(Rr[-1, 0] - Rr[0, 0]), 0), gps_satellites_median=float(np.median(D["GPS"][:, 1])), gps_hdop_median=float(np.median(D["GPS"][:, 2])),
                              pwm_rate_hz=round(len(Rr) / (Rr[-1, 0] - Rr[0, 0]), 1), imu_rate_hz=round(len(I) / (I[-1, 0] - I[0, 0]), 1),
                              battery_voltage_logged=False, mission_items=[[int(c[1]), int(c[2]), c[3]] for c in cmds]),
               seconds_replayed=round(float(sum(s[1] - s[0] for s in segs)), 0),
               pwm=dict(ardupilot_mean_us=round(float(ap.mean()), 0), network_mean_us=round(float(nn.mean()), 0),
                        ardupilot_range_us=[float(ap.min()), float(ap.max())], network_range_us=[round(float(nn.min())), round(float(nn.max()))],
                        ardupilot_change_per_0p1s_us=round(float(np.abs(np.diff(ap, axis=0)).mean()), 1), network_change_per_0p1s_us=round(float(np.abs(np.diff(nn, axis=0)).mean()), 1),
                        network_time_at_limits_share=round(float(((nn < 1010) | (nn > 1990)).mean()), 3)),
               shadow_replay=shadow, closed_loop_same_target_track=seg_stats)
    json.dump(res, open(os.path.join(out, TAG + ".json"), "w"), indent=1); print(json.dumps(res, indent=1))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    CA, CN, SEC, INK, BG = "#eb6834", "#2a78d6", "#52514e", "#0b0b0b", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})
    big = int(np.argmax([s[1] - s[0] for s in segs])); auto = [i for i, s in enumerate(segs) if s[2] == "AUTO"]; show = auto[0] if auto else big
    m = rec[:, 1] == show; t = rec[m, 0] - rec[m, 0][0]; w = t < 40
    fig, ax = plt.subplots(6, 1, figsize=(7, 21), gridspec_kw=dict(height_ratios=[1, 1, 1, 1, 1.5, 1]))
    for a, name, ttl in zip(ax[:4], ("collective", "roll", "pitch", "yaw"), ("Collective (average of the four motors)", "Roll command", "Pitch command", "Yaw command")):
        x, y = ca[name][m][w], cn[name][m][w]
        if name == "collective":
            a.plot(t[w], x, color=CA, lw=1.4, label="ArduPilot, real flight"); a.plot(t[w], y, color=CN, lw=1.2, label="Network, shadow")
        else:
            a.plot(t[w], x - x.mean(), color=CA, lw=1.4, label="ArduPilot, real flight"); a.plot(t[w], y - y.mean(), color=CN, lw=1.2, label="Network, shadow")
        a.set_title(f"{ttl}, PWM µs ({segs[show][2]} mode)"); a.legend(frameon=False, loc="upper right", ncol=2)
    ax[3].set_xlabel("Time in this stretch (s)")
    tr = sim_tracks[show]; a = ax[4]
    a.plot(tr["target"][:, 1], tr["target"][:, 0], "--", color=SEC, lw=1, label="ArduPilot's target track"); a.plot(tr["real"][:, 1], tr["real"][:, 0], color=CA, lw=1.5, label="Real flight (ArduPilot)")
    a.plot(tr["sim"][:, 1], tr["sim"][:, 0], color=CN, lw=1.3, label="Network in our simulator"); a.set_aspect("equal"); a.set_xlabel("East (m)"); a.set_ylabel("North (m)")
    a.set_title(f"Top view, {tr['mode']} stretch: same target track"); a.legend(frameon=False, loc="best")
    a = ax[5]; x = np.arange(len(seg_stats)); wd = 0.38
    a.bar(x - wd / 2, [s["real_horizontal_error_mean_m"] for s in seg_stats], wd, color=CA, label="Real flight (ArduPilot)")
    a.bar(x + wd / 2, [s["network_sim_horizontal_error_mean_m"] for s in seg_stats], wd, color=CN, label="Network in our simulator")
    a.set_xticks(x); a.set_xticklabels([f"{s['mode']}\n{s['length_s']:.0f} s" for s in seg_stats]); a.grid(axis="x", visible=False)
    a.set_title("Average horizontal distance from the target, each stretch (m)"); a.legend(frameon=False)
    fig.tight_layout(h_pad=1.5); fig.savefig(os.path.join(out, TAG + ".png"), dpi=130)


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest()
    else: main()
