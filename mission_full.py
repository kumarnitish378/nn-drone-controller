"""Full mission from disarmed to disarmed, with solid ground. Axes: x forward, y left, z up (metres from the take-off point).
The target point moves along straight lines at set speeds (like an autopilot's waypoint navigation)."""
import sys, json
import numpy as np
from sim import MultirotorEnv, euler_from_quat
from ppo import PPO

weights = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "mission_full"
A = [float(x) for x in sys.argv[3:7]]       # optional: mass kg, horizontal / climb / descent speed m/s
MASS = A[0] if len(A) > 0 else 95.0
V_H = A[1] if len(A) > 1 else 6.0; V_UP = A[2] if len(A) > 2 else 5.0; V_DN = A[3] if len(A) > 3 else 3.0
V_LAND = 0.5
WP1 = np.array([0, 0, 50.0]); WP2 = np.array([0, 0, 500.0]); WP3 = np.array([0, 300, 800.0])
WP4 = np.array([0, -100, 1300.0]); WP5 = np.array([0, 340, 400.0]); WP6 = np.array([0, -410, 1500.0])
R_C, SIDE = 100.0, 120.0


def build(dt):
    P, S, names = [], [], []
    pos = np.zeros(3)
    def seg(name):
        names.append(name); return len(names) - 1
    def goto(p, name, vdn=V_DN):
        nonlocal pos
        d = np.asarray(p, float) - pos
        T = max(np.hypot(d[0], d[1]) / V_H, d[2] / V_UP if d[2] > 0 else -d[2] / vdn, dt)
        n = int(np.ceil(T / dt)); k = seg(name)
        for i in range(1, n + 1): P.append(pos + d * i / n); S.append(k)
        pos = pos + d
    def hold(sec, name):
        k = seg(name)
        for _ in range(int(sec / dt)): P.append(pos.copy()); S.append(k)
    def circle(c, name):
        nonlocal pos
        n = int(2 * np.pi * R_C / V_H / dt); k = seg(name)
        for i in range(1, n + 1):
            th = 2 * np.pi * i / n; P.append(np.array([c[0] + R_C * np.cos(th), c[1] + R_C * np.sin(th), c[2]])); S.append(k)
        pos = P[-1].copy()
    goto(WP1, "take-off to 50 m"); hold(30, "WP1 hover 30 s")
    goto(WP2, "to WP2"); hold(8, "WP2 hold")
    goto(WP3, "to WP3"); hold(8, "WP3 hold")
    goto(WP4, "to WP4"); hold(8, "WP4 hold")
    goto(WP5, "to WP5"); hold(8, "WP5 hold")
    goto(WP5 + [R_C, 0, 0], "to circle edge"); hold(6, "circle edge hold")
    circle(WP5, "circle lap"); hold(6, "after circle hold")
    goto(WP6, "to WP6"); hold(8, "WP6 hold")
    c = [WP6 + [SIDE, 0, 0], WP6 + [SIDE, SIDE, 0], WP6 + [0, SIDE, 0], WP6]
    for i, p in enumerate(c): goto(p, f"rectangle side {i+1}"); hold(4, f"rectangle corner {i+1} hold")
    goto([0, 0, 10.0], "return to take-off point"); hold(8, "hold 10 m above take-off point")
    goto([0, 0, -3.0], "landing descent", vdn=V_LAND)
    return np.array(P), np.array(S), names


def rect_dist(p):
    """horizontal distance to the rectangle outline"""
    u = p[..., 0] - WP6[0]; v = p[..., 1] - WP6[1]
    dx = np.maximum(np.maximum(-u, u - SIDE), 0); dy = np.maximum(np.maximum(-v, v - SIDE), 0)
    outside = np.hypot(dx, dy)
    inside = np.minimum(np.minimum(u, SIDE - u), np.minimum(v, SIDE - v))
    return np.where((dx > 0) | (dy > 0), outside, np.abs(inside))



PWM_OFF, PWM_IDLE, PWM_SPOOL = 1000.0, 1150.0, 1450.0     # disarmed, armed idle, end of spool-up (below hover)
T_OFF, T_IDLE, T_SPOOL = 1.0, 2.0, 2.0                    # seconds in each pre-flight phase
T_CONFIRM, T_RAMP, T_IDLE_END = 1.0, 1.0, 2.0             # land confirm, ramp to idle, idle before disarm
T_TAIL = 12.0


def fly(n, randomize, seed, record=False):
    env = MultirotorEnv(n, randomize=randomize, noise=True, seed=seed, err_clip=2.5, mass=MASS, alt0=0.0, ground=True)
    ag = PPO(env.obs_dim, env.act_dim); ag.load(weights)
    env.reset_idx(np.arange(n), hard=False)
    env.mot[:] = 0.0; env.prev_a[:] = -1.0; env.on_ground[:] = True      # motors stopped, sitting on the ground
    dt = float(env.dt_env[0]); P0, S0, names = build(dt)
    land_seg = names.index("landing descent")
    pre = np.concatenate([np.full(int(T_OFF / dt), PWM_OFF), np.full(int(T_IDLE / dt), PWM_IDLE),
                          np.linspace(PWM_IDLE, PWM_SPOOL, int(T_SPOOL / dt))])
    n_pre = len(pre); n_tail = int(T_TAIL / dt)
    P = np.concatenate([np.zeros((n_pre, 3)), P0, np.repeat(P0[-1:], n_tail, 0)])
    S = np.concatenate([np.full(n_pre, -1), S0, np.full(n_tail, land_seg)])
    steps = len(P); err = np.zeros((steps, n), np.float32); pos = np.zeros((steps, n, 3), np.float32)
    crashed = np.zeros(n, bool)
    phase = np.zeros(n, int)            # 0 network flying, 1 ramp to idle, 2 idle, 3 disarmed
    t_phase = np.zeros(n); pwm_at_detect = np.zeros((n, env.M)); confirm = np.zeros(n)
    t_liftoff = np.full(n, np.nan); t_td = np.full(n, np.nan); t_detect = np.full(n, np.nan); t_disarm = np.full(n, np.nan)
    td_off = np.full(n, np.nan); moved_pre = np.zeros(n)
    rec = dict(t=[], p=[], tgt=[], pwm=[], rpy=[], vz=[]); pwm_full = np.zeros((steps, env.M), np.float32)
    for k in range(steps):
        t = (k + 1) * dt
        env.target[:] = P[k]; env.target_yaw[:] = 0.0
        a, _ = ag.act(env.observe(), deterministic=True)
        a = np.clip(a, -1, 1).astype(np.float64)
        if k < n_pre:                                              # pre-flight: fixed PWM, network not in control
            a[:] = (pre[k] - 1500.0) / 500.0
        else:
            ramp = phase == 1
            if ramp.any():
                f = np.clip((t - t_phase[ramp]) / T_RAMP, 0, 1)[:, None]
                a[ramp] = ((1 - f) * pwm_at_detect[ramp] + f * PWM_IDLE - 1500.0) / 500.0
            a[phase == 2] = (PWM_IDLE - 1500.0) / 500.0
            a[phase == 3] = (PWM_OFF - 1500.0) / 500.0
        _, _, _, _, _, info = env.step(a, auto_target=False)
        if k < n_pre: moved_pre = np.maximum(moved_pre, np.linalg.norm(env.p, axis=1))
        up = 1 - 2 * (env.q[:, 1] ** 2 + env.q[:, 2] ** 2)
        crashed |= (up < 0) | (np.linalg.norm(env.v, axis=1) > 40) | (env.impact > 2.0)
        lift = np.isnan(t_liftoff) & (env.p[:, 2] > 0.05); t_liftoff[lift] = t
        if S[k] == land_seg:
            td = np.isnan(t_td) & env.on_ground
            t_td[td] = t; td_off[td] = np.hypot(env.p[td, 0], env.p[td, 1])
            # land detector: not moving vertically while the target is well below, for T_CONFIRM,
            # and with the network already commanding less than lift-off throttle (so a slow drone
            # that is still hovering in the air can never be taken for landed)
            cond = ((phase == 0) & (np.abs(env.v[:, 2]) < 0.15) & (P[k, 2] < env.p[:, 2] - 0.5)
                    & (info["pwm"].mean(1) < PWM_SPOOL))
            confirm = np.where(cond, confirm + dt, 0.0)
            det = (phase == 0) & (confirm >= T_CONFIRM)
            phase[det] = 1; t_phase[det] = t; t_detect[det] = t; pwm_at_detect[det] = info["pwm"][det]
            to_idle = (phase == 1) & (t - t_phase >= T_RAMP); phase[to_idle] = 2; t_phase[to_idle] = t
            off = (phase == 2) & (t - t_phase >= T_IDLE_END); phase[off] = 3; t_disarm[off] = t
        err[k] = info["dist"]; pos[k] = env.p
        if record:
            pwm_full[k] = info["pwm"][0]
            if k % 5 == 0:
                r, p, y = euler_from_quat(env.q)
                rec["t"].append(t); rec["p"].append(env.p[0].copy()); rec["tgt"].append(P[k].copy())
                rec["pwm"].append(info["pwm"][0].copy()); rec["rpy"].append([r[0], p[0]]); rec["vz"].append(env.v[0, 2])
    return dict(dt=dt, P=P, S=S, names=names, err=err, pos=pos, crashed=crashed, landed=~np.isnan(t_td), disarmed=phase == 3,
                impact=env.impact.copy(), td_off=td_off, t_liftoff=t_liftoff, t_td=t_td, t_detect=t_detect, t_disarm=t_disarm,
                moved_pre=moved_pre, pwm_full=pwm_full, n_pre=n_pre, rec={k: np.array(v) for k, v in rec.items()})


def summarise(r, i=None):
    """i = drone index for per-drone numbers, None = median over drones"""
    S, names, err, pos, dt = r["S"], r["names"], r["err"], r["pos"], r["dt"]
    pick = (lambda x: float(x[i])) if i is not None else (lambda x: float(np.median(x)))
    out = {}
    for name in ["WP1 hover 30 s", "WP2 hold", "WP3 hold", "WP4 hold", "WP5 hold", "WP6 hold", "hold 10 m above take-off point"]:
        m = np.nonzero(S == names.index(name))[0]; last = m[-int(2.0 / dt):]
        out[name + ": error at end (m)"] = pick(err[last].mean(0))
    m = S == names.index("WP1 hover 30 s")
    out["WP1 hover: max drift after first 10 s (m)"] = pick(err[np.nonzero(m)[0][int(10 / dt):]].max(0))
    m = S == names.index("circle lap")
    rad = np.abs(np.hypot(pos[m, :, 0] - WP5[0], pos[m, :, 1] - WP5[1]) - R_C)
    out["circle: mean / max distance from line (m)"] = [pick(rad.mean(0)), pick(rad.max(0))]
    out["circle: max altitude error (m)"] = pick(np.abs(pos[m, :, 2] - WP5[2]).max(0))
    m = np.array([n.startswith("rectangle") for n in names])[S]
    rd = rect_dist(pos[m])
    out["rectangle: mean / max distance from outline (m)"] = [pick(rd.mean(0)), pick(rd.max(0))]
    mv = np.array([n.startswith("to ") or n.startswith("return") for n in names])[S]
    out["legs: mean / max distance behind moving target (m)"] = [pick(err[mv].mean(0)), pick(err[mv].max(0))]
    return out


nom = fly(1, False, 1, record=True)
import os
N_ROB = int(os.environ.get("MISSION_N", "100"))      # number of randomised drones (CI uses fewer)
rob = fly(N_ROB, True, 2)
pf = nom["pwm_full"]; dtn = nom["dt"]; tf = np.arange(1, len(pf) + 1) * dtn
T0 = T_OFF + T_IDLE + T_SPOOL
def at(t0, t1): return pf[(tf > t0) & (tf <= t1)]
res = dict(mass_kg=MASS, speeds_mps=dict(horizontal=V_H, climb=V_UP, descent=V_DN, final_landing=V_LAND),
           total_time_min=float(nom["t_disarm"][0] / 60),
           pwm_sequence_standard_drone={
               "1 disarmed (0-1 s)": float(at(0, T_OFF).mean()),
               "2 armed idle (1-3 s)": float(at(T_OFF, T_OFF + T_IDLE).mean()),
               "3 spool-up ramp (3-5 s) start/end": [float(at(T_OFF + T_IDLE, T0)[0].mean()), float(at(T_OFF + T_IDLE, T0)[-1].mean())],
               "4 network takes over: first output per motor": [float(x) for x in at(T0, T0 + 1)[0]],
               "5 lift-off time (s)": float(nom["t_liftoff"][0]),
               "5 PWM at lift-off (mean)": float(pf[int(nom["t_liftoff"][0] / dtn) - 1].mean()),
               "6 take-off peak PWM": float(at(T0, T0 + 6).max()),
               "7 hover at 50 m (mean)": float(pf[nom["S"] == nom["names"].index("WP1 hover 30 s")].mean()),
               "8 final descent before touchdown (mean)": float(at(nom["t_td"][0] - 10, nom["t_td"][0]).mean()),
               "9 at touchdown per motor": [float(x) for x in pf[int(nom["t_td"][0] / dtn) - 1]],
               "10 lowest PWM while pressing down before detection": float(at(nom["t_td"][0], nom["t_detect"][0]).min()),
               "11 at land detection per motor": [float(x) for x in pf[int(nom["t_detect"][0] / dtn) - 1]],
               "12 idle after ramp-down": float(at(nom["t_detect"][0] + T_RAMP + 0.2, nom["t_disarm"][0] - 0.1).mean()),
               "13 disarmed": float(at(nom["t_disarm"][0] + 0.1, tf[-1]).mean())},
           standard=dict(crashed=bool(nom["crashed"][0]), touchdown_speed_mps=float(nom["impact"][0]),
                         touchdown_offset_m=float(nom["td_off"][0]), moved_before_takeoff_m=float(nom["moved_pre"][0]),
                         touchdown_to_detect_s=float(nom["t_detect"][0] - nom["t_td"][0]),
                         touchdown_to_disarm_s=float(nom["t_disarm"][0] - nom["t_td"][0]), **summarise(nom, 0)),
           randomised_100=dict(crashed=int(rob["crashed"].sum()), landed=int(rob["landed"].sum()), disarmed=int(rob["disarmed"].sum()),
                               lifted_off=int((~np.isnan(rob["t_liftoff"])).sum()),
                               liftoff_time_min_max_s=[float(np.nanmin(rob["t_liftoff"])), float(np.nanmax(rob["t_liftoff"]))],
                               moved_before_takeoff_max_m=float(rob["moved_pre"].max()),
                               touchdown_speed_median_max_mps=[float(np.median(rob["impact"])), float(rob["impact"].max())],
                               touchdown_offset_median_max_m=[float(np.nanmedian(rob["td_off"])), float(np.nanmax(rob["td_off"]))],
                               touchdown_to_disarm_median_max_s=[float(np.nanmedian(rob["t_disarm"] - rob["t_td"])), float(np.nanmax(rob["t_disarm"] - rob["t_td"]))],
                               **summarise(rob)))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{tag}_results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]; INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC,
    "ytick.color": SEC, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK})
rec = nom["rec"]; t = rec["t"] / 60
fig, ax0 = plt.subplots(6, 1, figsize=(7, 28), gridspec_kw=dict(height_ratios=[1.6, 1.2, 1.3, 1.3, 1, 1]))
SEC2 = SEC; p_ = rec["p"]; tg_ = rec["tgt"]
a = ax0[0]; a.plot(tg_[:, 1], tg_[:, 2], "--", color=SEC, lw=1, label="Planned"); a.plot(p_[:, 1], p_[:, 2], color=C4[0], lw=2, label="Flown")
for nme, w in [("WP1", WP1), ("WP2", WP2), ("WP3", WP3), ("WP4", WP4), ("WP5", WP5), ("WP6", WP6)]:
    a.scatter([w[1]], [w[2]], s=45, color=BG, edgecolor=SEC, zorder=3); a.annotate(nme, (w[1], w[2]), textcoords="offset points", xytext=(8, 6), color=SEC)
a.invert_xaxis(); a.set_xlabel("← left     sideways position (m)     right →"); a.set_ylabel("height (m)")
a.set_title("Side view: height against sideways position"); a.legend(frameon=False, loc="center left")
a = ax0[1]; a.plot(tg_[:, 1], tg_[:, 0], "--", color=SEC, lw=1, label="Planned"); a.plot(p_[:, 1], p_[:, 0], color=C4[0], lw=2, label="Flown")
a.invert_xaxis(); a.set_aspect("equal"); a.set_xlabel("← left     sideways position (m)     right →"); a.set_ylabel("forward (m)")
a.set_title("Top view: circle at WP5 (left) and rectangle at WP6 (right)"); a.legend(frameon=False, loc="lower center", ncol=2)
ax = ax0[2:]
def panel(a, t0, t1, title, marks):
    m = (tf >= t0) & (tf <= t1)
    for i in range(pf.shape[1]): a.plot(tf[m] - t0, pf[m, i], color=C4[i], lw=1.3, label=f"Motor {i+1}")
    a.set_ylim(950, 1900); a.set_title(title); a.set_xlabel("Time (s)")
    for x, lab in marks:
        a.axvline(x - t0, color=SEC, lw=0.8, ls=":"); a.text(x - t0 + 0.08, 1880, lab, rotation=90, va="top", ha="left", color=SEC, fontsize=9)
    a.legend(frameon=False, ncol=4 if pf.shape[1] <= 4 else 3, loc="lower center")
panel(ax[0], 0, 14, "Start: disarmed to take-off, actual PWM per motor (µs)",
      [(0, "disarmed"), (T_OFF, "armed, idle"), (T_OFF + T_IDLE, "spool-up"), (T0, "network in control"), (nom["t_liftoff"][0], "lift-off")])
td, de, di = nom["t_td"][0], nom["t_detect"][0], nom["t_disarm"][0]
panel(ax[1], td - 6, di + 2, "End: landing to disarm, actual PWM per motor (µs)",
      [(td, "touchdown"), (de, "landing detected"), (de + T_RAMP, "idle"), (di, "disarmed")])
a = ax[2]
for i in range(rec["pwm"].shape[1]): a.plot(t, rec["pwm"][:, i], color=C4[i], lw=0.8, label=f"Motor {i+1}")
a.set_ylim(950, 1900); a.set_title("Whole mission: actual PWM per motor (µs)"); a.set_xlabel("Time (minutes)"); a.legend(frameon=False, ncol=4 if pf.shape[1] <= 4 else 3, loc="lower center")
a = ax[3]; a.plot(t, rec["tgt"][:, 2], "--", color=SEC, lw=1, label="Target"); a.plot(t, rec["p"][:, 2], color=C4[0], lw=2, label="Actual")
a.set_title("Height (m)"); a.set_xlabel("Time (minutes)"); a.legend(frameon=False, loc="upper left")
fig.tight_layout(h_pad=2); fig.savefig(f"{tag}_results.png", dpi=130)
