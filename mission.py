"""Long mission for the 95 kg drone. Axes: x forward, y left, z up (metres from the take-off point).
The target point moves along straight lines at set speeds (like an autopilot's waypoint navigation)."""
import sys, json
import numpy as np
from sim import MultirotorEnv, euler_from_quat
from ppo import PPO

weights = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "mission95"
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


def fly(n, randomize, seed, record=False):
    env = MultirotorEnv(n, randomize=randomize, noise=True, seed=seed, err_clip=2.5, mass=MASS, alt0=0.0)
    ag = PPO(env.obs_dim, env.act_dim); ag.load(weights)
    env.reset_idx(np.arange(n), hard=False)
    dt = float(env.dt_env[0]); P, S, names = build(dt)
    steps = len(P); err = np.zeros((steps, n), np.float32); pos = np.zeros((steps, n, 3), np.float32)
    crashed = np.zeros(n, bool); landed = np.zeros(n, bool)
    td_vz = np.full(n, np.nan); td_off = np.full(n, np.nan); td_t = np.full(n, np.nan)
    min_z_takeoff = np.zeros(n); land_seg = names.index("landing descent")
    rec = dict(t=[], p=[], tgt=[], pwm=[], rpy=[], vz=[]); pwm_full = np.zeros((steps, env.M), np.float32)
    for k in range(steps):
        env.target[:] = P[k]; env.target_yaw[:] = 0.0
        a, _ = ag.act(env.observe(), deterministic=True)
        _, _, _, _, _, info = env.step(a, auto_target=False)
        up = 1 - 2 * (env.q[:, 1] ** 2 + env.q[:, 2] ** 2)
        crashed |= ~landed & ((up < 0) | (np.linalg.norm(env.v, axis=1) > 40) | ((env.p[:, 2] < -3) & (S[k] != land_seg)))
        if S[k] == 0: min_z_takeoff = np.minimum(min_z_takeoff, env.p[:, 2])
        if S[k] == land_seg:
            td = ~landed & ~crashed & (env.p[:, 2] <= 0)
            td_vz[td] = -env.v[td, 2]; td_off[td] = np.hypot(env.p[td, 0], env.p[td, 1]); td_t[td] = (k + 1) * dt
            landed |= td
        err[k] = info["dist"]; pos[k] = env.p
        if record: pwm_full[k] = info["pwm"][0]
        if record and k % 5 == 0:
            r, p, y = euler_from_quat(env.q)
            rec["t"].append((k + 1) * dt); rec["p"].append(env.p[0].copy()); rec["tgt"].append(P[k].copy())
            rec["pwm"].append(info["pwm"][0].copy()); rec["rpy"].append([r[0], p[0]]); rec["vz"].append(env.v[0, 2])
    return dict(dt=dt, P=P, S=S, names=names, err=err, pos=pos, crashed=crashed, landed=landed, td_vz=td_vz,
                td_off=td_off, td_t=td_t, min_z=min_z_takeoff, pwm_full=pwm_full, rec={k: np.array(v) for k, v in rec.items()})


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
rob = fly(100, True, 2)
res = dict(mass_kg=MASS, speeds_mps=dict(horizontal=V_H, climb=V_UP, descent=V_DN, final_landing=V_LAND),
           mission_time_min=float(np.nanmax(nom["td_t"]) / 60) if nom["landed"][0] else None,
           standard=dict(crashed=bool(nom["crashed"][0]), landed=bool(nom["landed"][0]),
                         touchdown_speed_mps=float(nom["td_vz"][0]), touchdown_offset_m=float(nom["td_off"][0]),
                         lowest_point_during_takeoff_m=float(nom["min_z"][0]),
                         max_tilt_deg=float(np.rad2deg(np.abs(nom["rec"]["rpy"]).max())),
                         max_climb_mps=float(nom["rec"]["vz"].max()), max_descent_mps=float(-nom["rec"]["vz"].min()),
                         pwm_min=float(nom["pwm_full"].min()), pwm_max=float(nom["pwm_full"].max()),
                         pwm_hover_mean=float(nom["pwm_full"][nom["S"] == nom["names"].index("WP1 hover 30 s")].mean()),
                         pwm_step_change_mean_us=float(np.abs(np.diff(nom["pwm_full"], axis=0)).mean()), **summarise(nom, 0)),
           randomised_100=dict(crashed=int(rob["crashed"].sum()), landed=int(rob["landed"].sum()),
                               touchdown_speed_median_max_mps=[float(np.nanmedian(rob["td_vz"])), float(np.nanmax(rob["td_vz"]))],
                               touchdown_offset_median_max_m=[float(np.nanmedian(rob["td_off"])), float(np.nanmax(rob["td_off"]))],
                               lowest_point_during_takeoff_worst_m=float(rob["min_z"].min()), **summarise(rob)))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{tag}_results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834"]; INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC,
    "ytick.color": SEC, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK})
rec = nom["rec"]; t = rec["t"] / 60; p = rec["p"]; tg = rec["tgt"]
C4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, ax = plt.subplots(7, 1, figsize=(7, 29), gridspec_kw=dict(height_ratios=[1.6, 1.2, 1, 1, 1.1, 1.1, 1.1]))
a = ax[0]; a.plot(tg[:, 1], tg[:, 2], "--", color=SEC, lw=1, label="Planned"); a.plot(p[:, 1], p[:, 2], color=C[0], lw=2, label="Flown")
for nme, w in [("WP1", WP1), ("WP2", WP2), ("WP3", WP3), ("WP4", WP4), ("WP5", WP5), ("WP6", WP6)]:
    a.scatter([w[1]], [w[2]], s=45, color=BG, edgecolor=SEC, zorder=3); a.annotate(nme, (w[1], w[2]), textcoords="offset points", xytext=(8, 6), color=SEC)
a.invert_xaxis(); a.set_xlabel("← left     sideways position (m)     right →"); a.set_ylabel("height (m)")
a.set_title("Side view: height against sideways position"); a.legend(frameon=False, loc="center left")
a = ax[1]; a.plot(tg[:, 1], tg[:, 0], "--", color=SEC, lw=1, label="Planned"); a.plot(p[:, 1], p[:, 0], color=C[0], lw=2, label="Flown")
a.invert_xaxis(); a.set_aspect("equal"); a.set_xlabel("← left     sideways position (m)     right →"); a.set_ylabel("forward (m)")
a.set_title("Top view: circle at WP5 (left) and rectangle at WP6 (right)"); a.legend(frameon=False, loc="lower center", ncol=2)
a = ax[2]; a.plot(t, tg[:, 2], "--", color=SEC, lw=1, label="Target"); a.plot(t, p[:, 2], color=C[0], lw=2, label="Actual")
a.set_title("Height (m)"); a.legend(frameon=False, loc="upper left")
a = ax[3]; a.plot(t, np.linalg.norm(tg - p, axis=1), color=C[0], lw=1.5); a.set_title("Distance from the moving target point (m)")
a = ax[4]
for i in range(rec["pwm"].shape[1]): a.plot(t, rec["pwm"][:, i], color=C4[i], lw=0.8, label=f"Motor {i+1}")
a.set_ylim(1000, 2000); a.set_title("Actual PWM sent to each motor (µs), whole mission"); a.set_xlabel("Time (minutes)"); a.legend(frameon=False, ncol=4, loc="lower center")
pf = nom["pwm_full"]; dtn = nom["dt"]; tf = np.arange(1, len(pf) + 1) * dtn
def zoom(a, t0, t1, title):
    m = (tf >= t0) & (tf <= t1)
    for i in range(pf.shape[1]): a.plot(tf[m] - t0, pf[m, i], color=C4[i], lw=1.2, label=f"Motor {i+1}")
    a.set_title(title); a.set_xlabel("Time (s)"); a.legend(frameon=False, ncol=4, loc="lower center")
    lo, hi = pf[m].min(), pf[m].max(); a.set_ylim(max(1000, lo - 0.35 * (hi - lo) - 20), min(2000, hi + 20))
zoom(ax[5], 0, 12, "Actual PWM, zoom: first 12 s (take-off)")
k0 = np.nonzero(nom["S"] == nom["names"].index("circle lap"))[0][0] * dtn
zoom(ax[6], k0 - 3, k0 + 12, "Actual PWM, zoom: start of the circle lap")
fig.tight_layout(h_pad=2); fig.savefig(f"{tag}_results.png", dpi=130)
