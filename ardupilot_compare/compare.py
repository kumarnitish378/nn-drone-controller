"""Extract the ArduCopter SITL DataFlash log and compare it with the neural-network logs on the same mission.

usage: python3 compare.py out_dir
reads  out_dir/ardupilot_sitl.BIN, out_dir/network_quad_x.npz [, out_dir/network_hex_x.npz]
writes out_dir/ardupilot_extracted.csv, out_dir/comparison.json, out_dir/comparison.png
"""
import json, math, os, sys
import numpy as np
from pymavlink import mavutil
from mission_def import HOME, PARAMS

out = sys.argv[1]
WP2, WP3, CEN, R_C = np.array([60, 0, 30.0]), np.array([60, 60, 40.0]), np.array([0, 60, 30.0]), 15.0


# ------------------------------------------------------------------ ArduPilot log extraction
def read_bin(path):
    m = mavutil.mavlink_connection(path)
    pos, att, rcou, modes, arm, msgs, ctun, bat, sim, nsim = [], [], [], [], [], [], [], [], [], 0
    while True:
        x = m.recv_match(type=["POS", "ATT", "RCOU", "MODE", "ARM", "MSG", "CTUN", "BAT", "SIM2"])
        if x is None:
            break
        t = x.TimeUS * 1e-6; k = x.get_type()
        if k == "SIM2":                                  # the simulator's true position, 400 Hz; keep 50 Hz
            nsim += 1
            if nsim % 8 == 0: sim.append([t, x.PN, x.PE, -x.PD])
        elif k == "POS":
            pos.append([t, x.Lat, x.Lng, x.RelHomeAlt])
        elif k == "ATT":
            att.append([t, x.Roll, x.Pitch, x.Yaw])
        elif k == "RCOU":
            rcou.append([t, x.C1, x.C2, x.C3, x.C4])
        elif k == "MODE":
            modes.append([t, x.ModeNum])
        elif k == "ARM":
            arm.append([t, x.ArmState])
        elif k == "MSG":
            msgs.append([t, x.Message])
        elif k == "CTUN":
            ctun.append([t, x.ThO, x.ThH])
        elif k == "BAT":
            bat.append([t, x.Volt])
    A = lambda a: np.array(a, float)
    return dict(pos=A(pos), att=A(att), rcou=A(rcou), modes=modes, arm=A(arm), msgs=msgs, ctun=A(ctun), bat=A(bat), sim=A(sim))


def ardupilot_run(path):
    d = read_bin(path)
    # ArduPilot starts this log at arming, so the first sample is the arm time unless an ARM record says otherwise
    armed = d["arm"][d["arm"][:, 1] > 0.5] if len(d["arm"]) else []
    t_arm = armed[0, 0] if len(armed) else d["rcou"][0, 0]
    dis = d["arm"][(d["arm"][:, 1] < 0.5) & (d["arm"][:, 0] > t_arm)] if len(d["arm"]) else []
    t_dis = dis[0, 0] if len(dis) else d["pos"][-1, 0]
    # true position from the simulator (SIM2), so both sides are measured the same way; EKF estimate kept for reference
    P = d["sim"]; P = P[(P[:, 0] >= t_arm) & (P[:, 0] <= t_dis)]
    n = P[:, 1] - P[0, 1]; e = P[:, 2] - P[0, 2]; alt_true = P[:, 3] - P[0, 3]
    E = d["pos"]; E = E[(E[:, 0] >= t_arm) & (E[:, 0] <= t_dis)]
    en = (E[:, 1] - HOME[0]) * 111319.5; ee = (E[:, 2] - HOME[1]) * 111319.5 * math.cos(math.radians(HOME[0]))
    ekf_diff = float(np.max(np.hypot(np.interp(E[:, 0], P[:, 0], n) - (en - en[0]), np.interp(E[:, 0], P[:, 0], e) - (ee - ee[0]))))
    A = d["att"]; A = A[(A[:, 0] >= t_arm) & (A[:, 0] <= t_dis)]
    Rr = d["rcou"]; Rr = Rr[(Rr[:, 0] >= t_arm) & (Rr[:, 0] <= t_dis)]
    C = d["ctun"]; C = C[(C[:, 0] >= t_arm) & (C[:, 0] <= t_dis)]
    return dict(name="ArduCopter 4.7.2 (PID)", t=P[:, 0] - t_arm, n=n, e=e, alt=alt_true, ekf_diff=ekf_diff,
                t_att=A[:, 0] - t_arm, roll=A[:, 1], pitch=A[:, 2], t_pwm=Rr[:, 0] - t_arm, pwm=Rr[:, 1:],
                duration=t_dis - t_arm, hover_thr=float(np.median(C[:, 2])) if len(C) else None,
                rates=dict(position_hz=len(P) / (t_dis - t_arm), pwm_hz=len(Rr) / (t_dis - t_arm), attitude_hz=len(A) / (t_dis - t_arm)),
                msgs=[m for _, m in d["msgs"]][:12])


def network_run(path, name):
    z = np.load(path); D = z["data"].astype(float); M = D.shape[1] - 8
    return dict(name=name, t=D[:, 0], n=D[:, 1], e=D[:, 2], alt=D[:, 3], t_att=D[:, 0], roll=D[:, 4], pitch=D[:, 5],
                t_pwm=D[:, 0], pwm=D[:, 8:8 + M], duration=D[-1, 0], hover_thr=None,
                rates=dict(position_hz=len(D) / D[-1, 0], pwm_hz=len(D) / D[-1, 0], attitude_hz=len(D) / D[-1, 0]), msgs=[])


# ------------------------------------------------------------------ planned route as dense points
def plan_points(step=0.2):
    edge = CEN + np.array([R_C, 0, 0])                      # closest point of the circle to WP3 (north side)
    def seg(a, b):
        k = max(2, int(np.linalg.norm(np.array(b) - a) / step)); return [np.array(a) + (np.array(b) - a) * i / k for i in range(k + 1)]
    pts = seg([0, 0, 0], [0, 0, 20]) + seg([0, 0, 20], WP2) + seg(WP2, WP3) + seg(WP3, edge)
    pts += [CEN + np.array([R_C * math.cos(a), R_C * math.sin(a), 0]) for a in np.linspace(0, 2 * math.pi, int(2 * math.pi * R_C / step))]
    pts += seg(edge, [0, 0, 30]) + seg([0, 0, 30], [0, 0, 0])
    return np.array(pts)

PLAN = plan_points()

def path_error(n, e, alt):
    X = np.stack([n, e, alt], 1); err = np.empty(len(X))
    for i in range(0, len(X), 500):
        d = X[i:i + 500, None, :] - PLAN[None, :, :]; err[i:i + 500] = np.sqrt((d ** 2).sum(2)).min(1)
    return err


def resample(t, x, hz):
    tt = np.arange(t[0], t[-1], 1.0 / hz)
    return tt, np.stack([np.interp(tt, t, x[:, i]) for i in range(x.shape[1])], 1)


def metrics(r):
    t, n, e, alt = r["t"], r["n"], r["e"], r["alt"]
    perr = path_error(n, e, alt)
    air = alt > 0.5
    # hover: the 30 s hold at waypoint 3, taken as the time spent within 2.5 m of it; use the middle 20 s
    d3 = np.sqrt((n - WP3[0]) ** 2 + (e - WP3[1]) ** 2 + (alt - WP3[2]) ** 2)
    inside = np.nonzero(d3 < 2.5)[0]; t_in, t_out = t[inside[0]], t[inside[-1]]
    h0, h1 = t_in + 6, min(t_in + 26, t_out - 1)
    hv = (t >= h0) & (t <= h1)
    # circle: after the hold, while near 30 m and near the ring
    rad = np.hypot(n - CEN[0], e - CEN[1])
    cir = (t > t_out) & (np.abs(rad - R_C) < 3) & (np.abs(alt - 30) < 3)
    # longest unbroken stretch on the ring, minus 2 s at each end (joining and leaving the circle)
    idx = np.nonzero(cir)[0]; breaks = np.nonzero(np.diff(idx) > 1)[0]
    runs_ = np.split(idx, breaks + 1); ci = max(runs_, key=len)
    ci = ci[(t[ci] > t[ci[0]] + 2) & (t[ci] < t[ci[-1]] - 2)]
    # 10 Hz PWM for a like-for-like look at the motor outputs
    tp, pw = resample(r["t_pwm"], r["pwm"], 10.0)
    ph = pw[(tp >= h0) & (tp <= h1)]
    fly = pw[(tp > 8) & (tp < r["duration"] - 8)]
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(r["roll"])) * np.cos(np.radians(r["pitch"])), -1, 1)))
    spd = np.hypot(np.gradient(n, t), np.gradient(e, t))
    return {
        "mission time (s)": float(r["duration"]),
        "route error mean (m)": float(perr[air].mean()),
        "route error 95th percentile (m)": float(np.percentile(perr[air], 95)),
        "route error max (m)": float(perr[air].max()),
        "hover horizontal error mean (m)": float(np.hypot(n[hv] - WP3[0], e[hv] - WP3[1]).mean()),
        "hover horizontal wander, std (m)": float(np.sqrt(n[hv].var() + e[hv].var())),
        "hover height error mean (m)": float(np.abs(alt[hv] - WP3[2]).mean()),
        "hover height wander, std (m)": float(alt[hv].std()),
        "highest point above the 40 m target (m)": float(alt.max() - 40.0),
        "circle radius error mean (m)": float(np.abs(rad[ci] - R_C).mean()),
        "circle radius error max (m)": float(np.abs(rad[ci] - R_C).max()),
        "circle time on the ring (s)": float(t[ci[-1]] - t[ci[0]] + 4),
        "circle height error max (m)": float(np.abs(alt[ci] - 30).max()),
        "landing distance from take-off point (m)": float(np.hypot(n[-1], e[-1])),
        "steepest tilt (deg)": float(tilt.max()),
        "top horizontal speed (m/s)": float(np.percentile(spd, 99.5)),
        "PWM in hover, mean (us)": float(ph.mean()),
        "PWM in hover, spread between motors (us)": float(ph.mean(0).max() - ph.mean(0).min()),
        "PWM in hover, wander std per motor (us)": float(ph.std(0).mean()),
        "PWM in hover, mean change per 0.1 s (us)": float(np.abs(np.diff(ph, axis=0)).mean()),
        "PWM in flight, lowest (us)": float(fly.min()),
        "PWM in flight, highest (us)": float(fly.max()),
        "PWM in flight, mean change per 0.1 s (us)": float(np.abs(np.diff(fly, axis=0)).mean()),
        "_hover_window": [float(h0), float(h1)], "_perr": perr, "_pwm10": (tp, pw),
    }


runs = [ardupilot_run(os.path.join(out, "ardupilot_sitl.BIN")), network_run(os.path.join(out, "network_quad_x.npz"), "Neural network, quad")]
if os.path.exists(os.path.join(out, "network_hex_x.npz")):
    runs.append(network_run(os.path.join(out, "network_hex_x.npz"), "Neural network, hexa"))
n_chart = len(runs)
if os.path.exists(os.path.join(out, "network_quad_x_nonoise.npz")):                 # same quad network, sensor noise switched off
    runs.append(network_run(os.path.join(out, "network_quad_x_nonoise.npz"), "Quad network, no noise"))
M = [metrics(r) for r in runs]

# extracted ArduPilot data as CSV (10 Hz): time, position, attitude, motor PWM
ap = runs[0]; tt = np.arange(0, ap["duration"], 0.1)
cols = [tt, np.interp(tt, ap["t"], ap["n"]), np.interp(tt, ap["t"], ap["e"]), np.interp(tt, ap["t"], ap["alt"]),
        np.interp(tt, ap["t_att"], ap["roll"]), np.interp(tt, ap["t_att"], ap["pitch"])] + [np.interp(tt, ap["t_pwm"], ap["pwm"][:, i]) for i in range(4)]
np.savetxt(os.path.join(out, "ardupilot_extracted.csv"), np.stack(cols, 1), delimiter=",", fmt="%.3f",
           header="t_s,north_m,east_m,height_m,roll_deg,pitch_deg,pwm1_us,pwm2_us,pwm3_us,pwm4_us", comments="")

keys = [k for k in M[0] if not k.startswith("_")]
res = dict(mission="take-off 20 m, waypoint 60 m north at 30 m, 30 s hold 60 m north-east at 40 m, 15 m circle at 30 m, return and land",
           settings=PARAMS, ardupilot_ekf_vs_true_position_max_m=runs[0]["ekf_diff"], log_rates_hz={r["name"]: r["rates"] for r in runs}, ardupilot_hover_throttle=runs[0]["hover_thr"],
           metrics={r["name"]: {k: round(m[k], 3) for k in keys} for r, m in zip(runs, M)})
json.dump(res, open(os.path.join(out, "comparison.json"), "w"), indent=1)
w = max(len(k) for k in keys)
print(f"{'':{w}}  " + "  ".join(f"{r['name'][:22]:>22}" for r in runs))
for k in keys:
    print(f"{k:{w}}  " + "  ".join(f"{m[k]:22.3f}" for m in M))
print("log rates (Hz):", {k: {a: round(float(b), 1) for a, b in v.items()} for k, v in res["log_rates_hz"].items()}); print("ArduPilot EKF vs true position, max (m):", round(runs[0]["ekf_diff"], 3)); print("ArduPilot hover throttle (CTUN.ThH):", runs[0]["hover_thr"])

# ------------------------------------------------------------------ chart
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = ["#eb6834", "#2a78d6", "#1baf7a"]; MC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})
runs, M = runs[:n_chart], M[:n_chart]
nr = 4 + len(runs)
fig, ax = plt.subplots(nr, 1, figsize=(7, 4.6 + 3.3 * (nr - 1)), gridspec_kw=dict(height_ratios=[1.6] + [1] * (nr - 1)))
a = ax[0]; a.plot(PLAN[:, 1], PLAN[:, 0], "--", color=SEC, lw=1, label="Planned route")
for r, c in zip(runs, COL): a.plot(r["e"], r["n"], color=c, lw=1.6, label=r["name"])
a.set_aspect("equal"); a.set_xlabel("East (m)"); a.set_ylabel("North (m)"); a.set_title("Top view"); a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42))
a = ax[1]
for r, c in zip(runs, COL): a.plot(r["t"], r["alt"], color=c, lw=1.6, label=r["name"])
a.set_title("Height (m)"); a.legend(frameon=False, loc="lower center", ncol=1)
a = ax[2]
for r, m, c in zip(runs, M, COL): a.plot(r["t"], m["_perr"], color=c, lw=1.2, label=r["name"])
a.set_title("Distance from the planned route (m)"); a.set_ylim(0, None); a.legend(frameon=False, loc="upper right")
a = ax[3]
for r, m, c in zip(runs, M, COL):
    tp, pw = m["_pwm10"]; a.plot(tp, pw.mean(1), color=c, lw=1.2, label=r["name"])
a.set_title("Average motor PWM, whole flight (µs)"); a.set_ylim(1000, 2000); a.legend(frameon=False, loc="lower center", ncol=1)
for j, (r, m) in enumerate(zip(runs, M)):
    a = ax[4 + j]; tp, pw = m["_pwm10"]; h0, h1 = m["_hover_window"]; s = (tp >= h0) & (tp <= h0 + 10)
    for i in range(pw.shape[1]): a.plot(tp[s] - h0, pw[s, i], color=MC[i], lw=1.3, label=f"Motor {i+1}")
    lo, hi = pw[s].min(), pw[s].max(); mid = (lo + hi) / 2; half = max(60, (hi - lo) / 2 + 15)
    a.set_ylim(mid - half - 25, mid + half); a.set_title(f"{r['name']}: each motor's PWM during 10 s of the hover (µs)")
    a.legend(frameon=False, ncol=3 if pw.shape[1] > 4 else 4, loc="lower center")
ax[-1].set_xlabel("Time (s)")
fig.tight_layout(h_pad=1.6); fig.savefig(os.path.join(out, "comparison.png"), dpi=130)
