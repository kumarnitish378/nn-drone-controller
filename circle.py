"""Circle mission: climb to the start point, fly one lap following a moving target, then hold."""
import sys, json
import numpy as np
from sim import MultirotorEnv, euler_from_quat
from ppo import PPO

weights = sys.argv[1]; ERR_CLIP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
tag = sys.argv[3] if len(sys.argv) > 3 else "circle"
# optional: radius m, speed m/s, mass kg, circle altitude above sea level m, seconds allowed to reach the circle
A = [float(x) for x in sys.argv[4:9]]
R = A[0] if len(A) > 0 else 5.0; SPEED = A[1] if len(A) > 1 else 1.5
MASS = A[2] if len(A) > 2 else 1.0; ASL = A[3] if len(A) > 3 else 3.0
ALT = 3.0 if len(A) <= 3 else 5.0          # climb this much from the start point to the circle height
ALT0 = ASL - ALT                           # start altitude above sea level
T_GOTO = A[4] if len(A) > 4 else 6.0; T_HOLD = 5.0 * max(1.0, MASS ** (1 / 6))
T_LAP = 2 * np.pi * R / SPEED


def fly(n, randomize, seed, record=False):
    env = MultirotorEnv(n, randomize=randomize, noise=True, seed=seed, err_clip=ERR_CLIP, mass=MASS, alt0=ALT0)
    ag = PPO(env.obs_dim, env.act_dim); ag.load(weights)
    env.reset_idx(np.arange(n), hard=False)
    dt = float(env.dt_env[0]); steps = int((T_GOTO + T_LAP + T_HOLD) / dt)
    rad = np.zeros((steps, n)); alt = np.zeros((steps, n)); lag = np.zeros((steps, n))
    crashed = np.zeros(n, bool); rec = dict(t=[], p=[], tgt=[], pwm=[], rpy=[])
    for k in range(steps):
        t = (k + 1) * dt
        th = np.clip((t - T_GOTO) / T_LAP, 0, 1) * 2 * np.pi      # angle along the circle
        env.target[:] = [R * np.cos(th), R * np.sin(th), ALT]
        env.target_yaw[:] = th + np.pi / 2                          # nose along direction of travel
        a, _ = ag.act(env.observe(), deterministic=True)
        _, _, c, _, _, info = env.step(a, auto_target=False)
        crashed |= c
        rad[k] = np.hypot(env.p[:, 0], env.p[:, 1]) - R; alt[k] = env.p[:, 2] - ALT; lag[k] = info["dist"]
        if record:
            r, p, y = euler_from_quat(env.q)
            rec["t"].append(t); rec["p"].append(env.p[0].copy()); rec["tgt"].append(env.target[0].copy())
            rec["pwm"].append(info["pwm"][0].copy()); rec["rpy"].append([r[0], p[0], y[0]])
    lap = slice(int(T_GOTO / dt), int((T_GOTO + T_LAP) / dt)); hold = slice(steps - int(2.0 / dt), steps)
    return dict(crashed=crashed, rad_mean=np.abs(rad[lap]).mean(0), rad_max=np.abs(rad[lap]).max(0),
                alt_max=np.abs(alt[lap]).max(0), lag_mean=lag[lap].mean(0), hold=lag[hold].mean(0),
                rec={k: np.array(v) for k, v in rec.items()}, rad=rad[:, 0], lagt=lag[:, 0])


nom = fly(1, False, 1, record=True); rob = fly(500 if MASS == 1.0 else 200, True, 2)
out = dict(radius_m=R, altitude_asl_m=ASL, mass_kg=MASS, speed_mps=SPEED, lap_time_s=round(T_LAP, 1),
           nominal_radial_error_mean_m=float(nom["rad_mean"][0]), nominal_radial_error_max_m=float(nom["rad_max"][0]),
           nominal_altitude_error_max_m=float(nom["alt_max"][0]), nominal_lag_behind_target_m=float(nom["lag_mean"][0]),
           nominal_final_hold_error_m=float(nom["hold"][0]),
           robust_crashed=int(rob["crashed"].sum()),
           robust_radial_error_mean_median_m=float(np.median(rob["rad_mean"])),
           robust_radial_error_max_p95_m=float(np.percentile(rob["rad_max"], 95)),
           robust_radial_error_max_worst_m=float(rob["rad_max"].max()),
           robust_altitude_error_max_p95_m=float(np.percentile(rob["alt_max"], 95)),
           max_tilt_deg=float(np.rad2deg(np.abs(nom["rec"]["rpy"][:, :2]).max())))
print(json.dumps(out, indent=1)); json.dump(out, open(f"{tag}_results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC,
    "ytick.color": SEC, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK})
rec = nom["rec"]; t = rec["t"]; p = rec["p"]
fig, ax = plt.subplots(4, 1, figsize=(7, 16), gridspec_kw=dict(height_ratios=[2.2, 1, 1, 1]))
a = ax[0]; th = np.linspace(0, 2 * np.pi, 200)
a.plot(R * np.cos(th), R * np.sin(th), "--", color=SEC, lw=1, label="Planned circle")
a.plot(p[:, 0], p[:, 1], color=C[0], lw=2, label="Flown path")
a.scatter([0], [0], s=50, color=BG, edgecolor=SEC, zorder=3); a.annotate("Start", (0, 0), textcoords="offset points", xytext=(-10, 10), color=SEC)
a.set_aspect("equal"); a.set_xlabel("x forward (m)"); a.set_ylabel("y left (m)")
a.set_title(f"Circle, top view (radius {R:.0f} m, {SPEED:g} m/s, {MASS:g} kg, {ASL:g} m)"); a.legend(frameon=False, loc="lower left")
a = ax[1]; a.plot(t, nom["rad"], color=C[0], lw=2); a.axvspan(T_GOTO, T_GOTO + T_LAP, color="#e8e7e2", alpha=0.5, lw=0)
lim = max(1.0, 1.2 * np.abs(nom['rad'][int(T_GOTO / float(t[1] - t[0])):]).max()); a.set_ylim(-lim, lim); a.set_title("Distance from the circle line (m), shaded = lap; + is outside")
a = ax[2]; a.plot(t, ALT0 + rec["tgt"][:, 2], "--", color=SEC, lw=1, label="Target"); a.plot(t, ALT0 + p[:, 2], color=C[0], lw=2, label="Actual")
a.set_title("Altitude above sea level (m)"); a.ticklabel_format(useOffset=False); a.legend(frameon=False, loc="lower right")
a = ax[3]
for i in range(rec["pwm"].shape[1]): a.plot(t, rec["pwm"][:, i], color=C[i], lw=1.2, label=f"Motor {i+1}")
a.set_ylim(1000, 2000); a.set_title("Network output: motor PWM (µs)"); a.set_xlabel("Time (s)"); a.legend(frameon=False, ncol=4, loc="lower center")
fig.tight_layout(h_pad=2); fig.savefig(f"{tag}_results.png", dpi=130)
