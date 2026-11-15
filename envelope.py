"""Envelope tests: (1) climb to 2000 m, (2) drone mass from 1 kg to 100 kg.
usage: python3 envelope.py weights.npz [err_clip] [tag]"""
import sys, json
import numpy as np
_argv = sys.argv; sys.argv = [_argv[0], _argv[1]]
import evaluate as ev
sys.argv = _argv
from sim import G

weights = sys.argv[1]; ev.weights = weights
ev.ERR_CLIP = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
tag = sys.argv[3] if len(sys.argv) > 3 else "envelope"
out = {}

# ---------- 1. climb from sea level to 2000 m, fly a 20 m square there, hold ----------
TOP = 2000.0
M_HI = [(0, 0, TOP, 0), (20, 0, TOP, 0), (20, 20, TOP, 90), (0, 20, TOP, 180), (0, 0, TOP, -90)]
nom = ev.fly(1, False, True, 1, max_t=600.0, record=True, mission=M_HI)
rob = ev.fly(200, True, True, 2, max_t=600.0, mission=M_HI)
rec = nom["rec"]; t, p, pwm = rec["t"], rec["p"], rec["pwm"].mean(1)
vz = np.gradient(p[:, 2], t)
lo = (p[:, 2] > 50) & (p[:, 2] < 150) & (vz > 1); hi = (p[:, 2] > 1850) & (p[:, 2] < 1950) & (vz > 1)
out["climb_2000m"] = dict(
    nominal_completed=bool(nom["ok"][0]), nominal_time_to_2000m_s=float(t[np.argmax(p[:, 2] > TOP - 1)]) if (p[:, 2] > TOP - 1).any() else None,
    nominal_mission_time_s=float(nom["done_t"][0]), nominal_hold_error_at_2000m_m=float(nom["hold"][0]),
    climb_rate_near_ground_mps=float(vz[lo].mean()), climb_rate_near_2000m_mps=float(vz[hi].mean()) if hi.any() else None,
    mean_pwm_climbing_near_ground=float(pwm[lo].mean()), mean_pwm_climbing_near_2000m=float(pwm[hi].mean()) if hi.any() else None,
    mean_pwm_hover_at_2000m=float(pwm[-20:].mean()), max_altitude_m=float(p[:, 2].max()),
    robust_n=200, robust_completed=int(rob["ok"].sum()), robust_crashed=int(rob["crashed"].sum()),
    robust_hold_error_median_m=float(np.nanmedian(rob["hold"])) if rob["ok"].any() else None)
print(json.dumps(out["climb_2000m"], indent=1), flush=True)

# ---------- 2. mass sweep: the standard square mission with scaled-up airframes ----------
MASSES = [1, 2, 5, 10, 25, 50, 100]
rows = []
for m in MASSES:
    r = ev.fly(200, True, True, 3, max_t=90.0, mass=float(m))
    rows.append(dict(mass_kg=m, completed=int(r["ok"].sum()), crashed=int(r["crashed"].sum()), n=200,
                     mission_time_median_s=float(np.nanmedian(r["done_t"])) if r["ok"].any() else None,
                     hold_error_median_m=float(np.nanmedian(r["hold"])) if np.isfinite(r["hold"]).any() else None,
                     max_tilt_p95_deg=float(np.percentile(r["max_tilt"], 95))))
    print(rows[-1], flush=True)
out["mass_sweep"] = rows
json.dump(out, open(f"{tag}_results.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = ["#2a78d6", "#eb6834"]; INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC,
    "ytick.color": SEC, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlecolor": INK})
fig, ax = plt.subplots(4, 1, figsize=(7, 14))
a = ax[0]; a.plot(t, rec["tgt"][:, 2], "--", color=SEC, lw=1, label="Target"); a.plot(t, p[:, 2], color=C[0], lw=2, label="Actual")
a.set_title("Climb to 2000 m: altitude (m)"); a.set_xlabel("Time (s)"); a.legend(frameon=False, loc="lower right")
a = ax[1]; a.plot(p[:, 2], pwm, color=C[0], lw=1.5); a.set_ylim(1000, 2000)
a.set_title("Average motor PWM (µs) against altitude: thinner air needs more throttle"); a.set_xlabel("Altitude (m)")
x = np.arange(len(MASSES))
a = ax[2]; a.bar(x, [100 * r["completed"] / r["n"] for r in rows], width=0.6, color=C[0])
for i, r in enumerate(rows): a.text(i, 100 * r["completed"] / r["n"] + 2, f"{100*r['completed']//r['n']}%", ha="center", color=INK)
a.set_xticks(x); a.set_xticklabels([f"{m} kg" for m in MASSES]); a.set_ylim(0, 115); a.grid(axis="x", visible=False)
a.set_title("Mass sweep: share of 200 randomised drones completing the square mission")
a = ax[3]; a.bar(x, [100 * r["crashed"] / r["n"] for r in rows], width=0.6, color=C[1])
for i, r in enumerate(rows): a.text(i, 100 * r["crashed"] / r["n"] + 2, f"{100*r['crashed']//r['n']}%", ha="center", color=INK)
a.set_xticks(x); a.set_xticklabels([f"{m} kg" for m in MASSES]); a.set_ylim(0, 115); a.grid(axis="x", visible=False)
a.set_title("Mass sweep: share that crashed")
fig.tight_layout(h_pad=2); fig.savefig(f"{tag}_results.png", dpi=130)
