"""Fly the comparison mission with the neural network in this repository's simulator and save a log.
usage: python3 fly_network.py out_dir [quad_x|hex_x]"""
import os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); sys.path.insert(0, root)
import mavlink_sim as ms
from sim import euler_from_quat
from mission_def import HOME, MISSION, PARAMS, MASS, ll

out = sys.argv[1]; frame = sys.argv[2] if len(sys.argv) > 2 else "quad_x"
nonoise = len(sys.argv) > 3 and sys.argv[3] == "nonoise"
v = ms.Vehicle(frame, os.path.join(root, ms.DEFAULT_WEIGHTS[frame]), HOME, MASS, log=lambda *a: None)
if nonoise: v.set_param("SIM_NOISE", 0.0)
for k, val in PARAMS.items():
    if k in v.P: v.set_param(k, val)
v.mission = [dict(seq=0, frame=0, command=16, current=0, autocontinue=1, p1=0, p2=0, p3=0, p4=0, x=HOME[0], y=HOME[1], z=HOME[2])]
for c, pos, n, e, alt, p1, p3 in MISSION:
    lat, lon = ll(n, e) if pos else (0, 0)
    v.mission.append(dict(seq=len(v.mission), frame=3, command=c, current=0, autocontinue=1, p1=p1, p2=0, p3=p3, p4=0, x=lat, y=lon, z=alt))
v.mode = ms.AUTO; assert v.arm() and v.start_mission()
rows = []
while v.armed and v.t < 900:
    v.step(); e = v.env
    r, p, y = euler_from_quat(e.q)
    rows.append([v.t, e.p[0, 0], -e.p[0, 1], e.p[0, 2], np.degrees(r[0]), np.degrees(-p[0]), np.degrees(-y[0]) % 360, e.volt[0], *v.pwm])
rows = np.array(rows)
np.savez_compressed(os.path.join(out, f"network_{frame}{'_nonoise' if nonoise else ''}.npz"), data=rows.astype(np.float32),
                    columns=np.array(["t", "north", "east", "alt", "roll_deg", "pitch_deg", "heading_deg", "volt"] + [f"pwm{i+1}" for i in range(e.M)]))
print(f"network {frame}: {len(rows)} steps, {v.t:.1f} s, ended at N {e.p[0,0]:.2f} E {-e.p[0,1]:.2f}, crashed {v.crashed}, touchdown {e.impact[0]:.2f} m/s")
