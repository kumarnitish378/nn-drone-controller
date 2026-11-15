"""Check that the browser simulator core (web/core.js) flies the same as the Python simulator.
Flies one mission in both (sensor noise off) and compares the tracks. Needs node and pymavlink.
usage (from the repository root): python3 web/test_core.py"""
import json, os, subprocess, sys, tempfile
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); sys.path.insert(0, root)
import mavlink_sim as ms
from pymavlink.dialects.v20 import ardupilotmega as mav

JS = r"""
const fs=require('fs'); const {Vehicle}=require(process.argv[2]); const c=JSON.parse(fs.readFileSync(process.argv[3]));
const v=new Vehicle(c.frame,c.w,{SIM_MASS:c.mass,SIM_NOISE:0,SIM_WIND_SPD:2,SIM_WIND_DIR:45,HOME_ALT:0}); v.idleTime=0;
v.start([{cmd:'takeoff',alt:20},{cmd:'waypoint',n:60,e:0,alt:30},{cmd:'waypoint',n:60,e:60,alt:40,hold:2},{cmd:'circle',n:0.001,e:60,alt:30,radius:15,turns:1},{cmd:'rtl'}]);
let k=0,i=0,maxd=0; while(v.armed&&k<400000){v.step();k++; if((k-1)%50===0&&k>1&&i<c.traj.length){const r=c.traj[i++],p=v.pos;maxd=Math.max(maxd,Math.hypot(p[0]-r[0],p[1]-r[1],p[2]-r[2]));}}
console.log(JSON.stringify({steps:k,maxd:maxd,final:v.pos,crashed:v.crashed}));
"""
ok = True
for frame, npz, mass in (("quad_x", "quad_x_policy_v4.npz", 4.0), ("hex_x", "hex_x_policy_v1.npz", 95.0)):
    v = ms.Vehicle(frame, os.path.join(root, npz), (28.5355, 77.3910, 0.0), mass, log=lambda *a: None)
    for k, val in dict(SIM_NOISE=0.0, SIM_WIND_SPD=2.0, SIM_WIND_DIR=45.0).items(): v.set_param(k, val)
    def it(cmd, n=0, e=0, alt=0, p1=0, p3=0):
        lat, lon = v.to_global([n, -e, 0]) if (n or e) else (0, 0)
        return dict(seq=0, frame=3, command=cmd, current=0, autocontinue=1, p1=p1, p2=0, p3=p3, p4=0, x=lat, y=lon, z=alt)
    v.mission = [it(16), it(mav.MAV_CMD_NAV_TAKEOFF, alt=20), it(16, 60, 0, 30), it(16, 60, 60, 40, p1=2),
                 it(mav.MAV_CMD_NAV_LOITER_TURNS, 0.001, 60, 30, p1=1, p3=15), it(mav.MAV_CMD_NAV_RETURN_TO_LAUNCH)]
    v.mode = ms.AUTO; v.arm(); v.start_mission(); traj = []; k = 0
    while v.armed and k < 400000:
        v.step(); k += 1
        if k % 50 == 0: traj.append([float(x) for x in v.pos])
    d = np.load(os.path.join(root, npz)); W0 = d["pi_W0"]; M = d["pi_W2"].shape[1]; nin = 21 + M + 2 * (14 + M)
    if W0.shape[0] < nin: W0 = np.concatenate([W0, np.zeros((nin - W0.shape[0], W0.shape[1]), np.float32)])
    w = dict(W0=W0.tolist(), b0=d["pi_b0"].tolist(), W1=d["pi_W1"].tolist(), b1=d["pi_b1"].tolist(), W2=d["pi_W2"].tolist(), b2=d["pi_b2"].tolist())
    with tempfile.TemporaryDirectory() as td:
        json.dump(dict(frame=frame, mass=mass, w=w, traj=traj), open(os.path.join(td, "case.json"), "w"))
        open(os.path.join(td, "run.js"), "w").write(JS)
        out = json.loads(subprocess.run(["node", os.path.join(td, "run.js"), os.path.join(here, "core.js"), os.path.join(td, "case.json")],
                                        check=True, capture_output=True, text=True).stdout)
    good = out["steps"] == k and out["maxd"] < 0.5 * mass ** (1 / 3) and not out["crashed"]
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {frame} {mass:g} kg: steps {out['steps']} vs {k}, largest track difference {out['maxd']:.3f} m")
sys.exit(0 if ok else 1)
