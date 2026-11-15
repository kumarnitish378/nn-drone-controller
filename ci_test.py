"""CI gate: run the test cases on one trained network and fail (exit 1) if any threshold is missed.
usage: python3 ci_test.py <weights.npz> <quad_x|hex_x>      -> writes ci_report_<frame>.json"""
import os, sys, json, subprocess, time
weights, frame = sys.argv[1], sys.argv[2]
os.environ["NN_FRAME"] = frame
sys.argv = [sys.argv[0], weights, frame, "2.5"]
import numpy as np
import evaluate as ev

t0 = time.time(); checks = []; report = dict(weights=weights, frame=frame)

def check(name, value, ok, limit):
    checks.append(dict(test=name, value=value, limit=limit, passed=bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value}  (limit {limit})", flush=True)

def run(cmd, env=None):
    e = dict(os.environ, **(env or {}))
    subprocess.run([sys.executable] + cmd, check=True, env=e, stdout=subprocess.DEVNULL)

# 1. square waypoint mission, 200 randomised 1 kg drones
r = ev.fly(200, True, True, 2)
check("square: completed of 200", int(r["ok"].sum()), r["ok"].sum() >= 198, ">= 198")
check("square: crashed", int(r["crashed"].sum()), r["crashed"].sum() == 0, "== 0")
check("square: hover error median (m)", round(float(np.nanmedian(r["hold"])), 3), np.nanmedian(r["hold"]) < 0.10, "< 0.10")

# 2. same mission started at 2000 m (thin air)
r = ev.fly(200, True, True, 2, alt0=2000.0)
check("2000 m: completed of 200", int(r["ok"].sum()), r["ok"].sum() >= 180, ">= 180")
check("2000 m: crashed of 200", int(r["crashed"].sum()), r["crashed"].sum() <= 10, "<= 10")

# 3. heavy drones
for m in (10.0, 100.0):
    r = ev.fly(100, True, True, 3, max_t=90.0, mass=m)
    check(f"{m:g} kg: completed of 100", int(r["ok"].sum()), r["ok"].sum() >= 85, ">= 85")
    check(f"{m:g} kg: crashed", int(r["crashed"].sum()), r["crashed"].sum() == 0, "== 0")

# 4. circle, 5 m radius
run(["circle.py", weights, "2.5", f"ci_circle_{frame}"])
c = json.load(open(f"ci_circle_{frame}_results.json"))
check("circle: mean distance from line (m)", round(c["nominal_radial_error_mean_m"], 3), c["nominal_radial_error_mean_m"] < 0.25, "< 0.25")
check("circle: crashed", c["robust_crashed"], c["robust_crashed"] == 0, "== 0")

# 5. six-waypoint mission from disarmed to disarmed, 4 kg (standard drone + 10 randomised)
run(["mission_full.py", weights, f"ci_mission_{frame}", "4", "3.5", "3", "1.8"], env={"MISSION_N": "10"})
f = json.load(open(f"ci_mission_{frame}_results.json")); s, q = f["standard"], f["randomised_100"]
check("mission: standard drone crashed", s["crashed"], not s["crashed"], "False")
check("mission: touchdown speed (m/s)", round(s["touchdown_speed_mps"], 2), s["touchdown_speed_mps"] < 1.0, "< 1.0")
check("mission: touchdown offset (m)", round(s["touchdown_offset_m"], 2), s["touchdown_offset_m"] < 0.5, "< 0.5")
check("mission: ends disarmed at 1000 us", round(f["pwm_sequence_standard_drone"]["13 disarmed"]), f["pwm_sequence_standard_drone"]["13 disarmed"] == 1000.0, "== 1000")
check("mission: randomised crashed of 10", q["crashed"], q["crashed"] == 0, "== 0")
check("mission: randomised landed of 10", q["landed"], q["landed"] == 10, "== 10")

report.update(checks=checks, passed=all(c["passed"] for c in checks), seconds=round(time.time() - t0))
json.dump(report, open(f"ci_report_{frame}.json", "w"), indent=1)
print(f"\n{'ALL TESTS PASSED' if report['passed'] else 'TESTS FAILED'} in {report['seconds']} s")
sys.exit(0 if report["passed"] else 1)
