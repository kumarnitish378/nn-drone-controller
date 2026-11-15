"""Stability checks, ArduCopter SITL against the neural networks.
 A. steadiness of attitude in the hover and after take-off (from the mission logs made by fly_*.py)
 B. gust test: hover at 20 m, wind steps from 0 to 8 m/s for 15 s, then back to 0
 C. extra control delay the network tolerates (network only)
usage: python3 stability.py /path/to/ardupilot out_dir"""
import json, math, os, shutil, subprocess, sys, time, glob
import numpy as np
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here); sys.path.insert(0, root)
import mavlink_sim as ms
from sim import euler_from_quat
from mission_def import HOME, MASS

ap_root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
GUST, T_SETTLE, T_GUST, T_AFTER, ALT = 8.0, 12.0, 15.0, 15.0, 20.0
res = {}

def unwrap_deg(a): return np.degrees(np.unwrap(np.radians(a)))

# ---------------------------------------------------------------- A. attitude steadiness from the mission logs
def att_ardupilot(path):
    m = mavutil.mavlink_connection(path); a = []
    while True:
        x = m.recv_match(type=["ATT"])
        if x is None: break
        a.append([x.TimeUS * 1e-6, x.Roll, x.Pitch, x.Yaw])
    a = np.array(a); a[:, 0] -= a[0, 0]; return a
def att_network(path):
    D = np.load(path)["data"].astype(float); tt = np.arange(0, D[-1, 0], 0.1)
    return np.stack([tt, np.interp(tt, D[:, 0], D[:, 4]), np.interp(tt, D[:, 0], D[:, 5]), np.interp(tt, D[:, 0], unwrap_deg(D[:, 6]))], 1)
def steadiness(a, hover, climb):
    h = a[(a[:, 0] >= hover[0]) & (a[:, 0] <= hover[1])]; c = a[(a[:, 0] >= climb[0]) & (a[:, 0] <= climb[1])]
    yaw_h = unwrap_deg(h[:, 3]); yaw_c = unwrap_deg(c[:, 3])
    return {"hover roll wobble, std (deg)": float(h[:, 1].std()), "hover pitch wobble, std (deg)": float(h[:, 2].std()),
            "hover heading wobble, std (deg)": float(yaw_h.std()),
            "climb-out heading swing, peak to peak (deg)": float(yaw_c.max() - yaw_c.min()),
            "climb-out roll and pitch wobble, std (deg)": float(np.hypot(c[:, 1].std(), c[:, 2].std()))}
logs = {"ArduCopter 4.7.2 (PID)": att_ardupilot(os.path.join(out, "ardupilot_sitl.BIN")),
        "Neural network, quad": att_network(os.path.join(out, "network_quad_x.npz")),
        "Neural network, hexa": att_network(os.path.join(out, "network_hex_x.npz"))}
res["A_attitude_steadiness"] = {}

# hover windows: reuse the ones compare.py found, by recomputing from position logs
def hover_window_net(path):
    D = np.load(path)["data"].astype(float); d = np.sqrt((D[:, 1] - 60) ** 2 + (D[:, 2] - 60) ** 2 + (D[:, 3] - 40) ** 2); i = np.nonzero(d < 2.5)[0]
    return D[i[0], 0] + 6, D[i[0], 0] + 26
def hover_window_ap(path):
    m = mavutil.mavlink_connection(path); p = []
    while True:
        x = m.recv_match(type=["SIM2"])
        if x is None: break
        p.append([x.TimeUS * 1e-6, x.PN, x.PE, -x.PD])
    p = np.array(p[::8]); p[:, 0] -= p[0, 0]; p[:, 1:] -= p[0, 1:]
    d = np.sqrt((p[:, 1] - 60) ** 2 + (p[:, 2] - 60) ** 2 + (p[:, 3] - 40) ** 2); i = np.nonzero(d < 2.5)[0]
    return p[i[0], 0] + 6, p[i[0], 0] + 26
HW = {"ArduCopter 4.7.2 (PID)": hover_window_ap(os.path.join(out, "ardupilot_sitl.BIN")),
      "Neural network, quad": hover_window_net(os.path.join(out, "network_quad_x.npz")),
      "Neural network, hexa": hover_window_net(os.path.join(out, "network_hex_x.npz"))}
for name, a in logs.items():
    res["A_attitude_steadiness"][name] = steadiness(a, HW[name], (3, 9))

# ---------------------------------------------------------------- B. gust test
def gust_metrics(t, n, e, alt, roll, pitch, t_on):
    """t_on = gust start. Gust blows from the east, so the drone is pushed west (east < 0)."""
    t_off = t_on + T_GUST
    pre = (t > t_on - 5) & (t < t_on); n0, e0, a0 = n[pre].mean(), e[pre].mean(), alt[pre].mean()
    d = np.hypot(n - n0, e - e0); tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)) * np.cos(np.radians(pitch)), -1, 1)))
    g = (t >= t_on) & (t <= t_off); aft = (t > t_off) & (t <= t_off + T_AFTER)
    def settle(mask, t_start, tol=0.3):
        tt, dd = t[mask], d[mask]; bad = np.nonzero(dd > tol)[0]
        return 0.0 if len(bad) == 0 else float(tt[min(bad[-1] + 1, len(tt) - 1)] - t_start)
    late = (t > t_off - 4) & (t <= t_off)
    return {"pushed off position by, peak (m)": float(d[g].max()),
            "time to get back within 0.3 m while the gust blows (s)": settle(g, t_on),
            "offset still held at the end of the gust (m)": float(d[late].mean()),
            "tilt needed to hold against the gust (deg)": float(tilt[late].mean()),
            "steepest tilt during the gust (deg)": float(tilt[g].max()),
            "height change, peak (m)": float(np.abs(alt[g | aft] - a0).max()),
            "swing when the gust stops, peak (m)": float(d[aft].max()),
            "time to settle within 0.3 m after the gust stops (s)": settle(aft, t_off),
            "_trace": (t - t_on, d, tilt)}

def gust_network(frame):
    v = ms.Vehicle(frame, os.path.join(root, ms.DEFAULT_WEIGHTS[frame]), HOME, MASS, log=lambda *a: None)
    v.set_param("SIM_WIND_SPD", 0.0); v.set_param("SIM_WIND_DIR", 90.0)
    v.mode = ms.GUIDED; v.arm(); v.takeoff(ALT); rows = []; t_on = None; state = 0
    while v.t < 200:
        v.step(); e = v.env; r, p, y = euler_from_quat(e.q)
        rows.append([v.t, e.p[0, 0], -e.p[0, 1], e.p[0, 2], np.degrees(r[0]), np.degrees(-p[0])])
        if state == 0 and e.p[0, 2] > ALT - 0.3: state = 1; t_reach = v.t
        elif state == 1 and v.t > t_reach + T_SETTLE: state = 2; t_on = v.t; v.set_param("SIM_WIND_SPD", GUST)
        elif state == 2 and v.t > t_on + T_GUST: state = 3; v.set_param("SIM_WIND_SPD", 0.0)
        elif state == 3 and v.t > t_on + T_GUST + T_AFTER + 1: break
    D = np.array(rows); assert not v.crashed
    return gust_metrics(D[:, 0], D[:, 1], D[:, 2], D[:, 3], D[:, 4], D[:, 5], t_on)

def gust_ardupilot():
    work = os.path.join(out, "sitl_gust"); shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
    sitl = subprocess.Popen([os.path.join(ap_root, "build/sitl/bin/arducopter"), "-w", "--model", "quad", "--speedup", "5",
                             "--defaults", os.path.join(ap_root, "Tools/autotest/default_params/copter.parm"), "-I0",
                             "--home", f"{HOME[0]},{HOME[1]},{HOME[2]},0"], cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2); m = mavutil.mavlink_connection("tcp:127.0.0.1:5760", source_system=255, source_component=190, dialect="ardupilotmega")
    def wait(pred, timeout, types=None):
        t0 = time.time()
        while time.time() - t0 < timeout:
            x = m.recv_match(type=types, blocking=True, timeout=0.5)
            if x is not None and pred(x): return x
        return None
    def cmd(c, *p):
        p = list(p) + [0] * (7 - len(p)); m.mav.command_long_send(1, 1, c, 0, *p)
        a = wait(lambda x: x.get_type() == "COMMAND_ACK" and x.command == c, 5); return a is not None and a.result == 0
    def setp(n, val):
        for _ in range(3):
            m.mav.param_set_send(1, 1, n.encode(), float(val), mav.MAV_PARAM_TYPE_REAL32)
            if wait(lambda x: x.get_type() == "PARAM_VALUE" and x.param_id == n, 3) is not None: return True
        return False
    try:
        assert wait(lambda x: x.get_type() == "HEARTBEAT" and x.get_srcSystem() == 1, 60) is not None
        m.mav.request_data_stream_send(1, 1, mav.MAV_DATA_STREAM_ALL, 10, 1)
        assert setp("SIM_WIND_SPD", 0) and setp("SIM_WIND_DIR", 90) and setp("WP_SPD_UP", 3.0)
        wait(lambda x: x.get_type() == "STATUSTEXT" and "is using GPS" in x.text, 90)
        assert cmd(mav.MAV_CMD_DO_SET_MODE, 1, 4)
        t0 = time.time(); ok = False
        while time.time() - t0 < 200 and not ok:
            ok = cmd(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1) or (time.sleep(2) or False)
        assert ok, "arming refused"
        assert cmd(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, ALT)
        boot = lambda x: x.time_boot_ms / 1000.0
        up = wait(lambda x: x.get_type() == "GLOBAL_POSITION_INT" and x.relative_alt > (ALT - 0.3) * 1000, 120); assert up is not None
        wait(lambda x: x.get_type() == "GLOBAL_POSITION_INT" and boot(x) > boot(up) + T_SETTLE, 120)
        assert setp("SIM_WIND_SPD", GUST)
        on = wait(lambda x: x.get_type() == "GLOBAL_POSITION_INT", 5); t_on_boot = boot(on)
        wait(lambda x: x.get_type() == "GLOBAL_POSITION_INT" and boot(x) > t_on_boot + T_GUST, 120)
        assert setp("SIM_WIND_SPD", 0)
        wait(lambda x: x.get_type() == "GLOBAL_POSITION_INT" and boot(x) > t_on_boot + T_GUST + T_AFTER + 2, 120)
        cmd(mav.MAV_CMD_NAV_LAND)
        wait(lambda x: x.get_type() == "HEARTBEAT" and x.get_srcSystem() == 1 and not (x.base_mode & 128), 200)
        time.sleep(2)
    finally:
        sitl.terminate(); time.sleep(1)
    log = sorted(glob.glob(os.path.join(work, "logs", "*.BIN")))[-1]; shutil.copy(log, os.path.join(out, "ardupilot_gust.BIN"))
    d = mavutil.mavlink_connection(log); P, A, W = [], [], []
    while True:
        x = d.recv_match(type=["SIM2", "ATT", "PARM"])
        if x is None: break
        k = x.get_type()
        if k == "SIM2": P.append([x.TimeUS * 1e-6, x.PN, x.PE, -x.PD])
        elif k == "ATT": A.append([x.TimeUS * 1e-6, x.Roll, x.Pitch])
        elif k == "PARM" and x.Name == "SIM_WIND_SPD": W.append([x.TimeUS * 1e-6, x.Value])
    P = np.array(P[::8]); A = np.array(A); W = np.array(W)
    t_on = W[W[:, 1] > 1][0, 0]                                   # the moment the wind parameter changed, from the log itself
    return gust_metrics(P[:, 0], P[:, 1], P[:, 2], P[:, 3], np.interp(P[:, 0], A[:, 0], A[:, 1]), np.interp(P[:, 0], A[:, 0], A[:, 2]), t_on)

G = {"ArduCopter 4.7.2 (PID)": gust_ardupilot(), "Neural network, quad": gust_network("quad_x"), "Neural network, hexa": gust_network("hex_x")}
res["B_gust_0_to_8_mps"] = {k: {a: round(b, 3) for a, b in v.items() if not a.startswith("_")} for k, v in G.items()}

# ---------------------------------------------------------------- C. control delay the network tolerates
def delay_test(frame, k):
    """Hold motor commands back by k control steps; hover, then jump the target 5 m sideways. Returns worst tilt or None if it crashed."""
    v = ms.Vehicle(frame, os.path.join(root, ms.DEFAULT_WEIGHTS[frame]), HOME, MASS, log=lambda *a: None)
    v.set_param("SIM_WIND_SPD", 2.0); env = v.env; q = []; real = env.step
    def delayed(a, auto_target=True):
        q.append(np.array(a, float).copy()); return real(q.pop(0) if len(q) > k else q[0], auto_target=auto_target)
    env.step = delayed
    v.mode = ms.GUIDED; v.arm(); v.takeoff(10.0); worst = 0.0; moved = False; t_move = None
    while v.t < 60:
        v.step()
        if v.crashed: return None
        if v.phase == "fly":
            up = 1 - 2 * (env.q[0, 1] ** 2 + env.q[0, 2] ** 2); worst = max(worst, math.degrees(math.acos(max(-1, min(1, up)))))
            if not moved and v.t > 20: moved = True; t_move = v.t; v.goals = [ms.Goal("goto", [0.0, -5.0, 10.0]), ms.Goal("hold", None)]
    err = float(np.hypot(env.p[0, 0], env.p[0, 1] + 5.0))
    return dict(worst_tilt_deg=round(worst, 1), final_error_m=round(err, 2))
res["C_network_delay_tolerance"] = {}
for frame in ("quad_x", "hex_x"):
    dt_ms = 10.0 * MASS ** (1 / 6); rows = {}
    for k in (0, 1, 2, 3, 4, 5, 6, 8):
        r = delay_test(frame, k); rows[f"{k * dt_ms:.0f} ms"] = r if r else "crashed"
    res["C_network_delay_tolerance"][frame] = rows

json.dump(res, open(os.path.join(out, "stability.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = ["#eb6834", "#2a78d6", "#1baf7a"]; INK = "#0b0b0b"; SEC = "#52514e"; BG = "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK})
fig, ax = plt.subplots(3, 1, figsize=(7, 11))
for (name, g), c in zip(G.items(), COL):
    t, d, tilt = g["_trace"]; s = (t > -5) & (t < T_GUST + T_AFTER)
    ax[0].plot(t[s], d[s], color=c, lw=1.6, label=name); ax[1].plot(t[s], tilt[s], color=c, lw=1.3, label=name)
for a in ax[:2]: a.axvspan(0, T_GUST, color="#e8e7e2", alpha=0.6, lw=0)
ax[0].set_title("Gust test: distance pushed off the hover point (m), shaded = 8 m/s wind"); ax[0].legend(frameon=False, loc="upper right")
ax[1].set_title("Gust test: tilt (degrees)"); ax[1].set_xlabel("Time from gust start (s)"); ax[1].legend(frameon=False, loc="upper right")
a = ax[2]
for (name, arr), c in zip(logs.items(), COL):
    s = (arr[:, 0] > 2) & (arr[:, 0] < 14); y = unwrap_deg(arr[s, 3]); a.plot(arr[s, 0], y - y[0], color=c, lw=1.5, label=name)
a.set_title("Mission log: heading change during climb-out (degrees)"); a.set_xlabel("Time from arming (s)"); a.legend(frameon=False, loc="upper right")
fig.tight_layout(h_pad=1.6); fig.savefig(os.path.join(out, "stability.png"), dpi=130)
