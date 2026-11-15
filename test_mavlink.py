"""Scripted ground station: checks that mavlink_sim.py speaks enough MAVLink to configure the drone,
upload a mission and fly it. Starts the simulator itself (accelerated) and exits non-zero on failure.
usage: python3 test_mavlink.py [quad_x|hex_x] [mass_kg]"""
import os, sys, time, subprocess, math
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav

frame = sys.argv[1] if len(sys.argv) > 1 else "quad_x"
mass = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
PORT = 14560 if frame == "quad_x" else 14561
HOME = (28.5355, 77.3910, 200.0)
here = os.path.dirname(os.path.abspath(__file__))
gcs = mavutil.mavlink_connection(f"udpin:127.0.0.1:{PORT}", source_system=255, source_component=190, dialect="ardupilotmega")
sim = subprocess.Popen([sys.executable, os.path.join(here, "mavlink_sim.py"), "--frame", frame, "--mass", "1",
                        "--gcs", f"127.0.0.1:{PORT}", "--speedup", "30", "--home", ",".join(map(str, HOME)), "--duration", "280"],
                       stdout=subprocess.DEVNULL)
results = []; state = dict(texts=[], reached=[], pwm_max=0, alt_max=0.0, last={})

def check(name, ok, detail=""):
    results.append(ok); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

def pump(msg):
    t = msg.get_type(); state["last"][t] = msg
    if t == "STATUSTEXT": state["texts"].append(msg.text)
    elif t == "MISSION_ITEM_REACHED": state["reached"].append(msg.seq)
    elif t == "SERVO_OUTPUT_RAW": state["pwm_max"] = max(state["pwm_max"], msg.servo1_raw)
    elif t == "GLOBAL_POSITION_INT": state["alt_max"] = max(state["alt_max"], msg.relative_alt / 1000)

def wait(pred, timeout, types=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = gcs.recv_match(type=types, blocking=True, timeout=0.5)
        if m is None: continue
        pump(m)
        r = pred(m)
        if r: return m
    return None

def cmd(command, *p, expect=mav.MAV_RESULT_ACCEPTED):
    p = list(p) + [0] * (7 - len(p))
    gcs.mav.command_long_send(1, 1, command, 0, *p)
    a = wait(lambda m: m.get_type() == "COMMAND_ACK" and m.command == command, 5)
    return a is not None and a.result == expect

try:
    hb = wait(lambda m: m.get_type() == "HEARTBEAT", 30)
    check("heartbeat received", hb is not None)
    check("reports ArduPilot-style copter", hb.autopilot == mav.MAV_AUTOPILOT_ARDUPILOTMEGA and
          hb.type == (mav.MAV_TYPE_QUADROTOR if frame == "quad_x" else mav.MAV_TYPE_HEXAROTOR))

    # ---- parameters
    gcs.mav.param_request_list_send(1, 1); params = {}
    def got(m):
        if m.get_type() == "PARAM_VALUE":
            params[m.param_id] = m.param_value
            return len(params) >= m.param_count
    wait(got, 10)
    check("parameter list downloaded", len(params) >= 20 and "SIM_MASS" in params, f"{len(params)} parameters")
    gcs.mav.param_set_send(1, 1, b"SIM_MASS", mass, mav.MAV_PARAM_TYPE_REAL32)
    pv = wait(lambda m: m.get_type() == "PARAM_VALUE" and m.param_id == "SIM_MASS", 5)
    check("set SIM_MASS", pv is not None and abs(pv.param_value - mass) < 1e-3, f"{mass:g} kg")
    spd = 300.0 * mass ** (1 / 6)
    for n, val in (("WPNAV_SPEED", spd), ("WPNAV_SPEED_UP", spd), ("WPNAV_SPEED_DN", spd * 0.6), ("SIM_WIND_SPD", 2.0), ("SIM_WIND_DIR", 45.0)):
        gcs.mav.param_set_send(1, 1, n.encode(), val, mav.MAV_PARAM_TYPE_REAL32)
        wait(lambda m: m.get_type() == "PARAM_VALUE" and m.param_id == n, 5)

    cmd(mav.MAV_CMD_REQUEST_MESSAGE, mav.MAVLINK_MSG_ID_AUTOPILOT_VERSION)
    av = wait(lambda m: m.get_type() == "AUTOPILOT_VERSION", 5)
    check("autopilot version answered", av is not None)

    # ---- mission upload: home, take-off 20 m, two waypoints, a 15 m circle, return to launch
    def ll(north, east):
        return HOME[0] + north / 111319.5, HOME[1] + east / (111319.5 * math.cos(math.radians(HOME[0])))
    items = [(16, 0, *HOME[:2], 0, 0, 0)]                                             # 0: home
    items.append((mav.MAV_CMD_NAV_TAKEOFF, 3, 0, 0, 20, 0, 0))
    items.append((mav.MAV_CMD_NAV_WAYPOINT, 3, *ll(60, 0), 30, 0, 0))
    items.append((mav.MAV_CMD_NAV_WAYPOINT, 3, *ll(60, 60), 40, 2, 0))               # hold 2 s
    items.append((mav.MAV_CMD_NAV_LOITER_TURNS, 3, *ll(0, 60), 30, 1, 15))           # 1 turn, 15 m radius
    items.append((mav.MAV_CMD_NAV_RETURN_TO_LAUNCH, 3, 0, 0, 0, 0, 0))
    gcs.mav.mission_count_send(1, 1, len(items), 0)
    def serve(m):
        if m.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            c, fr, lat, lon, alt, p1, p3 = items[m.seq]
            gcs.mav.mission_item_int_send(1, 1, m.seq, fr, c, 0, 1, p1, 0, p3, 0, int(lat * 1e7), int(lon * 1e7), alt, 0)
        return m.get_type() == "MISSION_ACK"
    ackm = wait(serve, 15)
    check("mission upload accepted", ackm is not None and ackm.type == mav.MAV_MISSION_ACCEPTED, f"{len(items)} items")
    gcs.mav.mission_request_list_send(1, 1, 0)
    cnt = wait(lambda m: m.get_type() == "MISSION_COUNT", 5)
    back = []
    for i in range(cnt.count if cnt else 0):
        gcs.mav.mission_request_int_send(1, 1, i, 0)
        it = wait(lambda m: m.get_type() == "MISSION_ITEM_INT" and m.seq == i, 5)
        back.append(it)
    gcs.mav.mission_ack_send(1, 1, mav.MAV_MISSION_ACCEPTED, 0)
    check("mission download matches", cnt is not None and cnt.count == len(items) and all(b is not None for b in back) and
          back[4].command == mav.MAV_CMD_NAV_LOITER_TURNS and abs(back[4].param3 - 15) < 1e-3)

    # ---- fly it
    check("unsupported mode rejected", cmd(mav.MAV_CMD_DO_SET_MODE, 1, 0, expect=mav.MAV_RESULT_FAILED))
    check("mode AUTO", cmd(mav.MAV_CMD_DO_SET_MODE, 1, 3))
    check("arm", cmd(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1))
    hb = wait(lambda m: m.get_type() == "HEARTBEAT" and m.base_mode & 128, 5)
    check("heartbeat shows armed in AUTO", hb is not None and hb.custom_mode == 3)
    check("mission start", cmd(mav.MAV_CMD_MISSION_START))
    done = wait(lambda m: m.get_type() == "HEARTBEAT" and not (m.base_mode & 128), 200)
    check("mission flown and vehicle disarmed", done is not None)
    check("waypoints reported reached", all(s in state["reached"] for s in (1, 2, 3, 4)), str(state["reached"]))
    check("climbed to mission altitude", 38 < state["alt_max"] < 45, f"max {state['alt_max']:.1f} m")
    check("motor PWM reported", 1500 < state["pwm_max"] <= 2000, f"max {state['pwm_max']} us")
    gp = wait(lambda m: m.get_type() == "GLOBAL_POSITION_INT", 5)
    north = (gp.lat / 1e7 - HOME[0]) * 111319.5; east = (gp.lon / 1e7 - HOME[1]) * 111319.5 * math.cos(math.radians(HOME[0]))
    off = math.hypot(north, east)
    check("landed at home", gp.relative_alt < 300 and off < 2.0 * mass ** (1 / 3), f"{off:.2f} m from home, height {gp.relative_alt/1000:.2f} m")
    check("no crash reported", not any("CRASH" in t for t in state["texts"]))

    # ---- guided flight: take-off, go to a point, land
    check("mode GUIDED", cmd(mav.MAV_CMD_DO_SET_MODE, 1, 4))
    check("arm again", cmd(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1))
    check("guided take-off", cmd(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 10))
    up = wait(lambda m: m.get_type() == "GLOBAL_POSITION_INT" and m.relative_alt > 9000, 60)
    check("reached 10 m", up is not None)
    lat, lon = ll(-25, 10)
    gcs.mav.command_int_send(1, 1, 6, mav.MAV_CMD_DO_REPOSITION, 0, 0, -1, 0, 0, 0, int(lat * 1e7), int(lon * 1e7), 12)
    there = wait(lambda m: m.get_type() == "GLOBAL_POSITION_INT" and abs((m.lat / 1e7 - lat) * 111319.5) < 1.5 * mass ** (1 / 3)
                 and abs(m.relative_alt - 12000) < 1500, 90)
    check("guided go-to reached", there is not None)
    check("land command", cmd(mav.MAV_CMD_NAV_LAND))
    done = wait(lambda m: m.get_type() == "HEARTBEAT" and not (m.base_mode & 128), 120)
    check("landed and disarmed", done is not None)
    check("no crash reported (guided)", not any("CRASH" in t for t in state["texts"]))
finally:
    sim.terminate()
print("\nstatus texts:", state["texts"])
print("ALL PASSED" if all(results) else "FAILED")
sys.exit(0 if all(results) else 1)
