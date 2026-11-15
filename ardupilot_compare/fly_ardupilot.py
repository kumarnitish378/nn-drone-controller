"""Fly the comparison mission in ArduCopter SITL (ArduPilot's own simulator and PID controllers)
and keep its DataFlash log (.BIN).

usage: python3 fly_ardupilot.py /path/to/ardupilot  out_dir  [speedup]
needs a built SITL binary:  ./waf configure --board sitl && ./waf copter
"""
import math, os, shutil, subprocess, sys, time, glob
os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav
from mission_def import HOME, MISSION, AP_PARAMS as PARAMS, ll

ap_root = os.path.abspath(sys.argv[1]); out = os.path.abspath(sys.argv[2]); speedup = sys.argv[3] if len(sys.argv) > 3 else "10"
work = os.path.join(out, "sitl_run"); shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
binary = os.path.join(ap_root, "build/sitl/bin/arducopter")
defaults = os.path.join(ap_root, "Tools/autotest/default_params/copter.parm")
sitl = subprocess.Popen([binary, "-w", "--model", "quad", "--speedup", speedup, "--defaults", defaults, "-I0",
                         "--home", f"{HOME[0]},{HOME[1]},{HOME[2]},0"], cwd=work,
                        stdout=open(os.path.join(work, "sitl_stdout.txt"), "w"), stderr=subprocess.STDOUT)
time.sleep(2)
m = mavutil.mavlink_connection("tcp:127.0.0.1:5760", source_system=255, source_component=190, dialect="ardupilotmega")
texts = []

def wait(pred, timeout, types=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=types, blocking=True, timeout=0.5)
        if msg is None:
            continue
        if msg.get_type() == "STATUSTEXT":
            texts.append(msg.text); print("   AP:", msg.text, flush=True)
        if pred(msg):
            return msg
    return None

def cmd(command, *p, timeout=5):
    p = list(p) + [0] * (7 - len(p))
    m.mav.command_long_send(1, 1, command, 0, *p)
    a = wait(lambda x: x.get_type() == "COMMAND_ACK" and x.command == command, timeout)
    return a is not None and a.result == mav.MAV_RESULT_ACCEPTED

def set_param(name, value):
    for _ in range(3):
        m.mav.param_set_send(1, 1, name.encode(), float(value), mav.MAV_PARAM_TYPE_REAL32)
        pv = wait(lambda x: x.get_type() == "PARAM_VALUE" and x.param_id == name, 3)
        if pv is not None:
            return abs(pv.param_value - value) < 1e-3 * max(1, abs(value))
    return False

try:
    hb = wait(lambda x: x.get_type() == "HEARTBEAT" and x.get_srcSystem() == 1, 60)
    assert hb is not None, "no heartbeat from SITL"
    print("connected to ArduCopter SITL", flush=True)
    m.mav.request_data_stream_send(1, 1, mav.MAV_DATA_STREAM_ALL, 4, 1)
    for k, v in PARAMS.items():
        ok = set_param(k, v); print(f"   set {k} = {v}: {'ok' if ok else 'FAILED'}", flush=True)
        assert ok, f"parameter {k} was not accepted"

    # upload the mission (item 0 = home, ArduPilot convention)
    items = [(16, 0, HOME[0], HOME[1], HOME[2], 0, 0)] + [(c, 3, *((ll(n, e)) if pos else (0, 0)), alt, p1, p3) for c, pos, n, e, alt, p1, p3 in MISSION]
    wait(lambda x: x.get_type() == "STATUSTEXT" and "is using GPS" in x.text, 90)      # EKF has a position and home is set
    def serve(x):
        if x.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            c, fr, lat, lon, alt, p1, p3 = items[x.seq]
            m.mav.mission_item_int_send(1, 1, x.seq, fr, c, 0, 1, p1, 0, p3, 0, int(lat * 1e7), int(lon * 1e7), alt, 0)
        return x.get_type() == "MISSION_ACK"
    ack = None
    for attempt in range(4):
        m.mav.mission_count_send(1, 1, len(items), 0)
        ack = wait(serve, 15)
        if ack is not None and ack.type == mav.MAV_MISSION_ACCEPTED:
            break
        print("   mission upload attempt", attempt + 1, "failed:", None if ack is None else ack.type, flush=True); time.sleep(2)
    assert ack is not None and ack.type == mav.MAV_MISSION_ACCEPTED, "mission upload failed"
    print("mission uploaded", flush=True)

    # wait for the EKF and GPS, then arm in AUTO and start
    assert cmd(mav.MAV_CMD_DO_SET_MODE, 1, 3), "could not set AUTO"
    t0 = time.time(); armed = False
    while time.time() - t0 < 240 and not armed:
        if cmd(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1):
            armed = wait(lambda x: x.get_type() == "HEARTBEAT" and x.get_srcSystem() == 1 and x.base_mode & 128, 5) is not None
        else:
            time.sleep(2)
    assert armed, "arming refused"
    print("armed", flush=True)
    cmd(mav.MAV_CMD_MISSION_START)
    off = wait(lambda x: x.get_type() == "HEARTBEAT" and x.get_srcSystem() == 1 and not (x.base_mode & 128), 900)
    assert off is not None, "mission did not finish"
    print("mission finished, vehicle disarmed", flush=True)
    time.sleep(3)
finally:
    sitl.terminate(); time.sleep(1)
logs = sorted(glob.glob(os.path.join(work, "logs", "*.BIN")))
assert logs, "no .BIN log written"
dst = os.path.join(out, "ardupilot_sitl.BIN"); shutil.copy(logs[-1], dst)
print("log:", dst, os.path.getsize(dst) // 1024, "kB")
