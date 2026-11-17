"""Drive ArduCopter SITL over MAVLink: start the binary, wait for messages, send commands and parameters."""
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import ardupilotmega as mavlink  # noqa: E402

SITL_ADDRESS = "tcp:127.0.0.1:5760"
VEHICLE = (1, 1)
STARTUP_PAUSE_S = 2
SHUTDOWN_PAUSE_S = 1
PARAMETER_ATTEMPTS = 3
PARAMETER_REPLY_S = 3
COMMAND_REPLY_S = 5
RECEIVE_TIMEOUT_S = 0.5
HEARTBEAT_WAIT_S = 60
GPS_READY_WAIT_S = 90
GPS_READY_TEXT = "is using GPS"
ARMED_FLAG = mavlink.MAV_MODE_FLAG_SAFETY_ARMED
PARAMETER_RELATIVE_TOLERANCE = 1e-3


class SitlSession:
    def __init__(self, ardupilot_root, work_dir, home, speedup, echo_status=False, log_stdout=True):
        self.ardupilot_root = Path(ardupilot_root)
        self.work_dir = Path(work_dir)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        self.echo_status = echo_status
        self.status_texts = []
        binary = self.ardupilot_root / "build/sitl/bin/arducopter"
        defaults = self.ardupilot_root / "Tools/autotest/default_params/copter.parm"
        output = open(self.work_dir / "sitl_stdout.txt", "w") if log_stdout else subprocess.DEVNULL
        self.process = subprocess.Popen(
            [str(binary), "-w", "--model", "quad", "--speedup", str(speedup), "--defaults", str(defaults), "-I0",
             "--home", f"{home.latitude},{home.longitude},{home.altitude_asl},0"],
            cwd=self.work_dir, stdout=output, stderr=subprocess.STDOUT)
        time.sleep(STARTUP_PAUSE_S)
        self.link = mavutil.mavlink_connection(SITL_ADDRESS, source_system=255, source_component=190, dialect="ardupilotmega")

    def wait(self, predicate, timeout):
        started = time.time()
        while time.time() - started < timeout:
            message = self.link.recv_match(blocking=True, timeout=RECEIVE_TIMEOUT_S)
            if message is None:
                continue
            if message.get_type() == "STATUSTEXT":
                self.status_texts.append(message.text)
                if self.echo_status:
                    print("   AP:", message.text, flush=True)
            if predicate(message):
                return message
        return None

    def command(self, command, *params, timeout=COMMAND_REPLY_S):
        params = list(params) + [0] * (7 - len(params))
        self.link.mav.command_long_send(*VEHICLE, command, 0, *params)
        ack = self.wait(lambda message: message.get_type() == "COMMAND_ACK" and message.command == command, timeout)
        return ack is not None and ack.result == mavlink.MAV_RESULT_ACCEPTED

    def set_parameter(self, name, value):
        for _ in range(PARAMETER_ATTEMPTS):
            self.link.mav.param_set_send(*VEHICLE, name.encode(), float(value), mavlink.MAV_PARAM_TYPE_REAL32)
            reply = self.wait(lambda message: message.get_type() == "PARAM_VALUE" and message.param_id == name, PARAMETER_REPLY_S)
            if reply is not None:
                return abs(reply.param_value - value) < PARAMETER_RELATIVE_TOLERANCE * max(1, abs(value))
        return False

    def wait_for_heartbeat(self):
        heartbeat = self.wait(lambda message: message.get_type() == "HEARTBEAT" and message.get_srcSystem() == 1,
                              HEARTBEAT_WAIT_S)
        if heartbeat is None:
            raise RuntimeError("no heartbeat from SITL")

    def request_all_streams(self, rate_hz):
        self.link.mav.request_data_stream_send(*VEHICLE, mavlink.MAV_DATA_STREAM_ALL, rate_hz, 1)

    def wait_for_gps(self):
        self.wait(lambda message: message.get_type() == "STATUSTEXT" and GPS_READY_TEXT in message.text, GPS_READY_WAIT_S)

    def arm_when_ready(self, timeout, retry_pause_s=2):
        started = time.time()
        while time.time() - started < timeout:
            if self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1):
                if self.wait(lambda message: self._is_own_heartbeat(message) and message.base_mode & ARMED_FLAG,
                             COMMAND_REPLY_S) is not None:
                    return True
            else:
                time.sleep(retry_pause_s)
        return False

    def wait_until_disarmed(self, timeout):
        return self.wait(lambda message: self._is_own_heartbeat(message) and not (message.base_mode & ARMED_FLAG), timeout)

    @staticmethod
    def _is_own_heartbeat(message):
        return message.get_type() == "HEARTBEAT" and message.get_srcSystem() == 1

    def close(self):
        self.process.terminate()
        time.sleep(SHUTDOWN_PAUSE_S)

    def latest_log(self):
        logs = sorted(glob.glob(str(self.work_dir / "logs" / "*.BIN")))
        if not logs:
            raise RuntimeError("SITL wrote no .BIN log")
        return Path(logs[-1])
