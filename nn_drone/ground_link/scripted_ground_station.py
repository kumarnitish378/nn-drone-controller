"""Scripted ground station: checks the link speaks enough MAVLink to configure the drone, upload a mission and
fly it. It starts the simulator itself (accelerated) in a separate process."""
import math
import os
import subprocess
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import ardupilotmega as mavlink  # noqa: E402

from nn_drone.ground_link.geodesy import METRES_PER_DEGREE_LATITUDE, HomePoint  # noqa: E402
from nn_drone.ground_link.mavlink_server import DEGREES_E7  # noqa: E402
from nn_drone.ground_link.vehicle import CopterMode  # noqa: E402
from nn_drone.paths import REPOSITORY_ROOT  # noqa: E402
from nn_drone.simulation.scaling import length_scale, speed_scale  # noqa: E402

HOME = HomePoint(28.5355, 77.3910, 200.0)
PORTS = {"quad_x": 14560, "hex_x": 14561}
SPEEDUP = 30
SIMULATOR_LIFETIME_S = 280
VEHICLE = (1, 1)
ARMED_FLAG = mavlink.MAV_MODE_FLAG_SAFETY_ARMED
UNSUPPORTED_MODE = 1
REFERENCE_SPEED_CMS = 300.0
DESCENT_SHARE = 0.6
TEST_WIND_SPEED_MPS, TEST_WIND_DIRECTION_DEG = 2.0, 45.0
MILLIMETRES = 1000
MIN_PARAMETERS = 20
HEARTBEAT_WAIT_S = 30
REPLY_WAIT_S = 5
MISSION_FLIGHT_S = 200
CLIMB_WAIT_S = 60
GOTO_WAIT_S = 90
LANDING_WAIT_S = 120
MISSION_TOP_ALTITUDE_RANGE_M = (38, 45)
PWM_REPORTED_RANGE_US = (1500, 2000)
HOME_RADIUS_LENGTHS = 2.0
GOTO_RADIUS_LENGTHS = 1.5
GUIDED_TAKEOFF_M = 10
GUIDED_GOTO = (-25, 10, 12)
GOTO_HEIGHT_TOLERANCE_M = 1.5
LANDED_HEIGHT_MM = 300
CIRCLE_RADIUS_M = 15
CIRCLE_ITEM_SEQ = 4
WAYPOINT_ITEM_SEQS = (1, 2, 3, 4)


class ScriptedGroundStation:
    def __init__(self, frame, mass_kg):
        self.frame = frame
        self.mass_kg = mass_kg
        self.results = []
        self.texts, self.reached, self.max_pwm, self.max_height = [], [], 0, 0.0
        port = PORTS[frame]
        self.link = mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}", source_system=255, source_component=190,
                                               dialect="ardupilotmega")
        self.simulator = subprocess.Popen(
            [sys.executable, "-m", "nn_drone", "ground-link", "--frame", frame, "--mass", "1", "--gcs", f"127.0.0.1:{port}",
             "--speedup", str(SPEEDUP), "--home", f"{HOME.latitude},{HOME.longitude},{HOME.altitude_asl}",
             "--duration", str(SIMULATOR_LIFETIME_S)], cwd=REPOSITORY_ROOT, stdout=subprocess.DEVNULL)

    def check(self, name, passed, detail=""):
        self.results.append(bool(passed))
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)

    def _track(self, message):
        kind = message.get_type()
        if kind == "STATUSTEXT":
            self.texts.append(message.text)
        elif kind == "MISSION_ITEM_REACHED":
            self.reached.append(message.seq)
        elif kind == "SERVO_OUTPUT_RAW":
            self.max_pwm = max(self.max_pwm, message.servo1_raw)
        elif kind == "GLOBAL_POSITION_INT":
            self.max_height = max(self.max_height, message.relative_alt / MILLIMETRES)

    def wait(self, predicate, timeout):
        started = time.time()
        while time.time() - started < timeout:
            message = self.link.recv_match(blocking=True, timeout=0.5)
            if message is None:
                continue
            self._track(message)
            if predicate(message):
                return message
        return None

    def command(self, command, *params, expect=mavlink.MAV_RESULT_ACCEPTED):
        params = list(params) + [0] * (7 - len(params))
        self.link.mav.command_long_send(*VEHICLE, command, 0, *params)
        ack = self.wait(lambda message: message.get_type() == "COMMAND_ACK" and message.command == command, REPLY_WAIT_S)
        return ack is not None and ack.result == expect

    def set_parameter(self, name, value):
        self.link.mav.param_set_send(*VEHICLE, name.encode(), value, mavlink.MAV_PARAM_TYPE_REAL32)
        return self.wait(lambda message: message.get_type() == "PARAM_VALUE" and message.param_id == name, REPLY_WAIT_S)

    def run(self):
        try:
            self._handshake()
            self._parameters()
            items = self._upload_mission()
            self._download_mission(items)
            self._fly_mission()
            self._guided_flight()
        finally:
            self.simulator.terminate()
        print("\nstatus texts:", self.texts)
        passed = all(self.results)
        print("ALL PASSED" if passed else "FAILED")
        return passed

    def _handshake(self):
        heartbeat = self.wait(lambda message: message.get_type() == "HEARTBEAT", HEARTBEAT_WAIT_S)
        self.check("heartbeat received", heartbeat is not None)
        expected_type = mavlink.MAV_TYPE_QUADROTOR if self.frame == "quad_x" else mavlink.MAV_TYPE_HEXAROTOR
        self.check("reports ArduPilot-style copter",
                   heartbeat.autopilot == mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA and heartbeat.type == expected_type)

    def _parameters(self):
        self.link.mav.param_request_list_send(*VEHICLE)
        received = {}

        def collect(message):
            if message.get_type() == "PARAM_VALUE":
                received[message.param_id] = message.param_value
                return len(received) >= message.param_count
            return False
        self.wait(collect, 10)
        self.check("parameter list downloaded", len(received) >= MIN_PARAMETERS and "SIM_MASS" in received,
                   f"{len(received)} parameters")
        reply = self.set_parameter("SIM_MASS", self.mass_kg)
        self.check("set SIM_MASS", reply is not None and abs(reply.param_value - self.mass_kg) < 1e-3, f"{self.mass_kg:g} kg")
        speed = REFERENCE_SPEED_CMS * speed_scale(self.mass_kg)
        for name, value in (("WPNAV_SPEED", speed), ("WPNAV_SPEED_UP", speed), ("WPNAV_SPEED_DN", speed * DESCENT_SHARE),
                            ("SIM_WIND_SPD", TEST_WIND_SPEED_MPS), ("SIM_WIND_DIR", TEST_WIND_DIRECTION_DEG)):
            self.set_parameter(name, value)
        self.command(mavlink.MAV_CMD_REQUEST_MESSAGE, mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION)
        version = self.wait(lambda message: message.get_type() == "AUTOPILOT_VERSION", REPLY_WAIT_S)
        self.check("autopilot version answered", version is not None)

    def _upload_mission(self):
        """Home, take-off 20 m, two waypoints, a 15 m circle, return to launch."""
        relative = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        items = [(mavlink.MAV_CMD_NAV_WAYPOINT, mavlink.MAV_FRAME_GLOBAL, HOME.latitude, HOME.longitude, 0, 0, 0),
                 (mavlink.MAV_CMD_NAV_TAKEOFF, relative, 0, 0, 20, 0, 0),
                 (mavlink.MAV_CMD_NAV_WAYPOINT, relative, *HOME.offset(60, 0), 30, 0, 0),
                 (mavlink.MAV_CMD_NAV_WAYPOINT, relative, *HOME.offset(60, 60), 40, 2, 0),
                 (mavlink.MAV_CMD_NAV_LOITER_TURNS, relative, *HOME.offset(0, 60), 30, 1, CIRCLE_RADIUS_M),
                 (mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, relative, 0, 0, 0, 0, 0)]
        self.link.mav.mission_count_send(*VEHICLE, len(items), mavlink.MAV_MISSION_TYPE_MISSION)

        def serve(message):
            if message.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
                command, frame, latitude, longitude, altitude, param1, param3 = items[message.seq]
                self.link.mav.mission_item_int_send(*VEHICLE, message.seq, frame, command, 0, 1, param1, 0, param3, 0,
                                                    int(latitude * DEGREES_E7), int(longitude * DEGREES_E7), altitude,
                                                    mavlink.MAV_MISSION_TYPE_MISSION)
            return message.get_type() == "MISSION_ACK"
        ack = self.wait(serve, 15)
        self.check("mission upload accepted", ack is not None and ack.type == mavlink.MAV_MISSION_ACCEPTED, f"{len(items)} items")
        return items

    def _download_mission(self, items):
        self.link.mav.mission_request_list_send(*VEHICLE, mavlink.MAV_MISSION_TYPE_MISSION)
        count = self.wait(lambda message: message.get_type() == "MISSION_COUNT", REPLY_WAIT_S)
        downloaded = []
        for seq in range(count.count if count else 0):
            self.link.mav.mission_request_int_send(*VEHICLE, seq, mavlink.MAV_MISSION_TYPE_MISSION)
            downloaded.append(self.wait(lambda message: message.get_type() == "MISSION_ITEM_INT" and message.seq == seq,
                                        REPLY_WAIT_S))
        self.link.mav.mission_ack_send(*VEHICLE, mavlink.MAV_MISSION_ACCEPTED, mavlink.MAV_MISSION_TYPE_MISSION)
        self.check("mission download matches", count is not None and count.count == len(items)
                   and all(item is not None for item in downloaded)
                   and downloaded[CIRCLE_ITEM_SEQ].command == mavlink.MAV_CMD_NAV_LOITER_TURNS
                   and abs(downloaded[CIRCLE_ITEM_SEQ].param3 - CIRCLE_RADIUS_M) < 1e-3)

    def _fly_mission(self):
        set_mode = mavlink.MAV_CMD_DO_SET_MODE
        self.check("unsupported mode rejected", self.command(set_mode, 1, UNSUPPORTED_MODE, expect=mavlink.MAV_RESULT_FAILED))
        self.check("mode AUTO", self.command(set_mode, 1, CopterMode.AUTO))
        self.check("arm", self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1))
        heartbeat = self.wait(lambda message: message.get_type() == "HEARTBEAT" and message.base_mode & ARMED_FLAG, REPLY_WAIT_S)
        self.check("heartbeat shows armed in AUTO", heartbeat is not None and heartbeat.custom_mode == CopterMode.AUTO)
        self.check("mission start", self.command(mavlink.MAV_CMD_MISSION_START))
        finished = self.wait(self._disarmed, MISSION_FLIGHT_S)
        self.check("mission flown and vehicle disarmed", finished is not None)
        self.check("waypoints reported reached", all(seq in self.reached for seq in WAYPOINT_ITEM_SEQS), str(self.reached))
        low, high = MISSION_TOP_ALTITUDE_RANGE_M
        self.check("climbed to mission altitude", low < self.max_height < high, f"max {self.max_height:.1f} m")
        self.check("motor PWM reported", PWM_REPORTED_RANGE_US[0] < self.max_pwm <= PWM_REPORTED_RANGE_US[1],
                   f"max {self.max_pwm} us")
        position = self.wait(lambda message: message.get_type() == "GLOBAL_POSITION_INT", REPLY_WAIT_S)
        north = (position.lat / DEGREES_E7 - HOME.latitude) * METRES_PER_DEGREE_LATITUDE
        east = (position.lon / DEGREES_E7 - HOME.longitude) * HOME.metres_per_degree_longitude
        offset = math.hypot(north, east)
        self.check("landed at home", position.relative_alt < LANDED_HEIGHT_MM and offset < HOME_RADIUS_LENGTHS * length_scale(self.mass_kg),
                   f"{offset:.2f} m from home, height {position.relative_alt / MILLIMETRES:.2f} m")
        self.check("no crash reported", not any("CRASH" in text for text in self.texts))

    @staticmethod
    def _disarmed(message):
        return message.get_type() == "HEARTBEAT" and not (message.base_mode & ARMED_FLAG)

    def _guided_flight(self):
        self.check("mode GUIDED", self.command(mavlink.MAV_CMD_DO_SET_MODE, 1, CopterMode.GUIDED))
        self.check("arm again", self.command(mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1))
        self.check("guided take-off", self.command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, GUIDED_TAKEOFF_M))
        climbed = self.wait(lambda message: message.get_type() == "GLOBAL_POSITION_INT"
                            and message.relative_alt > (GUIDED_TAKEOFF_M - 1) * MILLIMETRES, CLIMB_WAIT_S)
        self.check(f"reached {GUIDED_TAKEOFF_M} m", climbed is not None)
        north, east, height = GUIDED_GOTO
        latitude, longitude = HOME.offset(north, east)
        self.link.mav.command_int_send(*VEHICLE, mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, mavlink.MAV_CMD_DO_REPOSITION, 0, 0,
                                       -1, 0, 0, 0, int(latitude * DEGREES_E7), int(longitude * DEGREES_E7), height)
        radius = GOTO_RADIUS_LENGTHS * length_scale(self.mass_kg)
        arrived = self.wait(lambda message: message.get_type() == "GLOBAL_POSITION_INT"
                            and abs((message.lat / DEGREES_E7 - latitude) * METRES_PER_DEGREE_LATITUDE) < radius
                            and abs(message.relative_alt - height * MILLIMETRES) < GOTO_HEIGHT_TOLERANCE_M * MILLIMETRES, GOTO_WAIT_S)
        self.check("guided go-to reached", arrived is not None)
        self.check("land command", self.command(mavlink.MAV_CMD_NAV_LAND))
        self.check("landed and disarmed", self.wait(self._disarmed, LANDING_WAIT_S) is not None)
        self.check("no crash reported (guided)", not any("CRASH" in text for text in self.texts))
