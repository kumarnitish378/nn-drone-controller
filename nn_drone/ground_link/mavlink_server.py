"""MAVLink front-end: answers a ground station (QGroundControl, Mission Planner) as an ArduCopter-type vehicle."""
import math
import os
import socket
import time

import numpy as np

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil  # noqa: E402
from pymavlink.dialects.v20 import ardupilotmega as mavlink  # noqa: E402

from nn_drone.ground_link.mission_items import MissionItem, home_item  # noqa: E402
from nn_drone.ground_link.vehicle import CopterMode  # noqa: E402
from nn_drone.units import (CENTIMETRES_PER_METRE, MICROSECONDS_PER_SECOND, MILLIMETRES_PER_METRE,  # noqa: E402
                           MILLISECONDS_PER_SECOND)

DEGREES_E7 = 1e7
UINT32_MASK = 0xFFFFFFFF
FORCE_ARM_MAGIC = 21196
GROUND_STATION_SYSTEM = 255
GROUND_STATION_COMPONENT = 190
AUTOPILOT_COMPONENT = 1
GUIDED_GOTO_CURRENT_FLAG = 2
REPORTED_FIRMWARE_VERSION = (4, 5, 0)
FIRMWARE_TYPE_OFFICIAL = 255
PARAMETERS_PER_TICK = 10
SERVO_CHANNELS = 8
GPS_FIX_3D = 3
GPS_HDOP_CM, GPS_VDOP_CM = 80, 120
GPS_SATELLITES = 14
CPU_LOAD_PERMILLE = 200
UNKNOWN = -1
EKF_ALL_GOOD_FLAGS = 831
EKF_VARIANCE = 0.05
MAX_WAYPOINT_DISTANCE_M = 65535
MISSION_REQUEST_RETRY_S = 1.0
ALL_SENSORS_EXCEPT_RC = 0x0FFFFFFF & ~(1 << 16)
CENTIDEGREES_PER_TURN = 36000
CENTI = 100
MILLIVOLTS_PER_VOLT = 1000
FAR_PAST = -1e9
MICROSECONDS_PER_MILLISECOND = 1000


class TelemetrySchedule:
    """Message groups and how often they are sent, in simulated seconds."""
    HEARTBEAT_S = 1.0
    FAST_S = 0.1
    SLOW_S = 0.5
    HOME_S = 5.0


def firmware_version_word(major, minor, patch, firmware_type=FIRMWARE_TYPE_OFFICIAL):
    return (major << 24) | (minor << 16) | (patch << 8) | firmware_type


def open_ground_station_link(address, system_id):
    """pymavlink's "udpout" binds the destination port locally on Windows, which collides with a ground station on
    the same PC. Bind an ephemeral local port instead and send to the ground station from there."""
    host, port = address.rsplit(":", 1)
    connection = mavutil.mavlink_connection("udpin:0.0.0.0:0", source_system=system_id,
                                            source_component=AUTOPILOT_COMPONENT, dialect="ardupilotmega")
    destination = (socket.gethostbyname(host), int(port))
    connection.clients.add(destination)
    connection.clients_last_alive[destination] = time.time()
    return connection


class MissionUpload:
    def __init__(self, count):
        self.count = count
        self.items = []
        self.last_activity = time.time()

    @property
    def next_seq(self):
        return len(self.items)

    @property
    def complete(self):
        return len(self.items) == self.count


class MavlinkServer:
    def __init__(self, vehicle, ground_station_address):
        self.vehicle = vehicle
        self.connection = open_ground_station_link(ground_station_address, int(vehicle.parameters["SYSID_THISMAV"]))
        self.mav = self.connection.mav
        self.parameter_queue = []
        self.last_sent = {}
        self.upload = None
        self.ground_station = (GROUND_STATION_SYSTEM, GROUND_STATION_COMPONENT)
        self.handlers = {
            "HEARTBEAT": self._on_heartbeat, "PARAM_REQUEST_LIST": self._on_parameter_list_request,
            "PARAM_REQUEST_READ": self._on_parameter_read, "PARAM_SET": self._on_parameter_set,
            "MISSION_COUNT": self._on_mission_count, "MISSION_ITEM_INT": self._on_mission_item, "MISSION_ITEM": self._on_mission_item,
            "MISSION_REQUEST_LIST": self._on_mission_list_request, "MISSION_REQUEST_INT": self._on_mission_request,
            "MISSION_REQUEST": self._on_mission_request, "MISSION_CLEAR_ALL": self._on_mission_clear,
            "MISSION_SET_CURRENT": self._on_mission_set_current, "SET_MODE": self._on_set_mode,
            "SET_POSITION_TARGET_GLOBAL_INT": self._on_position_target, "TIMESYNC": self._on_timesync,
            "COMMAND_INT": self._on_command_int, "COMMAND_LONG": self._on_command_long}
        self.commands = {
            mavlink.MAV_CMD_COMPONENT_ARM_DISARM: self._arm_or_disarm,
            mavlink.MAV_CMD_NAV_TAKEOFF: lambda message: self.vehicle.takeoff(message.param7),
            mavlink.MAV_CMD_DO_SET_MODE: lambda message: self.vehicle.set_mode(int(message.param2)),
            mavlink.MAV_CMD_NAV_LAND: lambda message: self.vehicle.set_mode(CopterMode.LAND),
            mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH: lambda message: self.vehicle.set_mode(CopterMode.RTL),
            mavlink.MAV_CMD_MISSION_START: lambda message: self.vehicle.start_mission(),
            mavlink.MAV_CMD_DO_CHANGE_SPEED: self._change_speed,
            mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES: self._send_version_and_accept,
            mavlink.MAV_CMD_GET_HOME_POSITION: self._send_home_and_accept,
            mavlink.MAV_CMD_SET_MESSAGE_INTERVAL: lambda message: True,
            mavlink.MAV_CMD_DO_SET_HOME: lambda message: True}

    def _due(self, key, period):
        now = self.vehicle.time
        if now - self.last_sent.get(key, FAR_PAST) >= period:
            self.last_sent[key] = now
            return True
        return False

    def _boot_milliseconds(self):
        return int(self.vehicle.time * MILLISECONDS_PER_SECOND) & UINT32_MASK

    def _acknowledge(self, command, accepted=True, result=None):
        if result is None:
            result = mavlink.MAV_RESULT_ACCEPTED if accepted else mavlink.MAV_RESULT_FAILED
        self.mav.command_ack_send(command, result)

    def _send_mission_ack(self, mission_type=mavlink.MAV_MISSION_TYPE_MISSION):
        self.mav.mission_ack_send(*self.ground_station, mavlink.MAV_MISSION_ACCEPTED, mission_type)

    def _send_parameter(self, name):
        names = list(self.vehicle.parameters)
        self.mav.param_value_send(name.encode(), float(self.vehicle.parameters[name]), mavlink.MAV_PARAM_TYPE_REAL32,
                                  len(names), names.index(name))

    def _send_home(self):
        home = self.vehicle.home
        self.mav.home_position_send(int(home.latitude * DEGREES_E7), int(home.longitude * DEGREES_E7),
                                    int(home.altitude_asl * MILLIMETRES_PER_METRE), 0, 0, 0, [1, 0, 0, 0], 0, 0, 0)

    def _send_version(self):
        capabilities = (mavlink.MAV_PROTOCOL_CAPABILITY_MISSION_INT | mavlink.MAV_PROTOCOL_CAPABILITY_COMMAND_INT
                        | mavlink.MAV_PROTOCOL_CAPABILITY_PARAM_FLOAT | mavlink.MAV_PROTOCOL_CAPABILITY_MAVLINK2
                        | mavlink.MAV_PROTOCOL_CAPABILITY_SET_POSITION_TARGET_GLOBAL_INT)
        unused_hashes = [0] * 8
        self.mav.autopilot_version_send(capabilities, firmware_version_word(*REPORTED_FIRMWARE_VERSION), 0, 0, 0,
                                        unused_hashes, unused_hashes, unused_hashes, 0, 0, 0)

    def _send_version_and_accept(self, message):
        self._acknowledge(message.command)
        self._send_version()
        return None

    def _send_home_and_accept(self, message):
        self._acknowledge(message.command)
        self._send_home()
        return None

    def _send_mission_item(self, seq, as_int):
        vehicle = self.vehicle
        items = vehicle.mission or [home_item(vehicle.home)]
        item = items[seq]
        current = 1 if seq == vehicle.seq and vehicle.mission_running else 0
        fields = (item.frame, item.command, current, item.autocontinue, item.param1, item.param2, item.param3, item.param4)
        if as_int:
            self.mav.mission_item_int_send(*self.ground_station, seq, *fields, int(round(item.x * DEGREES_E7)),
                                           int(round(item.y * DEGREES_E7)), item.z, mavlink.MAV_MISSION_TYPE_MISSION)
        else:
            self.mav.mission_item_send(*self.ground_station, seq, *fields, item.x, item.y, item.z,
                                       mavlink.MAV_MISSION_TYPE_MISSION)

    def handle(self, message):
        handler = self.handlers.get(message.get_type())
        if handler is not None:
            handler(message)

    def _on_heartbeat(self, message):
        if message.type == mavlink.MAV_TYPE_GCS:
            self.ground_station = (message.get_srcSystem(), message.get_srcComponent())

    def _on_parameter_list_request(self, message):
        self.parameter_queue = list(self.vehicle.parameters)

    def _on_parameter_read(self, message):
        parameters = self.vehicle.parameters
        name = message.param_id if isinstance(message.param_id, str) else message.param_id.decode()
        name = name.rstrip("\x00")
        names = list(parameters)
        if name not in parameters and 0 <= message.param_index < len(names):
            name = names[message.param_index]
        if name in parameters:
            self._send_parameter(name)

    def _on_parameter_set(self, message):
        name = message.param_id.rstrip("\x00")
        if self.vehicle.set_parameter(name, message.param_value):
            self._send_parameter(name)

    def _on_mission_count(self, message):
        mission_type = getattr(message, "mission_type", mavlink.MAV_MISSION_TYPE_MISSION)
        if mission_type != mavlink.MAV_MISSION_TYPE_MISSION:
            self._send_mission_ack(mission_type)
        elif message.count == 0:
            self.vehicle.mission = []
            self._send_mission_ack()
        else:
            self.upload = MissionUpload(message.count)
            self.mav.mission_request_int_send(*self.ground_station, 0, mavlink.MAV_MISSION_TYPE_MISSION)

    def _on_mission_item(self, message):
        scale = 1 / DEGREES_E7 if message.get_type() == "MISSION_ITEM_INT" else 1.0
        item = MissionItem(seq=message.seq, frame=message.frame, command=message.command, current=message.current,
                           autocontinue=message.autocontinue, param1=message.param1, param2=message.param2,
                           param3=message.param3, param4=message.param4, x=message.x * scale, y=message.y * scale, z=message.z)
        vehicle, upload = self.vehicle, self.upload
        if message.current == GUIDED_GOTO_CURRENT_FLAG and upload is None:
            vehicle.go_to(item.x, item.y, vehicle.home.height_above_home(item.z, item.frame))
            self._send_mission_ack()
        elif upload is not None and message.seq == upload.next_seq:
            upload.items.append(item)
            upload.last_activity = time.time()
            if upload.complete:
                vehicle.mission, self.upload = upload.items, None
                self._send_mission_ack()
                vehicle.say(f"Mission received: {len(vehicle.mission) - 1} items")
            else:
                self.mav.mission_request_int_send(*self.ground_station, upload.next_seq, mavlink.MAV_MISSION_TYPE_MISSION)

    def _on_mission_list_request(self, message):
        mission_type = getattr(message, "mission_type", mavlink.MAV_MISSION_TYPE_MISSION)
        count = max(len(self.vehicle.mission), 1) if mission_type == mavlink.MAV_MISSION_TYPE_MISSION else 0
        self.mav.mission_count_send(*self.ground_station, count, mission_type)

    def _on_mission_request(self, message):
        mission_type = getattr(message, "mission_type", mavlink.MAV_MISSION_TYPE_MISSION)
        if mission_type == mavlink.MAV_MISSION_TYPE_MISSION and message.seq < max(len(self.vehicle.mission), 1):
            self._send_mission_item(message.seq, message.get_type() == "MISSION_REQUEST_INT")

    def _on_mission_clear(self, message):
        self.vehicle.mission = []
        self._send_mission_ack()

    def _on_mission_set_current(self, message):
        if 0 < message.seq < len(self.vehicle.mission):
            self.vehicle.jump_to_item(message.seq)

    def _on_set_mode(self, message):
        self.vehicle.set_mode(int(message.custom_mode))

    def _on_position_target(self, message):
        vehicle = self.vehicle
        vehicle.go_to(message.lat_int / DEGREES_E7, message.lon_int / DEGREES_E7,
                      vehicle.home.height_above_home(message.alt, message.coordinate_frame))

    def _on_timesync(self, message):
        if message.tc1 == 0:
            self.mav.timesync_send(time.time_ns(), message.ts1)

    def _on_command_int(self, message):
        if message.command == mavlink.MAV_CMD_DO_REPOSITION:
            vehicle = self.vehicle
            accepted = vehicle.go_to(message.x / DEGREES_E7, message.y / DEGREES_E7,
                                     vehicle.home.height_above_home(message.z, message.frame))
            self._acknowledge(message.command, accepted)
        else:
            self._acknowledge(message.command, result=mavlink.MAV_RESULT_UNSUPPORTED)

    def _on_command_long(self, message):
        if message.command == mavlink.MAV_CMD_REQUEST_MESSAGE:
            self._answer_message_request(message)
            return
        command = self.commands.get(message.command)
        if command is None:
            self._acknowledge(message.command, result=mavlink.MAV_RESULT_UNSUPPORTED)
            return
        accepted = command(message)
        if accepted is not None:
            self._acknowledge(message.command, accepted)

    def _answer_message_request(self, message):
        requested = int(message.param1)
        senders = {mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION: self._send_version,
                   mavlink.MAVLINK_MSG_ID_HOME_POSITION: self._send_home}
        if requested not in senders:
            self._acknowledge(message.command, result=mavlink.MAV_RESULT_UNSUPPORTED)
            return
        self._acknowledge(message.command)
        senders[requested]()

    def _arm_or_disarm(self, message):
        force = abs(message.param2 - FORCE_ARM_MAGIC) < 1
        return self.vehicle.arm(force) if message.param1 >= 0.5 else self.vehicle.disarm(force)

    def _change_speed(self, message):
        if message.param2 > 0:
            self.vehicle.parameters["WPNAV_SPEED"] = message.param2 * CENTIMETRES_PER_METRE
            self._send_parameter("WPNAV_SPEED")
        return True

    def send_telemetry(self):
        vehicle = self.vehicle
        if self._due("heartbeat", TelemetrySchedule.HEARTBEAT_S):
            self._send_heartbeat()
        telemetry = None
        if self._due("fast", TelemetrySchedule.FAST_S):
            telemetry = vehicle.telemetry()
            self._send_fast_group(telemetry)
        if self._due("slow", TelemetrySchedule.SLOW_S):
            self._send_slow_group(telemetry or vehicle.telemetry())
        if self._due("home", TelemetrySchedule.HOME_S):
            self._send_home()
            home = vehicle.home
            self.mav.gps_global_origin_send(int(home.latitude * DEGREES_E7), int(home.longitude * DEGREES_E7),
                                            int(home.altitude_asl * MILLIMETRES_PER_METRE))
        for _ in range(min(len(self.parameter_queue), PARAMETERS_PER_TICK)):
            self._send_parameter(self.parameter_queue.pop(0))
        while vehicle.status_texts:
            severity, text = vehicle.status_texts.pop(0)
            self.mav.statustext_send(severity, text.encode())
        while vehicle.reached_items:
            self.mav.mission_item_reached_send(vehicle.reached_items.pop(0))
        if self.upload is not None and time.time() - self.upload.last_activity > MISSION_REQUEST_RETRY_S:
            self.upload.last_activity = time.time()
            self.mav.mission_request_int_send(*self.ground_station, self.upload.next_seq, mavlink.MAV_MISSION_TYPE_MISSION)

    def _send_heartbeat(self):
        vehicle = self.vehicle
        base_mode = (mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED
                     | mavlink.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED | mavlink.MAV_MODE_FLAG_GUIDED_ENABLED)
        if vehicle.armed:
            base_mode |= mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        vehicle_type = mavlink.MAV_TYPE_QUADROTOR if vehicle.frame == "quad_x" else mavlink.MAV_TYPE_HEXAROTOR
        if vehicle.crashed:
            system_state = mavlink.MAV_STATE_CRITICAL
        else:
            system_state = mavlink.MAV_STATE_ACTIVE if vehicle.armed else mavlink.MAV_STATE_STANDBY
        self.mav.heartbeat_send(vehicle_type, mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA, base_mode, vehicle.mode, system_state)

    def _send_fast_group(self, telemetry):
        boot_ms = self._boot_milliseconds()
        heading_cdeg = int(telemetry.heading_deg * CENTI) % CENTIDEGREES_PER_TURN
        self.mav.attitude_send(boot_ms, telemetry.roll, telemetry.pitch, telemetry.yaw, telemetry.roll_rate,
                               telemetry.pitch_rate, telemetry.yaw_rate)
        self.mav.global_position_int_send(boot_ms, int(telemetry.latitude * DEGREES_E7), int(telemetry.longitude * DEGREES_E7),
                                          int(telemetry.altitude_asl * MILLIMETRES_PER_METRE),
                                          int(telemetry.height_above_home * MILLIMETRES_PER_METRE),
                                          int(telemetry.velocity_north * CENTI), int(telemetry.velocity_east * CENTI),
                                          int(telemetry.velocity_down * CENTI), heading_cdeg)
        self.mav.vfr_hud_send(telemetry.ground_speed, telemetry.ground_speed, int(telemetry.heading_deg),
                              int(telemetry.throttle_percent), telemetry.altitude_asl, -telemetry.velocity_down)
        channels = ([int(pwm) for pwm in self.vehicle.pwm] + [0] * SERVO_CHANNELS)[:SERVO_CHANNELS]
        self.mav.servo_output_raw_send(boot_ms * MICROSECONDS_PER_MILLISECOND & UINT32_MASK, 0, *channels)

    def _send_slow_group(self, telemetry):
        vehicle = self.vehicle
        heading_cdeg = int(telemetry.heading_deg * CENTI) % CENTIDEGREES_PER_TURN
        self.mav.gps_raw_int_send(int(vehicle.time * MICROSECONDS_PER_SECOND), GPS_FIX_3D, int(telemetry.latitude * DEGREES_E7),
                                  int(telemetry.longitude * DEGREES_E7), int(telemetry.altitude_asl * MILLIMETRES_PER_METRE),
                                  GPS_HDOP_CM, GPS_VDOP_CM, int(telemetry.ground_speed * CENTI), heading_cdeg, GPS_SATELLITES)
        self.mav.sys_status_send(ALL_SENSORS_EXCEPT_RC, ALL_SENSORS_EXCEPT_RC, ALL_SENSORS_EXCEPT_RC, CPU_LOAD_PERMILLE,
                                 int(telemetry.battery_voltage * MILLIVOLTS_PER_VOLT), UNKNOWN, UNKNOWN, 0, 0, 0, 0, 0, 0)
        landed_state = mavlink.MAV_LANDED_STATE_IN_AIR if vehicle.airborne else mavlink.MAV_LANDED_STATE_ON_GROUND
        self.mav.extended_sys_state_send(mavlink.MAV_VTOL_STATE_UNDEFINED, landed_state)
        self.mav.mission_current_send(vehicle.seq if vehicle.mission_running else 0)
        self.mav.ekf_status_report_send(EKF_ALL_GOOD_FLAGS, *[EKF_VARIANCE] * 5)
        to_carrot = vehicle.carrot - vehicle.position
        bearing = int(math.degrees(math.atan2(-to_carrot[1], to_carrot[0])) % 360)
        self.mav.nav_controller_output_send(0, 0, bearing, bearing, int(min(np.hypot(to_carrot[0], to_carrot[1]),
                                                                           MAX_WAYPOINT_DISTANCE_M)), float(to_carrot[2]), 0, 0)

    def poll(self):
        while True:
            try:
                message = self.connection.recv_match(blocking=False)
            except ConnectionResetError:
                # Windows reports "port unreachable" for an earlier datagram (ground station not open yet) on the next receive.
                continue
            if message is None:
                return
            self.handle(message)
