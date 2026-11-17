"""Fly the comparison mission: in ArduCopter SITL (keeping its DataFlash log) and with a network in this
repository's simulator (saving a log with the same columns)."""
import shutil
import time
from pathlib import Path

import numpy as np
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.ardupilot_comparison.mission_definition import (ARDUPILOT_PARAMETERS, HOME, MASS_KG, NETWORK_PARAMETERS,
                                                              mission_item_rows)
from nn_drone.ardupilot_comparison.sitl_session import SitlSession
from nn_drone.geometry import euler_from_quaternion
from nn_drone.ground_link.mission_items import MissionItem
from nn_drone.ground_link.vehicle import CopterMode, SimulatedVehicle
from nn_drone.paths import default_weights

MISSION_UPLOAD_ATTEMPTS = 4
MISSION_UPLOAD_WAIT_S = 15
RETRY_PAUSE_S = 2
ARMING_WAIT_S = 240
MISSION_WAIT_S = 900
LOG_SETTLE_S = 3
DEGREES_E7 = 1e7
STREAM_RATE_HZ = 4
NETWORK_FLIGHT_LIMIT_S = 900
FULL_TURN_DEG = 360
LOG_COLUMNS = ["t", "north", "east", "alt", "roll_deg", "pitch_deg", "heading_deg", "volt"]


def fly_ardupilot(ardupilot_root, output_dir, speedup=10):
    output_dir = Path(output_dir).resolve()
    session = SitlSession(ardupilot_root, output_dir / "sitl_run", HOME, speedup, echo_status=True)
    try:
        session.wait_for_heartbeat()
        print("connected to ArduCopter SITL", flush=True)
        session.request_all_streams(STREAM_RATE_HZ)
        for name, value in ARDUPILOT_PARAMETERS.items():
            accepted = session.set_parameter(name, value)
            print(f"   set {name} = {value}: {'ok' if accepted else 'FAILED'}", flush=True)
            if not accepted:
                raise RuntimeError(f"parameter {name} was not accepted")
        session.wait_for_gps()
        _upload_mission(session)
        if not session.command(mavlink.MAV_CMD_DO_SET_MODE, 1, CopterMode.AUTO):
            raise RuntimeError("could not set AUTO")
        if not session.arm_when_ready(ARMING_WAIT_S):
            raise RuntimeError("arming refused")
        print("armed", flush=True)
        session.command(mavlink.MAV_CMD_MISSION_START)
        if session.wait_until_disarmed(MISSION_WAIT_S) is None:
            raise RuntimeError("mission did not finish")
        print("mission finished, vehicle disarmed", flush=True)
        time.sleep(LOG_SETTLE_S)
    finally:
        session.close()
    destination = output_dir / "ardupilot_sitl.BIN"
    shutil.copy(session.latest_log(), destination)
    print("log:", destination)
    return destination


def _upload_mission(session):
    rows = mission_item_rows()

    def serve(message):
        if message.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            command, frame, latitude, longitude, height, param1, param3 = rows[message.seq]
            session.link.mav.mission_item_int_send(1, 1, message.seq, frame, command, 0, 1, param1, 0, param3, 0,
                                                   int(latitude * DEGREES_E7), int(longitude * DEGREES_E7), height,
                                                   mavlink.MAV_MISSION_TYPE_MISSION)
        return message.get_type() == "MISSION_ACK"
    for attempt in range(1, MISSION_UPLOAD_ATTEMPTS + 1):
        session.link.mav.mission_count_send(1, 1, len(rows), mavlink.MAV_MISSION_TYPE_MISSION)
        ack = session.wait(serve, MISSION_UPLOAD_WAIT_S)
        if ack is not None and ack.type == mavlink.MAV_MISSION_ACCEPTED:
            print("mission uploaded", flush=True)
            return
        print("   mission upload attempt", attempt, "failed:", None if ack is None else ack.type, flush=True)
        time.sleep(RETRY_PAUSE_S)
    raise RuntimeError("mission upload failed")


def comparison_vehicle(frame, sensor_noise=True):
    vehicle = SimulatedVehicle(frame, default_weights(frame), HOME, MASS_KG, log=lambda *_: None)
    if not sensor_noise:
        vehicle.set_parameter("SIM_NOISE", 0.0)
    for name, value in NETWORK_PARAMETERS.items():
        vehicle.set_parameter(name, value)
    return vehicle


def fly_network(output_dir, frame="quad_x", sensor_noise=True):
    vehicle = comparison_vehicle(frame, sensor_noise)
    vehicle.mission = [MissionItem(seq=seq, frame=frame_id, command=command, param1=param1, param3=param3,
                                   x=latitude, y=longitude, z=height)
                       for seq, (command, frame_id, latitude, longitude, height, param1, param3) in enumerate(mission_item_rows())]
    vehicle.mode = CopterMode.AUTO
    if not (vehicle.arm() and vehicle.start_mission()):
        raise RuntimeError("the network vehicle refused the mission")
    rows = []
    state = vehicle.env.state
    while vehicle.armed and vehicle.time < NETWORK_FLIGHT_LIMIT_S:
        vehicle.step()
        roll, pitch, yaw = (angle[0] for angle in euler_from_quaternion(state.attitude))
        position = state.position[0]
        rows.append([vehicle.time, position[0], -position[1], position[2], np.degrees(roll), np.degrees(-pitch),
                     np.degrees(-yaw) % FULL_TURN_DEG, state.battery_voltage[0], *vehicle.pwm])
    rows = np.array(rows)
    motor_columns = [f"pwm{motor + 1}" for motor in range(vehicle.env.motor_count)]
    path = Path(output_dir) / f"network_{frame}{'' if sensor_noise else '_nonoise'}.npz"
    np.savez_compressed(path, data=rows.astype(np.float32), columns=np.array(LOG_COLUMNS + motor_columns))
    print(f"network {frame}: {len(rows)} steps, {vehicle.time:.1f} s, ended at N {position[0]:.2f} E {-position[1]:.2f}, "
          f"crashed {vehicle.crashed}, touchdown {state.touchdown_speed[0]:.2f} m/s")
    return path
