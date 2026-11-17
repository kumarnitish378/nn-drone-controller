"""Stability, ArduCopter SITL against the networks:
 A. attitude steadiness in the hover and after take-off (from the mission logs),
 B. gust test: hover at 20 m, wind steps from 0 to 8 m/s for 15 s, then back to 0,
 C. extra control delay the network tolerates (network only)."""
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.ardupilot_comparison.comparison import HOVER_WINDOW_S, hover_window, tilt_from_roll_pitch
from nn_drone.ardupilot_comparison.flight_logs import (ARDUPILOT_NAME, read_dataflash, read_network_flight, seconds,
                                                      true_positions)
from nn_drone.ardupilot_comparison.mission_definition import HOME, MASS_KG
from nn_drone.ardupilot_comparison.sitl_session import SitlSession
from nn_drone.geometry import euler_from_quaternion, tilt_degrees
from nn_drone.ground_link.navigation import Goal
from nn_drone.ground_link.vehicle import CopterMode, Phase, SimulatedVehicle
from nn_drone.paths import default_weights
from nn_drone.simulation.scaling import time_scale

GUST_SPEED_MPS = 8.0
SETTLE_BEFORE_GUST_S = 12.0
GUST_DURATION_S = 15.0
OBSERVE_AFTER_GUST_S = 15.0
HOVER_HEIGHT_M = 20.0
REACHED_HEIGHT_MARGIN_M = 0.3
GUST_FROM_EAST_DEG = 90.0
BASELINE_BEFORE_GUST_S = 5
SETTLED_RADIUS_M = 0.3
LATE_GUST_WINDOW_S = 4
NETWORK_GUST_LIMIT_S = 200
CLIMB_OUT_WINDOW_S = (3, 9)
ATTITUDE_SAMPLE_S = 0.1
SITL_SPEEDUP = 5
SITL_STREAM_HZ = 10
SITL_CLIMB_SPEED_MPS = 3.0
SITL_ARMING_WAIT_S = 200
SITL_STEP_WAIT_S = 120
SITL_LANDING_WAIT_S = 200
EXTRA_OBSERVATION_S = 2
LOG_SETTLE_S = 2
MILLISECONDS = 1000.0
MILLIMETRES = 1000
NOMINAL_CONTROL_PERIOD_MS = 10.0
DELAY_STEPS = (0, 1, 2, 3, 4, 5, 6, 8)
DELAY_TEST_WIND_MPS = 2.0
DELAY_TEST_HEIGHT_M = 10.0
DELAY_TEST_SIDESTEP_M = 5.0
DELAY_TEST_SIDESTEP_AT_S = 20
DELAY_TEST_DURATION_S = 60
RESULT_DECIMALS = 3
QUAD_NAME, HEXA_NAME = "Neural network, quad", "Neural network, hexa"


def unwrap_degrees(angles):
    return np.degrees(np.unwrap(np.radians(angles)))


def _ardupilot_attitude(log_path):
    attitude = np.array([[seconds(message), message.Roll, message.Pitch, message.Yaw]
                         for message in read_dataflash(log_path, ("ATT",))["ATT"]])
    attitude[:, 0] -= attitude[0, 0]
    return attitude


def _network_attitude(log_path):
    flight = read_network_flight(log_path, "")
    sample_time = np.arange(0, flight.time[-1], ATTITUDE_SAMPLE_S)
    return np.stack([sample_time, np.interp(sample_time, flight.time, flight.roll), np.interp(sample_time, flight.time, flight.pitch),
                     np.interp(sample_time, flight.time, unwrap_degrees(flight.heading))], 1)


def _hover_measurement(time, north, east, height):
    start = hover_window(time, north, east, height)[0]
    return start, start + HOVER_WINDOW_S


def _ardupilot_hover_window(log_path):
    truth = true_positions(read_dataflash(log_path, ("SIM2",))["SIM2"], first_sample=0)
    truth[:, 0] -= truth[0, 0]
    truth[:, 1:] -= truth[0, 1:]
    return _hover_measurement(truth[:, 0], truth[:, 1], truth[:, 2], truth[:, 3])


def _network_hover_window(log_path):
    flight = read_network_flight(log_path, "")
    return _hover_measurement(flight.time, flight.north, flight.east, flight.height)


def _default_vehicle(frame):
    """The network vehicle with its default navigation settings (not the comparison mission's)."""
    return SimulatedVehicle(frame, default_weights(frame), HOME, MASS_KG, log=lambda *_: None)


def steadiness(attitude, hover, climb_out):
    hovering = attitude[(attitude[:, 0] >= hover[0]) & (attitude[:, 0] <= hover[1])]
    climbing = attitude[(attitude[:, 0] >= climb_out[0]) & (attitude[:, 0] <= climb_out[1])]
    climb_heading = unwrap_degrees(climbing[:, 3])
    return {"hover roll wobble, std (deg)": float(hovering[:, 1].std()),
            "hover pitch wobble, std (deg)": float(hovering[:, 2].std()),
            "hover heading wobble, std (deg)": float(unwrap_degrees(hovering[:, 3]).std()),
            "climb-out heading swing, peak to peak (deg)": float(climb_heading.max() - climb_heading.min()),
            "climb-out roll and pitch wobble, std (deg)": float(np.hypot(climbing[:, 1].std(), climbing[:, 2].std()))}


def gust_metrics(time, north, east, height, roll, pitch, gust_start):
    """The gust blows from the east, so the drone is pushed west."""
    gust_end = gust_start + GUST_DURATION_S
    before = (time > gust_start - BASELINE_BEFORE_GUST_S) & (time < gust_start)
    north0, east0, height0 = north[before].mean(), east[before].mean(), height[before].mean()
    displacement = np.hypot(north - north0, east - east0)
    tilt = tilt_from_roll_pitch(roll, pitch)
    during = (time >= gust_start) & (time <= gust_end)
    after = (time > gust_end) & (time <= gust_end + OBSERVE_AFTER_GUST_S)
    late = (time > gust_end - LATE_GUST_WINDOW_S) & (time <= gust_end)

    def settling_time(window, start):
        window_time, window_displacement = time[window], displacement[window]
        outside = np.nonzero(window_displacement > SETTLED_RADIUS_M)[0]
        if len(outside) == 0:
            return 0.0
        return float(window_time[min(outside[-1] + 1, len(window_time) - 1)] - start)
    return {"pushed off position by, peak (m)": float(displacement[during].max()),
            "time to get back within 0.3 m while the gust blows (s)": settling_time(during, gust_start),
            "offset still held at the end of the gust (m)": float(displacement[late].mean()),
            "tilt needed to hold against the gust (deg)": float(tilt[late].mean()),
            "steepest tilt during the gust (deg)": float(tilt[during].max()),
            "height change, peak (m)": float(np.abs(height[during | after] - height0).max()),
            "swing when the gust stops, peak (m)": float(displacement[after].max()),
            "time to settle within 0.3 m after the gust stops (s)": settling_time(after, gust_end),
            "_trace": (time - gust_start, displacement, tilt)}


def network_gust(frame):
    vehicle = _default_vehicle(frame)
    vehicle.set_parameter("SIM_WIND_SPD", 0.0)
    vehicle.set_parameter("SIM_WIND_DIR", GUST_FROM_EAST_DEG)
    vehicle.mode = CopterMode.GUIDED
    vehicle.arm()
    vehicle.takeoff(HOVER_HEIGHT_M)
    rows, stage, reached_at, gust_start = [], "climbing", None, None
    state = vehicle.env.state
    while vehicle.time < NETWORK_GUST_LIMIT_S:
        vehicle.step()
        roll, pitch, _ = (angle[0] for angle in euler_from_quaternion(state.attitude))
        position = state.position[0]
        rows.append([vehicle.time, position[0], -position[1], position[2], np.degrees(roll), np.degrees(-pitch)])
        if stage == "climbing" and position[2] > HOVER_HEIGHT_M - REACHED_HEIGHT_MARGIN_M:
            stage, reached_at = "settling", vehicle.time
        elif stage == "settling" and vehicle.time > reached_at + SETTLE_BEFORE_GUST_S:
            stage, gust_start = "gust", vehicle.time
            vehicle.set_parameter("SIM_WIND_SPD", GUST_SPEED_MPS)
        elif stage == "gust" and vehicle.time > gust_start + GUST_DURATION_S:
            stage = "after"
            vehicle.set_parameter("SIM_WIND_SPD", 0.0)
        elif stage == "after" and vehicle.time > gust_start + GUST_DURATION_S + OBSERVE_AFTER_GUST_S + 1:
            break
    if vehicle.crashed:
        raise RuntimeError(f"{frame} network crashed in the gust test")
    rows = np.array(rows)
    return gust_metrics(*(rows[:, column] for column in range(6)), gust_start)


def ardupilot_gust(ardupilot_root, output_dir):
    session = SitlSession(ardupilot_root, Path(output_dir) / "sitl_gust", HOME, SITL_SPEEDUP, log_stdout=False)
    boot_seconds = lambda message: message.time_boot_ms / MILLISECONDS
    is_position = lambda message: message.get_type() == "GLOBAL_POSITION_INT"
    try:
        session.wait_for_heartbeat()
        session.request_all_streams(SITL_STREAM_HZ)
        if not (session.set_parameter("SIM_WIND_SPD", 0) and session.set_parameter("SIM_WIND_DIR", GUST_FROM_EAST_DEG)
                and session.set_parameter("WP_SPD_UP", SITL_CLIMB_SPEED_MPS)):
            raise RuntimeError("SITL refused the gust-test parameters")
        session.wait_for_gps()
        if not session.command(mavlink.MAV_CMD_DO_SET_MODE, 1, CopterMode.GUIDED):
            raise RuntimeError("could not set GUIDED")
        if not session.arm_when_ready(SITL_ARMING_WAIT_S):
            raise RuntimeError("arming refused")
        if not session.command(mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, HOVER_HEIGHT_M):
            raise RuntimeError("SITL refused the take-off")
        climbed = session.wait(lambda message: is_position(message) and message.relative_alt
                               > (HOVER_HEIGHT_M - REACHED_HEIGHT_MARGIN_M) * MILLIMETRES, SITL_STEP_WAIT_S)
        if climbed is None:
            raise RuntimeError("SITL did not reach the hover height")
        session.wait(lambda message: is_position(message) and boot_seconds(message) > boot_seconds(climbed) + SETTLE_BEFORE_GUST_S,
                     SITL_STEP_WAIT_S)
        session.set_parameter("SIM_WIND_SPD", GUST_SPEED_MPS)
        gust_started = boot_seconds(session.wait(is_position, 5))
        session.wait(lambda message: is_position(message) and boot_seconds(message) > gust_started + GUST_DURATION_S,
                     SITL_STEP_WAIT_S)
        session.set_parameter("SIM_WIND_SPD", 0)
        session.wait(lambda message: is_position(message) and boot_seconds(message)
                     > gust_started + GUST_DURATION_S + OBSERVE_AFTER_GUST_S + EXTRA_OBSERVATION_S, SITL_STEP_WAIT_S)
        session.command(mavlink.MAV_CMD_NAV_LAND)
        session.wait_until_disarmed(SITL_LANDING_WAIT_S)
        time.sleep(LOG_SETTLE_S)
    finally:
        session.close()
    log_path = session.latest_log()
    shutil.copy(log_path, Path(output_dir) / "ardupilot_gust.BIN")
    logged = read_dataflash(log_path, ("SIM2", "ATT", "PARM"))
    truth = true_positions(logged["SIM2"], first_sample=0)
    attitude = np.array([[seconds(message), message.Roll, message.Pitch] for message in logged["ATT"]])
    wind_changes = np.array([[seconds(message), message.Value] for message in logged["PARM"] if message.Name == "SIM_WIND_SPD"])
    gust_start = wind_changes[wind_changes[:, 1] > 1][0, 0]
    return gust_metrics(truth[:, 0], truth[:, 1], truth[:, 2], truth[:, 3], np.interp(truth[:, 0], attitude[:, 0], attitude[:, 1]),
                        np.interp(truth[:, 0], attitude[:, 0], attitude[:, 2]), gust_start)


def delay_tolerance(frame, delay_steps):
    """Hold motor commands back by `delay_steps` control steps; hover, then jump the target 5 m sideways.
    Returns the worst tilt and final error, or None if it crashed."""
    vehicle = _default_vehicle(frame)
    vehicle.set_parameter("SIM_WIND_SPD", DELAY_TEST_WIND_MPS)
    env = vehicle.env
    pending = []
    undelayed_step = env.step

    def delayed_step(action, training=True):
        pending.append(np.array(action, float).copy())
        return undelayed_step(pending.pop(0) if len(pending) > delay_steps else pending[0], training=training)
    env.step = delayed_step
    vehicle.mode = CopterMode.GUIDED
    vehicle.arm()
    vehicle.takeoff(DELAY_TEST_HEIGHT_M)
    worst_tilt, sidestepped = 0.0, False
    while vehicle.time < DELAY_TEST_DURATION_S:
        vehicle.step()
        if vehicle.crashed:
            return None
        if vehicle.phase is Phase.FLYING:
            worst_tilt = max(worst_tilt, float(tilt_degrees(env.state.attitude)[0]))
            if not sidestepped and vehicle.time > DELAY_TEST_SIDESTEP_AT_S:
                sidestepped = True
                vehicle.goals = [Goal.go_to([0.0, -DELAY_TEST_SIDESTEP_M, DELAY_TEST_HEIGHT_M]), Goal.hold()]
    final_error = float(np.hypot(env.state.position[0, 0], env.state.position[0, 1] + DELAY_TEST_SIDESTEP_M))
    return dict(worst_tilt_deg=round(worst_tilt, 1), final_error_m=round(final_error, 2))


def run_stability(ardupilot_root, output_dir):
    from nn_drone.charts.comparison_charts import stability_chart
    output_dir = Path(output_dir).resolve()
    sitl_log = output_dir / "ardupilot_sitl.BIN"
    network_logs = {QUAD_NAME: output_dir / "network_quad_x.npz", HEXA_NAME: output_dir / "network_hex_x.npz"}
    attitudes = {ARDUPILOT_NAME: _ardupilot_attitude(sitl_log),
                 **{name: _network_attitude(path) for name, path in network_logs.items()}}
    hover_windows = {ARDUPILOT_NAME: _ardupilot_hover_window(sitl_log),
                     **{name: _network_hover_window(path) for name, path in network_logs.items()}}
    results = {"A_attitude_steadiness": {name: steadiness(attitude, hover_windows[name], CLIMB_OUT_WINDOW_S)
                                         for name, attitude in attitudes.items()}}
    gusts = {ARDUPILOT_NAME: ardupilot_gust(ardupilot_root, output_dir), QUAD_NAME: network_gust("quad_x"),
             HEXA_NAME: network_gust("hex_x")}
    results["B_gust_0_to_8_mps"] = {name: {key: round(value, RESULT_DECIMALS) for key, value in metrics.items()
                                           if not key.startswith("_")} for name, metrics in gusts.items()}
    step_ms = NOMINAL_CONTROL_PERIOD_MS * float(time_scale(MASS_KG))
    results["C_network_delay_tolerance"] = {
        frame: {f"{steps * step_ms:.0f} ms": delay_tolerance(frame, steps) or "crashed" for steps in DELAY_STEPS}
        for frame in ("quad_x", "hex_x")}
    (output_dir / "stability.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    stability_chart(output_dir / "stability.png", gusts, attitudes, GUST_DURATION_S, OBSERVE_AFTER_GUST_S)
    return results
