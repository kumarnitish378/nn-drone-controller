"""Waypoint missions: fly to each point in turn, then hold the last one.

The square mission (3-5 m) and the high mission (20 m and 30 m legs) are run on one nominal drone and
on hundreds of randomised ones; the envelope test and the CI gate reuse the same flight.
"""
from dataclasses import dataclass

import numpy as np

from nn_drone.flight_tests.recording import FlightRecorder, crashed_in_flight, current_tilt_degrees

SQUARE_MISSION = ((0, 0, 3, 0), (6, 0, 3, 0), (6, 6, 3, 90), (0, 6, 5, 180), (0, 0, 5, -90), (0, 0, 2, 0))
HIGH_MISSION = ((0, 0, 20, 0), (8, 0, 20, 0), (8, 8, 20, 90), (0, 8, 30, 180), (0, 0, 30, -90), (0, 0, 5, 0))
"""(x forward, y left, z up) metres from the start, heading degrees."""

ARRIVAL_RADIUS_M = 0.4
FINAL_HOLD_S = 8.0
HOLD_MEASUREMENT_WINDOW_S = 2.0
MIN_HOLD_SAMPLES = 50
DEFAULT_DURATION_S = 70.0
RECORDING_REFERENCE_DURATION_S = 70.0
DEFAULT_POSITION_ERROR_CLIP = 2.5


@dataclass
class WaypointMissionResult:
    completed: np.ndarray
    crashed: np.ndarray
    finish_time: np.ndarray
    hold_error: np.ndarray
    max_tilt_deg: np.ndarray
    recording: dict


def fly_waypoint_mission(pilot, count, randomize, seed, mission=SQUARE_MISSION, duration_s=DEFAULT_DURATION_S,
                         final_hold_s=FINAL_HOLD_S, record=False, noise=True,
                         position_error_clip=DEFAULT_POSITION_ERROR_CLIP, **config_overrides):
    config_overrides.setdefault("position_error_clip_m", position_error_clip)
    env = pilot.make_environment(count, randomize=randomize, seed=seed, noise=noise, **config_overrides)
    waypoints = np.array(mission, dtype=float)
    last_waypoint = len(waypoints) - 1
    waypoint_index = np.zeros(count, dtype=int)
    finish_time = np.full(count, np.nan)
    crashed = np.zeros(count, bool)
    hold_error_sum, hold_samples = np.zeros(count), np.zeros(count)
    max_tilt = np.zeros(count)
    recorder = FlightRecorder(every_steps=max(1, int(duration_s / RECORDING_REFERENCE_DURATION_S)))
    period = env.control_period
    for step in range(int(duration_s / period)):
        current = waypoints[waypoint_index]
        env.set_target(current[:, :3], np.deg2rad(current[:, 3]))
        result = pilot.fly_step(env)
        crashed |= crashed_in_flight(env)
        time = (step + 1) * period
        max_tilt = np.maximum(max_tilt, current_tilt_degrees(env))
        arrived = result.distance_to_target < ARRIVAL_RADIUS_M
        on_final_leg = waypoint_index == last_waypoint
        just_finished = on_final_leg & arrived & np.isnan(finish_time)
        waypoint_index[arrived & (waypoint_index < last_waypoint)] += 1
        finish_time[just_finished] = time
        time_since_finish = time - finish_time
        measuring = (on_final_leg & ~np.isnan(finish_time) & (time_since_finish > final_hold_s - HOLD_MEASUREMENT_WINDOW_S)
                     & (time_since_finish <= final_hold_s))
        hold_error_sum[measuring] += result.distance_to_target[measuring]
        hold_samples[measuring] += 1
        if record:
            recorder.capture(step, time, env, result.pwm)
        if np.all(~np.isnan(finish_time) & (time_since_finish > final_hold_s) | crashed):
            break
    hold_error = np.where(hold_samples > MIN_HOLD_SAMPLES, hold_error_sum / np.maximum(hold_samples, 1), np.nan)
    return WaypointMissionResult(completed=~np.isnan(finish_time) & ~crashed, crashed=crashed, finish_time=finish_time,
                                 hold_error=hold_error, max_tilt_deg=max_tilt, recording=recorder.arrays())
