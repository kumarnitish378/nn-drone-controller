"""The long route in the air (no ground): from a hover at the take-off point to a touchdown at z = 0."""
from dataclasses import dataclass

import numpy as np

from nn_drone.flight_tests.long_route import FIRST_HOVER, LANDING_DESCENT, build_long_route, summarize_route_tracking
from nn_drone.flight_tests.recording import FlightRecorder, lost_control

RUNAWAY_SPEED_MPS = 40.0
FELL_BELOW_START_M = -3.0
RECORD_EVERY_STEPS = 5
TAKE_OFF_SEGMENT = 0
NOMINAL_SEED, ROBUST_SEED = 1, 2
ROBUST_DRONES = 100
POSITION_ERROR_CLIP = 2.5
SECONDS_PER_MINUTE = 60


@dataclass
class LongMissionFlight:
    period: float
    route: object
    distance_to_target: np.ndarray
    positions: np.ndarray
    crashed: np.ndarray
    landed: np.ndarray
    touchdown_speed: np.ndarray
    touchdown_offset: np.ndarray
    touchdown_time: np.ndarray
    lowest_point_during_takeoff: np.ndarray
    pwm_every_step: np.ndarray
    recording: dict


def fly_long_mission(pilot, mass_kg, speeds, count, randomize, seed, record=False):
    env = pilot.make_environment(count, randomize=randomize, seed=seed, position_error_clip_m=POSITION_ERROR_CLIP,
                                 nominal_mass_kg=mass_kg, start_altitude_m=0.0)
    period = env.control_period
    route = build_long_route(period, speeds)
    landing_segment = route.segment(LANDING_DESCENT)
    steps = len(route.targets)
    distance_to_target = np.zeros((steps, count), np.float32)
    positions = np.zeros((steps, count, 3), np.float32)
    crashed, landed = np.zeros(count, bool), np.zeros(count, bool)
    touchdown_speed, touchdown_offset, touchdown_time = (np.full(count, np.nan) for _ in range(3))
    lowest_during_takeoff = np.zeros(count)
    pwm_every_step = np.zeros((steps, env.motor_count), np.float32)
    recorder = FlightRecorder(every_steps=RECORD_EVERY_STEPS)
    state = env.state
    for step in range(steps):
        segment = route.segment_of_step[step]
        env.set_target(route.targets[step], 0.0)
        result = pilot.fly_step(env)
        fell = (state.position[:, 2] < FELL_BELOW_START_M) & (segment != landing_segment)
        crashed |= ~landed & (lost_control(env, RUNAWAY_SPEED_MPS) | fell)
        if segment == TAKE_OFF_SEGMENT:
            lowest_during_takeoff = np.minimum(lowest_during_takeoff, state.position[:, 2])
        if segment == landing_segment:
            touching = ~landed & ~crashed & (state.position[:, 2] <= 0)
            touchdown_speed[touching] = -state.velocity[touching, 2]
            touchdown_offset[touching] = np.hypot(state.position[touching, 0], state.position[touching, 1])
            touchdown_time[touching] = (step + 1) * period
            landed |= touching
        distance_to_target[step] = result.distance_to_target
        positions[step] = state.position
        if record:
            pwm_every_step[step] = result.pwm[0]
            recorder.capture(step, (step + 1) * period, env, result.pwm, target=route.targets[step])
    return LongMissionFlight(period, route, distance_to_target, positions, crashed, landed, touchdown_speed, touchdown_offset,
                             touchdown_time, lowest_during_takeoff, pwm_every_step, recorder.arrays())


def run_long_mission_test(pilot, mass_kg, speeds, robust_drones=ROBUST_DRONES):
    nominal = fly_long_mission(pilot, mass_kg, speeds, 1, randomize=False, seed=NOMINAL_SEED, record=True)
    robust = fly_long_mission(pilot, mass_kg, speeds, robust_drones, randomize=True, seed=ROBUST_SEED)
    pwm, recording = nominal.pwm_every_step, nominal.recording
    route = nominal.route
    hover_steps = route.segment_of_step == route.segment(FIRST_HOVER)
    summary = dict(
        mass_kg=mass_kg, speeds_mps=speeds.as_report(),
        mission_time_min=float(np.nanmax(nominal.touchdown_time) / SECONDS_PER_MINUTE) if nominal.landed[0] else None,
        standard=dict(
            crashed=bool(nominal.crashed[0]), landed=bool(nominal.landed[0]),
            touchdown_speed_mps=float(nominal.touchdown_speed[0]), touchdown_offset_m=float(nominal.touchdown_offset[0]),
            lowest_point_during_takeoff_m=float(nominal.lowest_point_during_takeoff[0]),
            max_tilt_deg=float(np.rad2deg(np.abs(recording["roll_pitch_yaw"][:, :2]).max())),
            max_climb_mps=float(recording["vertical_speed"].max()), max_descent_mps=float(-recording["vertical_speed"].min()),
            pwm_min=float(pwm.min()), pwm_max=float(pwm.max()), pwm_hover_mean=float(pwm[hover_steps].mean()),
            pwm_step_change_mean_us=float(np.abs(np.diff(pwm, axis=0)).mean()),
            **summarize_route_tracking(route, nominal.period, nominal.distance_to_target, nominal.positions, drone=0)),
        randomised_100=dict(
            crashed=int(robust.crashed.sum()), landed=int(robust.landed.sum()),
            touchdown_speed_median_max_mps=[float(np.nanmedian(robust.touchdown_speed)), float(np.nanmax(robust.touchdown_speed))],
            touchdown_offset_median_max_m=[float(np.nanmedian(robust.touchdown_offset)), float(np.nanmax(robust.touchdown_offset))],
            lowest_point_during_takeoff_worst_m=float(robust.lowest_point_during_takeoff.min()),
            **summarize_route_tracking(robust.route, robust.period, robust.distance_to_target, robust.positions)))
    return nominal, summary
