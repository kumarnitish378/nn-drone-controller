"""The long route from disarmed to disarmed on solid ground: fixed PWM while disarmed, idle and spooling up,
then the network flies; after touchdown a landing detector ramps the motors to idle and disarms."""
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from nn_drone.flight_tests.landing import (PWM_ARMED_IDLE_US, PWM_SPOOL_END_US, LandingDetector, MotorSequenceTiming)
from nn_drone.flight_tests.long_route import (FIRST_HOVER, LANDING_DESCENT, RouteTrajectory, build_long_route,
                                              summarize_route_tracking)
from nn_drone.flight_tests.recording import FlightRecorder, lost_control
from nn_drone.units import PWM_DISARMED_US, pwm_to_action

RUNAWAY_SPEED_MPS = 40.0
HARD_TOUCHDOWN_MPS = 2.0
LIFT_OFF_HEIGHT_M = 0.05
TAIL_AFTER_ROUTE_S = 12.0
RECORD_EVERY_STEPS = 5
PRE_FLIGHT_SEGMENT = -1
NOMINAL_SEED, ROBUST_SEED = 1, 2
DEFAULT_ROBUST_DRONES = 100
POSITION_ERROR_CLIP = 2.5
SECONDS_PER_MINUTE = 60
TAKE_OFF_WINDOW_S = 6.0
FINAL_DESCENT_WINDOW_S = 10.0
DISARMED_SETTLE_S = 0.1
IDLE_SETTLE_S = 0.2


class LandingPhase(IntEnum):
    FLYING = 0
    RAMP_TO_IDLE = 1
    IDLE = 2
    DISARMED = 3


@dataclass
class FullMissionFlight:
    period: float
    route: object
    pre_flight_steps: int
    segment_of_step: np.ndarray
    distance_to_target: np.ndarray
    positions: np.ndarray
    crashed: np.ndarray
    landed: np.ndarray
    disarmed: np.ndarray
    touchdown_speed: np.ndarray
    touchdown_offset: np.ndarray
    lift_off_time: np.ndarray
    touchdown_time: np.ndarray
    detection_time: np.ndarray
    disarm_time: np.ndarray
    moved_before_takeoff: np.ndarray
    pwm_every_step: np.ndarray
    recording: dict

    @property
    def whole_flight_route(self):
        """Segments for every step including pre-flight and the tail, so indices line up with the flight arrays."""
        return RouteTrajectory(None, self.segment_of_step, self.route.segment_names)


def pre_flight_pwm(period, timing):
    return np.concatenate([np.full(int(timing.disarmed_s / period), PWM_DISARMED_US),
                           np.full(int(timing.armed_idle_s / period), PWM_ARMED_IDLE_US),
                           np.linspace(PWM_ARMED_IDLE_US, PWM_SPOOL_END_US, int(timing.spool_up_s / period))])


def fly_full_mission(pilot, mass_kg, speeds, count, randomize, seed, record=False, timing=MotorSequenceTiming()):
    env = pilot.make_environment(count, randomize=randomize, seed=seed, position_error_clip_m=POSITION_ERROR_CLIP,
                                 nominal_mass_kg=mass_kg, start_altitude_m=0.0, solid_ground=True)
    env.park_on_ground()
    period = env.control_period
    route = build_long_route(period, speeds)
    landing_segment = route.segment(LANDING_DESCENT)
    pre_flight = pre_flight_pwm(period, timing)
    pre_flight_steps, tail_steps = len(pre_flight), int(TAIL_AFTER_ROUTE_S / period)
    targets = np.concatenate([np.zeros((pre_flight_steps, 3)), route.targets, np.repeat(route.targets[-1:], tail_steps, 0)])
    segment_of_step = np.concatenate([np.full(pre_flight_steps, PRE_FLIGHT_SEGMENT), route.segment_of_step,
                                      np.full(tail_steps, landing_segment)])
    steps = len(targets)
    distance_to_target = np.zeros((steps, count), np.float32)
    positions = np.zeros((steps, count, 3), np.float32)
    crashed = np.zeros(count, bool)
    phase = np.full(count, LandingPhase.FLYING, dtype=int)
    phase_started = np.zeros(count)
    pwm_at_detection = np.zeros((count, env.motor_count))
    detector = LandingDetector(count, period, timing=timing)
    lift_off_time, touchdown_time, detection_time, disarm_time, touchdown_offset = (np.full(count, np.nan) for _ in range(5))
    moved_before_takeoff = np.zeros(count)
    pwm_every_step = np.zeros((steps, env.motor_count), np.float32)
    recorder = FlightRecorder(every_steps=RECORD_EVERY_STEPS)
    state = env.state
    for step in range(steps):
        time = (step + 1) * period
        env.set_target(targets[step], 0.0)
        action = np.clip(pilot.command(env.observe()), -1, 1).astype(np.float64)
        if step < pre_flight_steps:
            action[:] = pwm_to_action(pre_flight[step])
        else:
            ramping = phase == LandingPhase.RAMP_TO_IDLE
            if ramping.any():
                progress = np.clip((time - phase_started[ramping]) / timing.ramp_to_idle_s, 0, 1)[:, None]
                action[ramping] = pwm_to_action((1 - progress) * pwm_at_detection[ramping] + progress * PWM_ARMED_IDLE_US)
            action[phase == LandingPhase.IDLE] = pwm_to_action(PWM_ARMED_IDLE_US)
            action[phase == LandingPhase.DISARMED] = pwm_to_action(PWM_DISARMED_US)
        result = env.step(action, training=False)
        if step < pre_flight_steps:
            moved_before_takeoff = np.maximum(moved_before_takeoff, np.linalg.norm(state.position, axis=1))
        crashed |= lost_control(env, RUNAWAY_SPEED_MPS) | (state.touchdown_speed > HARD_TOUCHDOWN_MPS)
        lifting_off = np.isnan(lift_off_time) & (state.position[:, 2] > LIFT_OFF_HEIGHT_M)
        lift_off_time[lifting_off] = time
        if segment_of_step[step] == landing_segment:
            touching = np.isnan(touchdown_time) & state.on_ground
            touchdown_time[touching] = time
            touchdown_offset[touching] = np.hypot(state.position[touching, 0], state.position[touching, 1])
            flying = phase == LandingPhase.FLYING
            candidate = flying & detector.looks_landed(state.velocity[:, 2], state.position[:, 2], targets[step, 2],
                                                       result.pwm.mean(1))
            detected = flying & detector.update(candidate)
            phase[detected] = LandingPhase.RAMP_TO_IDLE
            phase_started[detected] = time
            detection_time[detected] = time
            pwm_at_detection[detected] = result.pwm[detected]
            to_idle = (phase == LandingPhase.RAMP_TO_IDLE) & (time - phase_started >= timing.ramp_to_idle_s)
            phase[to_idle] = LandingPhase.IDLE
            phase_started[to_idle] = time
            switching_off = (phase == LandingPhase.IDLE) & (time - phase_started >= timing.idle_before_disarm_s)
            phase[switching_off] = LandingPhase.DISARMED
            disarm_time[switching_off] = time
        distance_to_target[step] = result.distance_to_target
        positions[step] = state.position
        if record:
            pwm_every_step[step] = result.pwm[0]
            recorder.capture(step, time, env, result.pwm, target=targets[step])
    return FullMissionFlight(period, route, pre_flight_steps, segment_of_step, distance_to_target, positions, crashed,
                             landed=~np.isnan(touchdown_time), disarmed=phase == LandingPhase.DISARMED,
                             touchdown_speed=state.touchdown_speed.copy(), touchdown_offset=touchdown_offset,
                             lift_off_time=lift_off_time, touchdown_time=touchdown_time, detection_time=detection_time,
                             disarm_time=disarm_time, moved_before_takeoff=moved_before_takeoff,
                             pwm_every_step=pwm_every_step, recording=recorder.arrays())


class PwmTimeline:
    """The nominal drone's PWM indexed by time, for the motor-sequence report."""

    def __init__(self, pwm_every_step, period):
        self.pwm = pwm_every_step
        self.period = period
        self.time = np.arange(1, len(pwm_every_step) + 1) * period

    def between(self, start, end):
        return self.pwm[(self.time > start) & (self.time <= end)]

    def at(self, time):
        return self.pwm[int(time / self.period) - 1]


def motor_sequence_report(flight, timing):
    timeline = PwmTimeline(flight.pwm_every_step, flight.period)
    armed_at = timing.disarmed_s
    spool_from = armed_at + timing.armed_idle_s
    network_from = spool_from + timing.spool_up_s
    lift_off, touchdown = flight.lift_off_time[0], flight.touchdown_time[0]
    detected, disarmed = flight.detection_time[0], flight.disarm_time[0]
    spool = timeline.between(spool_from, network_from)
    hover_steps = flight.segment_of_step == flight.route.segment(FIRST_HOVER)
    return {
        "1 disarmed (0-1 s)": float(timeline.between(0, armed_at).mean()),
        "2 armed idle (1-3 s)": float(timeline.between(armed_at, spool_from).mean()),
        "3 spool-up ramp (3-5 s) start/end": [float(spool[0].mean()), float(spool[-1].mean())],
        "4 network takes over: first output per motor": [float(value) for value in timeline.between(network_from, network_from + 1)[0]],
        "5 lift-off time (s)": float(lift_off),
        "5 PWM at lift-off (mean)": float(timeline.at(lift_off).mean()),
        "6 take-off peak PWM": float(timeline.between(network_from, network_from + TAKE_OFF_WINDOW_S).max()),
        "7 hover at 50 m (mean)": float(flight.pwm_every_step[hover_steps].mean()),
        "8 final descent before touchdown (mean)": float(timeline.between(touchdown - FINAL_DESCENT_WINDOW_S, touchdown).mean()),
        "9 at touchdown per motor": [float(value) for value in timeline.at(touchdown)],
        "10 lowest PWM while pressing down before detection": float(timeline.between(touchdown, detected).min()),
        "11 at land detection per motor": [float(value) for value in timeline.at(detected)],
        "12 idle after ramp-down": float(timeline.between(detected + timing.ramp_to_idle_s + IDLE_SETTLE_S,
                                                          disarmed - DISARMED_SETTLE_S).mean()),
        "13 disarmed": float(timeline.between(disarmed + DISARMED_SETTLE_S, timeline.time[-1]).mean())}


def run_full_mission_test(pilot, mass_kg, speeds, robust_drones=DEFAULT_ROBUST_DRONES, timing=MotorSequenceTiming()):
    nominal = fly_full_mission(pilot, mass_kg, speeds, 1, randomize=False, seed=NOMINAL_SEED, record=True, timing=timing)
    robust = fly_full_mission(pilot, mass_kg, speeds, robust_drones, randomize=True, seed=ROBUST_SEED, timing=timing)
    tracking = lambda flight, drone=None: summarize_route_tracking(
        flight.whole_flight_route, flight.period, flight.distance_to_target, flight.positions, drone)
    summary = dict(
        mass_kg=mass_kg, speeds_mps=speeds.as_report(),
        total_time_min=float(nominal.disarm_time[0] / SECONDS_PER_MINUTE),
        pwm_sequence_standard_drone=motor_sequence_report(nominal, timing),
        standard=dict(crashed=bool(nominal.crashed[0]), touchdown_speed_mps=float(nominal.touchdown_speed[0]),
                      touchdown_offset_m=float(nominal.touchdown_offset[0]),
                      moved_before_takeoff_m=float(nominal.moved_before_takeoff[0]),
                      touchdown_to_detect_s=float(nominal.detection_time[0] - nominal.touchdown_time[0]),
                      touchdown_to_disarm_s=float(nominal.disarm_time[0] - nominal.touchdown_time[0]),
                      **tracking(nominal, 0)),
        randomised_100=dict(
            crashed=int(robust.crashed.sum()), landed=int(robust.landed.sum()), disarmed=int(robust.disarmed.sum()),
            lifted_off=int((~np.isnan(robust.lift_off_time)).sum()),
            liftoff_time_min_max_s=[float(np.nanmin(robust.lift_off_time)), float(np.nanmax(robust.lift_off_time))],
            moved_before_takeoff_max_m=float(robust.moved_before_takeoff.max()),
            touchdown_speed_median_max_mps=[float(np.median(robust.touchdown_speed)), float(robust.touchdown_speed.max())],
            touchdown_offset_median_max_m=[float(np.nanmedian(robust.touchdown_offset)), float(np.nanmax(robust.touchdown_offset))],
            touchdown_to_disarm_median_max_s=[float(np.nanmedian(robust.disarm_time - robust.touchdown_time)),
                                              float(np.nanmax(robust.disarm_time - robust.touchdown_time))],
            **tracking(robust)))
    return nominal, summary
