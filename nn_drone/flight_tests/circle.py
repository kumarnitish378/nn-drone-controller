"""Circle test: climb to the start of the circle, fly one lap behind a moving target, then hold."""
from dataclasses import dataclass

import numpy as np

from nn_drone.flight_tests.recording import FlightRecorder

STANDARD_RADIUS_M = 5.0
STANDARD_SPEED_MPS = 1.5
STANDARD_ALTITUDE_M = 3.0
CLIMB_WHEN_ALTITUDE_GIVEN_M = 5.0
STANDARD_APPROACH_S = 6.0
BASE_HOLD_S = 5.0
HOLD_MEASUREMENT_S = 2.0
NOMINAL_SEED = 1
ROBUST_SEED = 2
ROBUST_DRONES_AT_1KG = 500
ROBUST_DRONES_HEAVY = 200
HIGH_PERCENTILE = 95


@dataclass(frozen=True)
class CirclePlan:
    radius_m: float = STANDARD_RADIUS_M
    speed_mps: float = STANDARD_SPEED_MPS
    mass_kg: float = 1.0
    altitude_asl_m: float = STANDARD_ALTITUDE_M
    approach_s: float = STANDARD_APPROACH_S
    climb_m: float = STANDARD_ALTITUDE_M
    position_error_clip: float = 2.5

    @classmethod
    def at_altitude(cls, radius_m, speed_mps, mass_kg, altitude_asl_m, approach_s=STANDARD_APPROACH_S, **options):
        return cls(radius_m, speed_mps, mass_kg, altitude_asl_m, approach_s, climb_m=CLIMB_WHEN_ALTITUDE_GIVEN_M, **options)

    @property
    def start_altitude_asl_m(self):
        return self.altitude_asl_m - self.climb_m

    @property
    def lap_s(self):
        return 2 * np.pi * self.radius_m / self.speed_mps

    @property
    def hold_s(self):
        return BASE_HOLD_S * max(1.0, self.mass_kg ** (1 / 6))

    @property
    def robust_drones(self):
        return ROBUST_DRONES_AT_1KG if self.mass_kg == 1.0 else ROBUST_DRONES_HEAVY


@dataclass
class CircleFlight:
    crashed: np.ndarray
    radial_error_mean: np.ndarray
    radial_error_max: np.ndarray
    altitude_error_max: np.ndarray
    lag_mean: np.ndarray
    hold_error: np.ndarray
    radial_error_trace: np.ndarray
    recording: dict


def fly_circle(pilot, plan, count, randomize, seed, record=False):
    env = pilot.make_environment(count, randomize=randomize, seed=seed, position_error_clip_m=plan.position_error_clip,
                                 nominal_mass_kg=plan.mass_kg, start_altitude_m=plan.start_altitude_asl_m)
    period = env.control_period
    steps = int((plan.approach_s + plan.lap_s + plan.hold_s) / period)
    radial_error, altitude_error, lag = (np.zeros((steps, count)) for _ in range(3))
    crashed = np.zeros(count, bool)
    recorder = FlightRecorder()
    for step in range(steps):
        time = (step + 1) * period
        angle = np.clip((time - plan.approach_s) / plan.lap_s, 0, 1) * 2 * np.pi
        env.set_target([plan.radius_m * np.cos(angle), plan.radius_m * np.sin(angle), plan.climb_m], angle + np.pi / 2)
        result = pilot.fly_step(env)
        crashed |= result.crashed
        position = env.state.position
        radial_error[step] = np.hypot(position[:, 0], position[:, 1]) - plan.radius_m
        altitude_error[step] = position[:, 2] - plan.climb_m
        lag[step] = result.distance_to_target
        if record:
            recorder.capture(step, time, env, result.pwm)
    lap = slice(int(plan.approach_s / period), int((plan.approach_s + plan.lap_s) / period))
    hold = slice(steps - int(HOLD_MEASUREMENT_S / period), steps)
    return CircleFlight(crashed=crashed, radial_error_mean=np.abs(radial_error[lap]).mean(0),
                        radial_error_max=np.abs(radial_error[lap]).max(0), altitude_error_max=np.abs(altitude_error[lap]).max(0),
                        lag_mean=lag[lap].mean(0), hold_error=lag[hold].mean(0), radial_error_trace=radial_error[:, 0],
                        recording=recorder.arrays())


def run_circle_test(pilot, plan):
    nominal = fly_circle(pilot, plan, 1, randomize=False, seed=NOMINAL_SEED, record=True)
    robust = fly_circle(pilot, plan, plan.robust_drones, randomize=True, seed=ROBUST_SEED)
    summary = dict(
        radius_m=plan.radius_m, altitude_asl_m=plan.altitude_asl_m, mass_kg=plan.mass_kg, speed_mps=plan.speed_mps,
        lap_time_s=round(plan.lap_s, 1),
        nominal_radial_error_mean_m=float(nominal.radial_error_mean[0]),
        nominal_radial_error_max_m=float(nominal.radial_error_max[0]),
        nominal_altitude_error_max_m=float(nominal.altitude_error_max[0]),
        nominal_lag_behind_target_m=float(nominal.lag_mean[0]),
        nominal_final_hold_error_m=float(nominal.hold_error[0]),
        robust_crashed=int(robust.crashed.sum()),
        robust_radial_error_mean_median_m=float(np.median(robust.radial_error_mean)),
        robust_radial_error_max_p95_m=float(np.percentile(robust.radial_error_max, HIGH_PERCENTILE)),
        robust_radial_error_max_worst_m=float(robust.radial_error_max.max()),
        robust_altitude_error_max_p95_m=float(np.percentile(robust.altitude_error_max, HIGH_PERCENTILE)),
        max_tilt_deg=float(np.rad2deg(np.abs(nominal.recording["roll_pitch_yaw"][:, :2]).max())))
    return nominal, summary
