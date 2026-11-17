"""Square / high mission test: one nominal drone and 500 randomised ones, summarised for results.json."""
import json
from pathlib import Path

import numpy as np

from nn_drone.flight_tests.waypoint_mission import HIGH_MISSION, SQUARE_MISSION, fly_waypoint_mission

ROBUST_DRONES = 500
NOMINAL_SEED = 1
ROBUST_SEED = 2
HOVER_TAIL_SAMPLES = 200
LEVEL_FLIGHT_ERROR_LIMIT_M = 1.0
LEVEL_FLIGHT_MAX_VERTICAL_SPEED = 0.5
HIGH_PERCENTILE = 95


def run_waypoint_test(pilot, high=False, position_error_clip=2.5):
    mission = HIGH_MISSION if high else SQUARE_MISSION
    nominal = fly_waypoint_mission(pilot, 1, randomize=False, seed=NOMINAL_SEED, mission=mission, record=True,
                                   position_error_clip=position_error_clip)
    robust = fly_waypoint_mission(pilot, ROBUST_DRONES, randomize=True, seed=ROBUST_SEED, mission=mission,
                                  position_error_clip=position_error_clip)
    return mission, nominal, robust


def summarize(nominal, robust):
    recording = nominal.recording
    pwm = recording["pwm"]
    hover_pwm = pwm[-HOVER_TAIL_SAMPLES:]
    summary = dict(
        nominal_completed=bool(nominal.completed[0]),
        nominal_mission_time_s=float(nominal.finish_time[0]),
        nominal_final_hold_error_m=float(nominal.hold_error[0]),
        robust_n=len(robust.completed),
        robust_completed=int(robust.completed.sum()),
        robust_crashed=int(robust.crashed.sum()),
        robust_timeout=int((~robust.completed & ~robust.crashed).sum()),
        robust_mission_time_median_s=float(np.nanmedian(robust.finish_time)),
        robust_hold_error_median_m=float(np.nanmedian(robust.hold_error)),
        robust_hold_error_p95_m=float(np.nanpercentile(robust.hold_error, HIGH_PERCENTILE)),
        robust_hold_error_max_m=float(np.nanmax(robust.hold_error)),
        robust_max_tilt_p95_deg=float(np.percentile(robust.max_tilt_deg, HIGH_PERCENTILE)),
        pwm_min=float(pwm.min()), pwm_max=float(pwm.max()),
        pwm_hover_mean=float(hover_pwm.mean()),
        pwm_jitter_hover_us=float(np.abs(np.diff(hover_pwm, axis=0)).mean()),
        max_tilt_deg=float(np.rad2deg(np.abs(recording["roll_pitch_yaw"][:, :2]).max())),
    )
    summary.update(altitude_tracking(recording))
    return summary


def altitude_tracking(recording):
    height, target_height, time = recording["position"][:, 2], recording["target"][:, 2], recording["time"]
    vertical_speed = np.gradient(height, time)
    level = ((np.abs(np.gradient(target_height)) == 0) & (np.abs(target_height - height) < LEVEL_FLIGHT_ERROR_LIMIT_M)
             & (np.abs(vertical_speed) < LEVEL_FLIGHT_MAX_VERTICAL_SPEED))
    return dict(max_climb_rate_mps=float(vertical_speed.max()), max_descent_rate_mps=float(-vertical_speed.min()),
                max_altitude_overshoot_m=float(max(0.0, (height - np.maximum.accumulate(target_height)).max())),
                level_flight_altitude_error_mean_m=float(np.abs(target_height[level] - height[level]).mean()))


def write_outputs(output_dir, mission, nominal, summary):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=1))
    np.savez(output_dir / "flight_log.npz", **nominal.recording, mission=np.array(mission))
