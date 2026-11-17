"""Flight envelope: climb from sea level to 2000 m (thinner air), and the square mission from 1 kg to 100 kg."""
import numpy as np

from nn_drone.flight_tests.waypoint_mission import fly_waypoint_mission

CLIMB_TOP_M = 2000.0
CLIMB_SQUARE_SIDE_M = 20.0
CLIMB_MISSION = ((0, 0, CLIMB_TOP_M, 0), (CLIMB_SQUARE_SIDE_M, 0, CLIMB_TOP_M, 0),
                 (CLIMB_SQUARE_SIDE_M, CLIMB_SQUARE_SIDE_M, CLIMB_TOP_M, 90), (0, CLIMB_SQUARE_SIDE_M, CLIMB_TOP_M, 180),
                 (0, 0, CLIMB_TOP_M, -90))
CLIMB_DURATION_S = 600.0
CLIMB_ROBUST_DRONES = 200
REACHED_TOP_MARGIN_M = 1.0
LOW_CLIMB_BAND_M = (50.0, 150.0)
HIGH_CLIMB_BAND_M = (1850.0, 1950.0)
CLIMBING_SPEED_MPS = 1.0
HOVER_TAIL_SAMPLES = 20

SWEEP_MASSES_KG = (1, 2, 5, 10, 25, 50, 100)
SWEEP_DRONES = 200
SWEEP_DURATION_S = 90.0
NOMINAL_SEED, ROBUST_SEED, SWEEP_SEED = 1, 2, 3
HIGH_PERCENTILE = 95


def _mean_or_none(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def climb_test(pilot, position_error_clip):
    nominal = fly_waypoint_mission(pilot, 1, randomize=False, seed=NOMINAL_SEED, mission=CLIMB_MISSION,
                                   duration_s=CLIMB_DURATION_S, record=True, position_error_clip=position_error_clip)
    robust = fly_waypoint_mission(pilot, CLIMB_ROBUST_DRONES, randomize=True, seed=ROBUST_SEED, mission=CLIMB_MISSION,
                                  duration_s=CLIMB_DURATION_S, position_error_clip=position_error_clip)
    recording = nominal.recording
    time, height, mean_pwm = recording["time"], recording["position"][:, 2], recording["pwm"].mean(1)
    vertical_speed = np.gradient(height, time)
    climbing = vertical_speed > CLIMBING_SPEED_MPS
    low = (height > LOW_CLIMB_BAND_M[0]) & (height < LOW_CLIMB_BAND_M[1]) & climbing
    high = (height > HIGH_CLIMB_BAND_M[0]) & (height < HIGH_CLIMB_BAND_M[1]) & climbing
    at_top = height > CLIMB_TOP_M - REACHED_TOP_MARGIN_M
    summary = dict(
        nominal_completed=bool(nominal.completed[0]),
        nominal_time_to_2000m_s=float(time[np.argmax(at_top)]) if at_top.any() else None,
        nominal_mission_time_s=float(nominal.finish_time[0]), nominal_hold_error_at_2000m_m=float(nominal.hold_error[0]),
        climb_rate_near_ground_mps=float(vertical_speed[low].mean()),
        climb_rate_near_2000m_mps=_mean_or_none(vertical_speed, high),
        mean_pwm_climbing_near_ground=float(mean_pwm[low].mean()),
        mean_pwm_climbing_near_2000m=_mean_or_none(mean_pwm, high),
        mean_pwm_hover_at_2000m=float(mean_pwm[-HOVER_TAIL_SAMPLES:].mean()), max_altitude_m=float(height.max()),
        robust_n=CLIMB_ROBUST_DRONES, robust_completed=int(robust.completed.sum()), robust_crashed=int(robust.crashed.sum()),
        robust_hold_error_median_m=float(np.nanmedian(robust.hold_error)) if robust.completed.any() else None)
    return nominal, summary


def mass_sweep(pilot, position_error_clip, log=print):
    rows = []
    for mass in SWEEP_MASSES_KG:
        flight = fly_waypoint_mission(pilot, SWEEP_DRONES, randomize=True, seed=SWEEP_SEED, duration_s=SWEEP_DURATION_S,
                                      position_error_clip=position_error_clip, nominal_mass_kg=float(mass))
        rows.append(dict(mass_kg=mass, completed=int(flight.completed.sum()), crashed=int(flight.crashed.sum()), n=SWEEP_DRONES,
                         mission_time_median_s=float(np.nanmedian(flight.finish_time)) if flight.completed.any() else None,
                         hold_error_median_m=(float(np.nanmedian(flight.hold_error)) if np.isfinite(flight.hold_error).any()
                                              else None),
                         max_tilt_p95_deg=float(np.percentile(flight.max_tilt_deg, HIGH_PERCENTILE))))
        log(rows[-1])
    return rows
