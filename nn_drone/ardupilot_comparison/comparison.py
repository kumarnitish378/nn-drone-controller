"""Measure ArduPilot's and the networks' flights of the comparison mission with the same code."""
import json
import math
from pathlib import Path

import numpy as np

from nn_drone.ardupilot_comparison.flight_logs import read_ardupilot_flight, read_network_flight
from nn_drone.ardupilot_comparison.mission_definition import (CIRCLE_CENTER, CIRCLE_RADIUS_M, HOVER_POINT, NETWORK_PARAMETERS,
                                                              RETURN_HEIGHT_M, TAKEOFF_HEIGHT_M, WAYPOINT_NORTH)

ROUTE_POINT_SPACING_M = 0.2
PATH_ERROR_CHUNK = 500
AIRBORNE_HEIGHT_M = 0.5
HOVER_RADIUS_M = 2.5
HOVER_SKIP_S = 6
HOVER_WINDOW_S = 20
HOVER_END_MARGIN_S = 1
RING_TOLERANCE_M = 3
RING_JOIN_TRIM_S = 2
PWM_COMPARISON_HZ = 10.0
TAKEOFF_LANDING_TRIM_S = 8
TOP_SPEED_PERCENTILE = 99.5
HIGH_PERCENTILE = 95
CSV_PERIOD_S = 0.1
RESULT_DECIMALS = 3
MISSION_SUMMARY = ("take-off 20 m, waypoint 60 m north at 30 m, 30 s hold 60 m north-east at 40 m, "
                   "15 m circle at 30 m, return and land")


def planned_route(spacing=ROUTE_POINT_SPACING_M):
    edge = CIRCLE_CENTER + np.array([CIRCLE_RADIUS_M, 0, 0])

    def line(start, end):
        start, end = np.array(start, float), np.array(end, float)
        count = max(2, int(np.linalg.norm(end - start) / spacing))
        return [start + (end - start) * index / count for index in range(count + 1)]
    takeoff_top, return_top = [0, 0, TAKEOFF_HEIGHT_M], [0, 0, RETURN_HEIGHT_M]
    points = line([0, 0, 0], takeoff_top) + line(takeoff_top, WAYPOINT_NORTH) + line(WAYPOINT_NORTH, HOVER_POINT)
    points += line(HOVER_POINT, edge)
    points += [CIRCLE_CENTER + np.array([CIRCLE_RADIUS_M * math.cos(angle), CIRCLE_RADIUS_M * math.sin(angle), 0])
               for angle in np.linspace(0, 2 * math.pi, int(2 * math.pi * CIRCLE_RADIUS_M / spacing))]
    points += line(edge, return_top) + line(return_top, [0, 0, 0])
    return np.array(points)


PLANNED_ROUTE = planned_route()


def distance_from_route(north, east, height):
    positions = np.stack([north, east, height], 1)
    distance = np.empty(len(positions))
    for start in range(0, len(positions), PATH_ERROR_CHUNK):
        offsets = positions[start:start + PATH_ERROR_CHUNK, None, :] - PLANNED_ROUTE[None, :, :]
        distance[start:start + PATH_ERROR_CHUNK] = np.sqrt((offsets ** 2).sum(2)).min(1)
    return distance


def resample(time, values, rate_hz):
    uniform = np.arange(time[0], time[-1], 1.0 / rate_hz)
    return uniform, np.stack([np.interp(uniform, time, values[:, column]) for column in range(values.shape[1])], 1)


def hover_window(time, north, east, height):
    """The 30 s hold, taken as the time within 2.5 m of the hover point; the middle 20 s are measured."""
    distance = np.sqrt((north - HOVER_POINT[0]) ** 2 + (east - HOVER_POINT[1]) ** 2 + (height - HOVER_POINT[2]) ** 2)
    inside = np.nonzero(distance < HOVER_RADIUS_M)[0]
    entered, left = time[inside[0]], time[inside[-1]]
    return entered + HOVER_SKIP_S, min(entered + HOVER_SKIP_S + HOVER_WINDOW_S, left - HOVER_END_MARGIN_S), left


def tilt_from_roll_pitch(roll_deg, pitch_deg):
    return np.degrees(np.arccos(np.clip(np.cos(np.radians(roll_deg)) * np.cos(np.radians(pitch_deg)), -1, 1)))


def _longest_ring_stretch(time, on_ring):
    indices = np.nonzero(on_ring)[0]
    stretches = np.split(indices, np.nonzero(np.diff(indices) > 1)[0] + 1)
    longest = max(stretches, key=len)
    return longest[(time[longest] > time[longest[0]] + RING_JOIN_TRIM_S) & (time[longest] < time[longest[-1]] - RING_JOIN_TRIM_S)]


def flight_metrics(flight):
    time, north, east, height = flight.time, flight.north, flight.east, flight.height
    route_error = distance_from_route(north, east, height)
    airborne = height > AIRBORNE_HEIGHT_M
    hover_start, hover_end, hover_left = hover_window(time, north, east, height)
    hovering = (time >= hover_start) & (time <= hover_end)
    ring_radius = np.hypot(north - CIRCLE_CENTER[0], east - CIRCLE_CENTER[1])
    on_ring = ((time > hover_left) & (np.abs(ring_radius - CIRCLE_RADIUS_M) < RING_TOLERANCE_M)
               & (np.abs(height - CIRCLE_CENTER[2]) < RING_TOLERANCE_M))
    ring = _longest_ring_stretch(time, on_ring)
    pwm_time, pwm = resample(flight.pwm_time, flight.pwm, PWM_COMPARISON_HZ)
    hover_pwm = pwm[(pwm_time >= hover_start) & (pwm_time <= hover_end)]
    flight_pwm = pwm[(pwm_time > TAKEOFF_LANDING_TRIM_S) & (pwm_time < flight.duration - TAKEOFF_LANDING_TRIM_S)]
    speed = np.hypot(np.gradient(north, time), np.gradient(east, time))
    return {
        "mission time (s)": float(flight.duration),
        "route error mean (m)": float(route_error[airborne].mean()),
        "route error 95th percentile (m)": float(np.percentile(route_error[airborne], HIGH_PERCENTILE)),
        "route error max (m)": float(route_error[airborne].max()),
        "hover horizontal error mean (m)": float(np.hypot(north[hovering] - HOVER_POINT[0], east[hovering] - HOVER_POINT[1]).mean()),
        "hover horizontal wander, std (m)": float(np.sqrt(north[hovering].var() + east[hovering].var())),
        "hover height error mean (m)": float(np.abs(height[hovering] - HOVER_POINT[2]).mean()),
        "hover height wander, std (m)": float(height[hovering].std()),
        "highest point above the 40 m target (m)": float(height.max() - HOVER_POINT[2]),
        "circle radius error mean (m)": float(np.abs(ring_radius[ring] - CIRCLE_RADIUS_M).mean()),
        "circle radius error max (m)": float(np.abs(ring_radius[ring] - CIRCLE_RADIUS_M).max()),
        "circle time on the ring (s)": float(time[ring[-1]] - time[ring[0]] + 2 * RING_JOIN_TRIM_S),
        "circle height error max (m)": float(np.abs(height[ring] - CIRCLE_CENTER[2]).max()),
        "landing distance from take-off point (m)": float(np.hypot(north[-1], east[-1])),
        "steepest tilt (deg)": float(tilt_from_roll_pitch(flight.roll, flight.pitch).max()),
        "top horizontal speed (m/s)": float(np.percentile(speed, TOP_SPEED_PERCENTILE)),
        "PWM in hover, mean (us)": float(hover_pwm.mean()),
        "PWM in hover, spread between motors (us)": float(hover_pwm.mean(0).max() - hover_pwm.mean(0).min()),
        "PWM in hover, wander std per motor (us)": float(hover_pwm.std(0).mean()),
        "PWM in hover, mean change per 0.1 s (us)": float(np.abs(np.diff(hover_pwm, axis=0)).mean()),
        "PWM in flight, lowest (us)": float(flight_pwm.min()),
        "PWM in flight, highest (us)": float(flight_pwm.max()),
        "PWM in flight, mean change per 0.1 s (us)": float(np.abs(np.diff(flight_pwm, axis=0)).mean()),
    }, dict(hover_window=(float(hover_start), float(hover_end)), route_error=route_error, pwm_10hz=(pwm_time, pwm))


def write_ardupilot_csv(path, flight):
    time = np.arange(0, flight.duration, CSV_PERIOD_S)
    columns = [time, np.interp(time, flight.time, flight.north), np.interp(time, flight.time, flight.east),
               np.interp(time, flight.time, flight.height), np.interp(time, flight.attitude_time, flight.roll),
               np.interp(time, flight.attitude_time, flight.pitch)]
    columns += [np.interp(time, flight.pwm_time, flight.pwm[:, motor]) for motor in range(flight.pwm.shape[1])]
    np.savetxt(path, np.stack(columns, 1), delimiter=",", fmt="%.3f", comments="",
               header="t_s,north_m,east_m,height_m,roll_deg,pitch_deg,pwm1_us,pwm2_us,pwm3_us,pwm4_us")


def compare_flights(log_dir):
    from nn_drone.charts.comparison_charts import comparison_chart
    log_dir = Path(log_dir)
    flights = [read_ardupilot_flight(log_dir / "ardupilot_sitl.BIN"),
               read_network_flight(log_dir / "network_quad_x.npz", "Neural network, quad")]
    if (log_dir / "network_hex_x.npz").exists():
        flights.append(read_network_flight(log_dir / "network_hex_x.npz", "Neural network, hexa"))
    charted = len(flights)
    if (log_dir / "network_quad_x_nonoise.npz").exists():
        flights.append(read_network_flight(log_dir / "network_quad_x_nonoise.npz", "Quad network, no noise"))
    measured = [flight_metrics(flight) for flight in flights]
    ardupilot = flights[0]
    write_ardupilot_csv(log_dir / "ardupilot_extracted.csv", ardupilot)
    report = dict(mission=MISSION_SUMMARY, settings=NETWORK_PARAMETERS, ardupilot_ekf_vs_true_position_max_m=ardupilot.ekf_error_max,
                  log_rates_hz={flight.name: flight.log_rates for flight in flights},
                  ardupilot_hover_throttle=ardupilot.hover_throttle,
                  metrics={flight.name: {key: round(value, RESULT_DECIMALS) for key, value in metrics.items()}
                           for flight, (metrics, _) in zip(flights, measured)})
    (log_dir / "comparison.json").write_text(json.dumps(report, indent=1))
    keys = list(measured[0][0])
    width = max(len(key) for key in keys)
    print(f"{'':{width}}  " + "  ".join(f"{flight.name[:22]:>22}" for flight in flights))
    for key in keys:
        print(f"{key:{width}}  " + "  ".join(f"{metrics[key]:22.3f}" for metrics, _ in measured))
    print("ArduPilot EKF vs true position, max (m):", round(ardupilot.ekf_error_max, 3))
    print("ArduPilot hover throttle (CTUN.ThH):", ardupilot.hover_throttle)
    comparison_chart(log_dir / "comparison.png", flights[:charted], [traces for _, traces in measured[:charted]], PLANNED_ROUTE)
    return report
