"""The six-waypoint route up to 1500 m with a circle and a rectangle, flown by a target point that moves along
straight lines at set speeds (like an autopilot's waypoint navigation). Axes: x forward, y left, z up,
metres from the take-off point."""
from dataclasses import dataclass

import numpy as np

WAYPOINTS = {"WP1": np.array([0, 0, 50.0]), "WP2": np.array([0, 0, 500.0]), "WP3": np.array([0, 300, 800.0]),
             "WP4": np.array([0, -100, 1300.0]), "WP5": np.array([0, 340, 400.0]), "WP6": np.array([0, -410, 1500.0])}
CIRCLE_CENTER = WAYPOINTS["WP5"]
CIRCLE_RADIUS_M = 100.0
RECTANGLE_CORNER = WAYPOINTS["WP6"]
RECTANGLE_SIDE_M = 120.0
RETURN_POINT = np.array([0, 0, 10.0])
LANDING_END = np.array([0, 0, -3.0])
FINAL_LANDING_SPEED_MPS = 0.5
FIRST_HOVER_S = 30
WAYPOINT_HOLD_S = 8
CIRCLE_HOLD_S = 6
CORNER_HOLD_S = 4

FIRST_HOVER = "WP1 hover 30 s"
CIRCLE_LAP = "circle lap"
LANDING_DESCENT = "landing descent"
HOLD_SEGMENTS = (FIRST_HOVER, "WP2 hold", "WP3 hold", "WP4 hold", "WP5 hold", "WP6 hold", "hold 10 m above take-off point")
END_OF_HOLD_WINDOW_S = 2.0
HOVER_SETTLE_S = 10.0


@dataclass(frozen=True)
class RouteSpeeds:
    horizontal_mps: float = 6.0
    climb_mps: float = 5.0
    descent_mps: float = 3.0
    final_landing_mps: float = FINAL_LANDING_SPEED_MPS

    def as_report(self):
        return dict(horizontal=self.horizontal_mps, climb=self.climb_mps, descent=self.descent_mps,
                    final_landing=self.final_landing_mps)


@dataclass
class RouteTrajectory:
    """Target position for every control step and which named segment it belongs to."""
    targets: np.ndarray
    segment_of_step: np.ndarray
    segment_names: list

    def segment(self, name):
        return self.segment_names.index(name)

    def steps_in(self, name):
        return np.nonzero(self.segment_of_step == self.segment(name))[0]

    def steps_where(self, name_predicate):
        return np.array([name_predicate(name) for name in self.segment_names])[self.segment_of_step]


class RouteBuilder:
    def __init__(self, period, speeds):
        self.period = period
        self.speeds = speeds
        self.targets, self.segment_of_step, self.segment_names = [], [], []
        self.position = np.zeros(3)

    def _start_segment(self, name):
        self.segment_names.append(name)
        return len(self.segment_names) - 1

    def go_to(self, point, name, descent_speed=None):
        descent_speed = descent_speed or self.speeds.descent_mps
        travel = np.asarray(point, float) - self.position
        vertical_time = travel[2] / self.speeds.climb_mps if travel[2] > 0 else -travel[2] / descent_speed
        duration = max(np.hypot(travel[0], travel[1]) / self.speeds.horizontal_mps, vertical_time, self.period)
        steps = int(np.ceil(duration / self.period))
        segment = self._start_segment(name)
        for step in range(1, steps + 1):
            self.targets.append(self.position + travel * step / steps)
            self.segment_of_step.append(segment)
        self.position = self.position + travel

    def hold(self, seconds, name):
        segment = self._start_segment(name)
        for _ in range(int(seconds / self.period)):
            self.targets.append(self.position.copy())
            self.segment_of_step.append(segment)

    def circle(self, center, radius, name):
        steps = int(2 * np.pi * radius / self.speeds.horizontal_mps / self.period)
        segment = self._start_segment(name)
        for step in range(1, steps + 1):
            angle = 2 * np.pi * step / steps
            self.targets.append(np.array([center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]]))
            self.segment_of_step.append(segment)
        self.position = self.targets[-1].copy()

    def build(self):
        return RouteTrajectory(np.array(self.targets), np.array(self.segment_of_step), self.segment_names)


def build_long_route(period, speeds):
    route = RouteBuilder(period, speeds)
    route.go_to(WAYPOINTS["WP1"], "take-off to 50 m")
    route.hold(FIRST_HOVER_S, FIRST_HOVER)
    for name in ("WP2", "WP3", "WP4", "WP5"):
        route.go_to(WAYPOINTS[name], f"to {name}")
        route.hold(WAYPOINT_HOLD_S, f"{name} hold")
    route.go_to(CIRCLE_CENTER + [CIRCLE_RADIUS_M, 0, 0], "to circle edge")
    route.hold(CIRCLE_HOLD_S, "circle edge hold")
    route.circle(CIRCLE_CENTER, CIRCLE_RADIUS_M, CIRCLE_LAP)
    route.hold(CIRCLE_HOLD_S, "after circle hold")
    route.go_to(WAYPOINTS["WP6"], "to WP6")
    route.hold(WAYPOINT_HOLD_S, "WP6 hold")
    corners = [RECTANGLE_CORNER + [RECTANGLE_SIDE_M, 0, 0], RECTANGLE_CORNER + [RECTANGLE_SIDE_M, RECTANGLE_SIDE_M, 0],
               RECTANGLE_CORNER + [0, RECTANGLE_SIDE_M, 0], RECTANGLE_CORNER]
    for number, corner in enumerate(corners, start=1):
        route.go_to(corner, f"rectangle side {number}")
        route.hold(CORNER_HOLD_S, f"rectangle corner {number} hold")
    route.go_to(RETURN_POINT, "return to take-off point")
    route.hold(WAYPOINT_HOLD_S, "hold 10 m above take-off point")
    route.go_to(LANDING_END, LANDING_DESCENT, descent_speed=speeds.final_landing_mps)
    return route.build()


def rectangle_outline_distance(positions):
    """Horizontal distance from the rectangle's outline."""
    along = positions[..., 0] - RECTANGLE_CORNER[0]
    across = positions[..., 1] - RECTANGLE_CORNER[1]
    outside_along = np.maximum(np.maximum(-along, along - RECTANGLE_SIDE_M), 0)
    outside_across = np.maximum(np.maximum(-across, across - RECTANGLE_SIDE_M), 0)
    inside = np.minimum(np.minimum(along, RECTANGLE_SIDE_M - along), np.minimum(across, RECTANGLE_SIDE_M - across))
    is_outside = (outside_along > 0) | (outside_across > 0)
    return np.where(is_outside, np.hypot(outside_along, outside_across), np.abs(inside))


def summarize_route_tracking(route, period, distance_to_target, positions, drone=None):
    """Per-drone numbers when `drone` is given, otherwise the median over all drones."""
    pick = (lambda values: float(values[drone])) if drone is not None else (lambda values: float(np.median(values)))
    window = int(END_OF_HOLD_WINDOW_S / period)
    summary = {}
    for name in HOLD_SEGMENTS:
        summary[name + ": error at end (m)"] = pick(distance_to_target[route.steps_in(name)[-window:]].mean(0))
    settled_hover = route.steps_in(FIRST_HOVER)[int(HOVER_SETTLE_S / period):]
    summary["WP1 hover: max drift after first 10 s (m)"] = pick(distance_to_target[settled_hover].max(0))
    lap = route.segment_of_step == route.segment(CIRCLE_LAP)
    radial = np.abs(np.hypot(positions[lap, :, 0] - CIRCLE_CENTER[0], positions[lap, :, 1] - CIRCLE_CENTER[1]) - CIRCLE_RADIUS_M)
    summary["circle: mean / max distance from line (m)"] = [pick(radial.mean(0)), pick(radial.max(0))]
    summary["circle: max altitude error (m)"] = pick(np.abs(positions[lap, :, 2] - CIRCLE_CENTER[2]).max(0))
    outline = rectangle_outline_distance(positions[route.steps_where(lambda name: name.startswith("rectangle"))])
    summary["rectangle: mean / max distance from outline (m)"] = [pick(outline.mean(0)), pick(outline.max(0))]
    legs = route.steps_where(lambda name: name.startswith("to ") or name.startswith("return"))
    summary["legs: mean / max distance behind moving target (m)"] = [pick(distance_to_target[legs].mean(0)),
                                                                      pick(distance_to_target[legs].max(0))]
    return summary
