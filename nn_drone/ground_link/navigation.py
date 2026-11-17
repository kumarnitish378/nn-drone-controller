"""Guidance: a target point (the "carrot") moves toward the current goal at the navigation speeds;
the network flies to wherever the carrot is."""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from nn_drone.units import CENTIMETRES_PER_METRE

MIN_NAVIGATION_SPEED_MPS = 0.1
FACE_TRAVEL_MIN_DISTANCE_LENGTHS = 3.0
"""Turn toward the travel direction only when the goal is further than this (in drone lengths)."""
LANDING_CARROT_DEPTH_LENGTHS = 3.0
"""The landing carrot keeps sinking until this far below ground, pressing the drone onto it."""
FACE_TRAVEL = 0.5


class GoalKind(Enum):
    GO_TO = "goto"
    HOLD = "hold"
    CIRCLE = "circle"
    LAND = "land"


@dataclass
class Goal:
    kind: GoalKind
    position: np.ndarray = None
    hold_s: float = 0.0
    center: np.ndarray = None
    radius: float = 0.0
    turns: float = 0.0
    arrived_at: float = None
    start_angle: float = None
    angle_flown: float = 0.0

    def __post_init__(self):
        if self.position is not None:
            self.position = np.array(self.position, float)

    @classmethod
    def go_to(cls, position, hold_s=0.0):
        return cls(GoalKind.GO_TO, position, hold_s=hold_s)

    @classmethod
    def hold(cls, position=None):
        """Hold at `position`, or wherever the carrot is when the goal starts."""
        return cls(GoalKind.HOLD, position)

    @classmethod
    def land(cls, position):
        return cls(GoalKind.LAND, position)

    @classmethod
    def circle(cls, edge, center, radius, turns):
        """Positive radius circles clockwise seen from above."""
        return cls(GoalKind.CIRCLE, edge, center=center, radius=radius, turns=turns)


@dataclass(frozen=True)
class NavigationSpeeds:
    horizontal: float
    climb: float
    descent: float
    landing: float
    landing_slow_below: float
    arrival_radius: float
    face_travel: bool

    @classmethod
    def from_parameters(cls, parameters):
        per_metre = CENTIMETRES_PER_METRE
        return cls(horizontal=parameters["WPNAV_SPEED"] / per_metre, climb=parameters["WPNAV_SPEED_UP"] / per_metre,
                   descent=parameters["WPNAV_SPEED_DN"] / per_metre, landing=parameters["LAND_SPEED"] / per_metre,
                   landing_slow_below=parameters["LAND_ALT_LOW"] / per_metre,
                   arrival_radius=parameters["WPNAV_RADIUS"] / per_metre,
                   face_travel=parameters["WP_YAW_BEHAVIOR"] >= FACE_TRAVEL)


@dataclass
class GoalProgress:
    finished: bool = False
    reached_waypoint: bool = False


class CarrotGuidance:
    def __init__(self, length_scale):
        self.length_scale = length_scale
        self.carrot = np.zeros(3)
        self.target_heading = 0.0

    def advance(self, goal, speeds, vehicle_position, time, period):
        """Move the carrot one control period toward `goal`."""
        if goal.position is None:
            goal.position = self.carrot.copy()
        if goal.kind in (GoalKind.GO_TO, GoalKind.HOLD):
            return self._toward_point(goal, speeds, vehicle_position, time, period)
        if goal.kind is GoalKind.CIRCLE:
            return self._around_circle(goal, speeds, period)
        self._descend(goal, speeds, vehicle_position, period)
        return GoalProgress()

    def _toward_point(self, goal, speeds, vehicle_position, time, period):
        travel = goal.position - self.carrot
        horizontal = np.hypot(travel[0], travel[1])
        if travel[2] > 0:
            vertical_time = travel[2] / max(speeds.climb, MIN_NAVIGATION_SPEED_MPS)
        else:
            vertical_time = -travel[2] / max(speeds.descent, MIN_NAVIGATION_SPEED_MPS)
        travel_time = max(horizontal / max(speeds.horizontal, MIN_NAVIGATION_SPEED_MPS), vertical_time)
        if travel_time > period:
            self.carrot = self.carrot + travel * (period / travel_time)
            if speeds.face_travel and horizontal > FACE_TRAVEL_MIN_DISTANCE_LENGTHS * self.length_scale:
                self.target_heading = math.atan2(travel[1], travel[0])
            return GoalProgress()
        self.carrot = goal.position.copy()
        near = np.linalg.norm(vehicle_position - goal.position) < speeds.arrival_radius
        progress = GoalProgress()
        if goal.kind is GoalKind.GO_TO and near:
            if goal.arrived_at is None:
                goal.arrived_at = time
                progress.reached_waypoint = True
            progress.finished = time - goal.arrived_at >= goal.hold_s
        return progress

    def _around_circle(self, goal, speeds, period):
        if goal.start_angle is None:
            offset = self.carrot - goal.center
            goal.start_angle = math.atan2(offset[1], offset[0])
            goal.angle_flown = 0.0
        direction = -1.0 if goal.radius > 0 else 1.0
        goal.angle_flown += speeds.horizontal * period / abs(goal.radius)
        angle = goal.start_angle + direction * goal.angle_flown
        radius = abs(goal.radius)
        self.carrot = np.array([goal.center[0] + radius * math.cos(angle), goal.center[1] + radius * math.sin(angle),
                                goal.center[2]])
        if speeds.face_travel:
            self.target_heading = angle + direction * math.pi / 2
        finished = goal.angle_flown >= 2 * math.pi * goal.turns
        return GoalProgress(finished=finished, reached_waypoint=finished)

    def _descend(self, goal, speeds, vehicle_position, period):
        self.carrot[:2] = goal.position[:2]
        rate = speeds.landing if vehicle_position[2] < speeds.landing_slow_below else speeds.descent
        if self.carrot[2] > -LANDING_CARROT_DEPTH_LENGTHS * self.length_scale:
            self.carrot[2] -= rate * period
