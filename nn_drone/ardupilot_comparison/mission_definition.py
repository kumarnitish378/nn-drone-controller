"""The one mission and the settings that both ArduCopter SITL and the neural-network simulator fly."""
from dataclasses import dataclass

import numpy as np
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.ground_link.geodesy import HomePoint

HOME = HomePoint(28.5355, 77.3910, 200.0)
MASS_KG = 3.0
"""ArduPilot's default SITL quad."""


@dataclass(frozen=True)
class MissionStep:
    command: int
    has_position: bool
    north_m: float
    east_m: float
    height_m: float
    param1: float = 0
    param3: float = 0


MISSION = (
    MissionStep(mavlink.MAV_CMD_NAV_TAKEOFF, False, 0, 0, 20),
    MissionStep(mavlink.MAV_CMD_NAV_WAYPOINT, True, 60, 0, 30),
    MissionStep(mavlink.MAV_CMD_NAV_LOITER_TIME, True, 60, 60, 40, param1=30),
    MissionStep(mavlink.MAV_CMD_NAV_LOITER_TURNS, True, 0, 60, 30, param1=1, param3=15),
    MissionStep(mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, False, 0, 0, 0),
)
"""Take off to 20 m; 60 m north climbing to 30 m; 60 m east climbing to 40 m and hold 30 s;
one 15 m circle at 30 m; return and land."""

WAYPOINT_NORTH = np.array([60, 0, 30.0])
HOVER_POINT = np.array([60, 60, 40.0])
CIRCLE_CENTER = np.array([0, 60, 30.0])
CIRCLE_RADIUS_M = 15.0
TAKEOFF_HEIGHT_M = 20.0
RETURN_HEIGHT_M = 30.0

NETWORK_PARAMETERS = {
    "WPNAV_SPEED": 300, "WPNAV_SPEED_UP": 300, "WPNAV_SPEED_DN": 180, "WPNAV_RADIUS": 144, "RTL_ALT": 1500,
    "LAND_SPEED": 50, "SIM_WIND_SPD": 2, "SIM_WIND_DIR": 45, "AUTO_OPTIONS": 3,
}
"""Older ArduCopter names in centimetres, as the network simulator's ground link uses them."""

ARDUPILOT_PARAMETERS = {
    "WP_SPD": 3.0, "WP_SPD_UP": 3.0, "WP_SPD_DN": 1.8, "WP_RADIUS_M": 1.44, "RTL_ALT_M": 15.0, "LAND_SPD_MS": 0.5,
    "SIM_WIND_SPD": 2, "SIM_WIND_DIR": 45, "AUTO_OPTIONS": 3,
}
"""The same settings for ArduCopter 4.7, which renamed these parameters and moved them to metres."""


def mission_item_rows():
    """(command, frame, latitude, longitude, height, param1, param3); item 0 is home (ArduPilot convention)."""
    rows = [(mavlink.MAV_CMD_NAV_WAYPOINT, mavlink.MAV_FRAME_GLOBAL, HOME.latitude, HOME.longitude, HOME.altitude_asl, 0, 0)]
    for step in MISSION:
        latitude, longitude = HOME.offset(step.north_m, step.east_m) if step.has_position else (0, 0)
        rows.append((step.command, mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, latitude, longitude, step.height_m,
                     step.param1, step.param3))
    return rows
