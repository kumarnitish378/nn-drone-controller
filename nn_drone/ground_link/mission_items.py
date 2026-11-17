"""Mission items as uploaded by a ground station (MAVLink MISSION_ITEM(_INT) fields)."""
from dataclasses import dataclass

from pymavlink.dialects.v20 import ardupilotmega as mavlink

FIRST_NON_NAVIGATION_COMMAND = 100
HOME_SEQUENCE = 0


@dataclass
class MissionItem:
    seq: int
    frame: int
    command: int
    current: int = 0
    autocontinue: int = 1
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def is_navigation(self):
        return self.command < FIRST_NON_NAVIGATION_COMMAND

    @property
    def has_position(self):
        return abs(self.x) > 1e-9 or abs(self.y) > 1e-9


def home_item(home):
    return MissionItem(seq=HOME_SEQUENCE, frame=mavlink.MAV_FRAME_GLOBAL, command=mavlink.MAV_CMD_NAV_WAYPOINT,
                       x=home.latitude, y=home.longitude, z=home.altitude_asl)
