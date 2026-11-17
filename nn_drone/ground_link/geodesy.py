"""Local flat-earth frame around the home point. Local axes: x north, y west, z up (the simulator's axes)."""
import math
from dataclasses import dataclass

import numpy as np

METRES_PER_DEGREE_LATITUDE = 111319.5
ABSOLUTE_ALTITUDE_FRAMES = (0, 5)
"""MAV_FRAME_GLOBAL and MAV_FRAME_GLOBAL_INT: altitude above sea level. Other frames are relative to home."""


@dataclass(frozen=True)
class HomePoint:
    latitude: float
    longitude: float
    altitude_asl: float

    @property
    def metres_per_degree_longitude(self):
        return METRES_PER_DEGREE_LATITUDE * math.cos(math.radians(self.latitude))

    def to_local(self, latitude, longitude, height_above_home):
        north = (latitude - self.latitude) * METRES_PER_DEGREE_LATITUDE
        west = -(longitude - self.longitude) * self.metres_per_degree_longitude
        return np.array([north, west, height_above_home])

    def to_global(self, local_position):
        latitude = self.latitude + local_position[0] / METRES_PER_DEGREE_LATITUDE
        longitude = self.longitude - local_position[1] / self.metres_per_degree_longitude
        return latitude, longitude

    def height_above_home(self, altitude, coordinate_frame):
        return altitude - self.altitude_asl if coordinate_frame in ABSOLUTE_ALTITUDE_FRAMES else altitude

    def offset(self, north_m, east_m):
        """Latitude and longitude of a point north and east of home."""
        return (self.latitude + north_m / METRES_PER_DEGREE_LATITUDE,
                self.longitude + east_m / self.metres_per_degree_longitude)
