"""The one mission and the settings that both ArduCopter SITL and the neural-network simulator fly."""
import math
from pymavlink.dialects.v20 import ardupilotmega as mav

HOME = (28.5355, 77.3910, 200.0)          # lat, lon, altitude above sea level (m)
MASS = 3.0                                 # kg: ArduPilot's default SITL quad

def ll(north, east):
    return HOME[0] + north / 111319.5, HOME[1] + east / (111319.5 * math.cos(math.radians(HOME[0])))

# (command, has position, north m, east m, height m, param1, param3)
MISSION = [
    (mav.MAV_CMD_NAV_TAKEOFF,          False,  0,  0, 20,  0,  0),   # 1 take off to 20 m
    (mav.MAV_CMD_NAV_WAYPOINT,         True,  60,  0, 30,  0,  0),   # 2 north 60 m, climb to 30 m
    (mav.MAV_CMD_NAV_LOITER_TIME,      True,  60, 60, 40, 30,  0),   # 3 east 60 m, climb to 40 m, hold 30 s
    (mav.MAV_CMD_NAV_LOITER_TURNS,     True,   0, 60, 30,  1, 15),   # 4 one 15 m circle at 30 m
    (mav.MAV_CMD_NAV_RETURN_TO_LAUNCH, False,  0,  0,  0,  0,  0),   # 5 return and land
]

# navigation and environment for the neural-network simulator (older ArduCopter names, cm units)
PARAMS = {
    "WPNAV_SPEED": 300,       # cm/s
    "WPNAV_SPEED_UP": 300,    # cm/s
    "WPNAV_SPEED_DN": 180,    # cm/s
    "WPNAV_RADIUS": 144,      # cm
    "RTL_ALT": 1500,          # cm
    "LAND_SPEED": 50,         # cm/s
    "SIM_WIND_SPD": 2,        # m/s
    "SIM_WIND_DIR": 45,       # deg
    "AUTO_OPTIONS": 3,        # allow arming and take-off in AUTO
}

# the same settings for ArduCopter 4.7, which renamed these parameters and moved them to metres
AP_PARAMS = {
    "WP_SPD": 3.0,            # m/s
    "WP_SPD_UP": 3.0,         # m/s
    "WP_SPD_DN": 1.8,         # m/s
    "WP_RADIUS_M": 1.44,      # m
    "RTL_ALT_M": 15.0,        # m
    "LAND_SPD_MS": 0.5,       # m/s
    "SIM_WIND_SPD": 2,        # m/s
    "SIM_WIND_DIR": 45,       # deg, direction the wind comes from
    "AUTO_OPTIONS": 3,
}
