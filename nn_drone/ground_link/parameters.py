"""Vehicle parameters exposed to the ground station. Navigation parameters use ArduCopter's names and units."""
from nn_drone.simulation.scaling import length_scale, speed_scale

REFERENCE_HORIZONTAL_SPEED_CMS = 250
REFERENCE_VERTICAL_SPEED_CMS = 250
REFERENCE_DESCENT_SPEED_CMS = 150
REFERENCE_WAYPOINT_RADIUS_CM = 100
QUAD_FRAME_CLASS = 1.0
HEXA_FRAME_CLASS = 2.0
X_FRAME_TYPE = 1.0
ALLOW_ARMING_AND_TAKEOFF_IN_AUTO = 3.0

SPEEDUP = "SIM_SPEEDUP"
WIND_PARAMETERS = ("SIM_WIND_SPD", "SIM_WIND_DIR")
AIRFRAME_PREFIX = "SIM_"
NETWORK_PARAMETERS = ("NN_ERR_CLIP",)


def default_parameters(frame, mass_kg):
    """Speeds default to values that scale with the drone's size, like the network's own time scale."""
    speed = speed_scale(mass_kg)
    return {
        "SIM_MASS": mass_kg,
        "SIM_TW_RATIO": 2.8,
        "SIM_WIND_SPD": 0.0,
        "SIM_WIND_DIR": 0.0,
        "SIM_BATT_VOLT": 16.0,
        "SIM_NOISE": 1.0,
        SPEEDUP: 1.0,
        "NN_ERR_CLIP": 2.5,
        "WPNAV_SPEED": round(REFERENCE_HORIZONTAL_SPEED_CMS * speed),
        "WPNAV_SPEED_UP": round(REFERENCE_VERTICAL_SPEED_CMS * speed),
        "WPNAV_SPEED_DN": round(REFERENCE_DESCENT_SPEED_CMS * speed),
        "WPNAV_RADIUS": round(REFERENCE_WAYPOINT_RADIUS_CM * length_scale(mass_kg)),
        "LAND_SPEED": 50.0,
        "LAND_ALT_LOW": 1000.0,
        "RTL_ALT": 1500.0,
        "CIRCLE_RADIUS": 1000.0,
        "WP_YAW_BEHAVIOR": 1.0,
        "MOT_SPIN_ARM": 0.15,
        "DISARM_DELAY": 10.0,
        "AUTO_OPTIONS": ALLOW_ARMING_AND_TAKEOFF_IN_AUTO,
        "FRAME_CLASS": QUAD_FRAME_CLASS if frame == "quad_x" else HEXA_FRAME_CLASS,
        "FRAME_TYPE": X_FRAME_TYPE,
        "SYSID_THISMAV": 1.0,
    }


def rebuilds_airframe(name):
    return (name.startswith(AIRFRAME_PREFIX) and name != SPEEDUP) or name in NETWORK_PARAMETERS
