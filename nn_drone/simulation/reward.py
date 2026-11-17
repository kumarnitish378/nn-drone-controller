"""Training reward: stay alive, get close to the target, face the right way, and move the motors smoothly."""
import numpy as np

SURVIVAL_BONUS = 1.0
NEAR_TARGET_BONUS = 1.5
NEAR_TARGET_SHARPNESS = 2.0
AT_TARGET_BONUS = 1.0
AT_TARGET_SHARPNESS = 10.0
DISTANCE_PENALTY = 0.25
DISTANCE_PENALTY_CAP = 6.0
BODY_RATE_PENALTY = 0.05
HEADING_ERROR_PENALTY = 0.15
CRASH_REWARD = -10.0
REWARD_SCALE = 0.1

CRASH_DISTANCE = 12.0
CRASH_SPEED = 20.0


def flight_reward(weights, distance, heading_error, body_rate, action_change, speed, tilt):
    """All lengths and speeds are size-normalised."""
    return (SURVIVAL_BONUS
            + NEAR_TARGET_BONUS * np.exp(-NEAR_TARGET_SHARPNESS * distance)
            + AT_TARGET_BONUS * np.exp(-AT_TARGET_SHARPNESS * distance)
            - DISTANCE_PENALTY * np.minimum(distance, DISTANCE_PENALTY_CAP)
            - BODY_RATE_PENALTY * body_rate
            - HEADING_ERROR_PENALTY * heading_error
            - weights.action_change_squared * (action_change ** 2).sum(1)
            - weights.action_change_absolute * np.abs(action_change).sum(1)
            - weights.excess_speed * np.maximum(speed - weights.free_speed, 0.0)
            - weights.excess_tilt * np.maximum(tilt - weights.free_tilt_rad, 0.0))


def has_crashed(upright_cosine, distance, speed):
    return (upright_cosine < 0.0) | (distance > CRASH_DISTANCE) | (speed > CRASH_SPEED)
