"""Physical constants and the motor-command convention shared by the simulator, the network and ArduPilot."""
import numpy as np

GRAVITY_MPS2 = 9.81

PWM_DISARMED_US = 1000.0
PWM_FULL_THROTTLE_US = 2000.0
PWM_CENTER_US = 1500.0
PWM_HALF_RANGE_US = 500.0
PWM_SPAN_US = PWM_FULL_THROTTLE_US - PWM_DISARMED_US

CENTIMETRES_PER_METRE = 100.0
MILLIMETRES_PER_METRE = 1000.0
MILLISECONDS_PER_SECOND = 1000.0
MICROSECONDS_PER_SECOND = 1e6


def action_to_pwm(action):
    """Network output in [-1, 1] -> motor pulse width in microseconds."""
    return PWM_CENTER_US + PWM_HALF_RANGE_US * action


def pwm_to_action(pwm_us):
    return (pwm_us - PWM_CENTER_US) / PWM_HALF_RANGE_US


def pwm_to_throttle(pwm_us):
    """Pulse width -> throttle fraction in [0, 1]."""
    return (pwm_us - PWM_DISARMED_US) / PWM_SPAN_US


def clip_action(action):
    return np.clip(action, -1.0, 1.0).astype(np.float64)
