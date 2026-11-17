"""Shared pieces of the flight tests: crash rules and a recorder for the first drone's flight."""
import numpy as np

from nn_drone.geometry import euler_from_quaternion, tilt_degrees, upright_cosine

RUNAWAY_SPEED_MPS = 20.0
FELL_BELOW_START_M = -3.0
NOMINAL_DRONE = 0


def lost_control(env, runaway_speed=RUNAWAY_SPEED_MPS):
    state = env.state
    return (upright_cosine(state.attitude) < 0) | (np.linalg.norm(state.velocity, axis=1) > runaway_speed)


def crashed_in_flight(env):
    """Flipped over, running away, or fell below the start point. Distance from the target is not a crash here."""
    return lost_control(env) | (env.state.position[:, 2] < FELL_BELOW_START_M)


def current_tilt_degrees(env):
    return tilt_degrees(env.state.attitude)


class FlightRecorder:
    """Samples the first drone every `every_steps` control steps."""

    def __init__(self, every_steps=1):
        self.every_steps = every_steps
        self.samples = {name: [] for name in ("time", "position", "target", "pwm", "roll_pitch_yaw", "battery_voltage",
                                              "vertical_speed")}

    def capture(self, step, time, env, pwm, target=None):
        if step % self.every_steps:
            return
        state = env.state
        roll, pitch, yaw = euler_from_quaternion(state.attitude)
        self.samples["time"].append(time)
        self.samples["position"].append(state.position[NOMINAL_DRONE].copy())
        self.samples["target"].append(np.array(env.targets.position[NOMINAL_DRONE] if target is None else target, float))
        self.samples["pwm"].append(pwm[NOMINAL_DRONE].copy())
        self.samples["roll_pitch_yaw"].append([roll[NOMINAL_DRONE], pitch[NOMINAL_DRONE], yaw[NOMINAL_DRONE]])
        self.samples["battery_voltage"].append(state.battery_voltage[NOMINAL_DRONE])
        self.samples["vertical_speed"].append(state.velocity[NOMINAL_DRONE, 2])

    def arrays(self):
        return {name: np.array(values) for name, values in self.samples.items()}
