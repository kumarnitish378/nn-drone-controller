"""Integral of the position error ("I term" input), with the anti-windup rule of each profile."""
import numpy as np

from nn_drone.simulation.config import REFERENCE_CONTROL_PERIOD_S, IntegratorSettleRule


class PositionErrorIntegrator:
    def __init__(self, count, settings):
        self.settings = settings
        self.value = np.zeros((count, 3))
        self.previous_target = np.zeros((count, 3))

    def reset(self, indices):
        self.value[indices] = 0

    def set_reference_target(self, indices, target_position):
        self.previous_target[indices] = target_position[indices]

    def update(self, position_error, velocity, target_position, length_scale, control_period, normalised_step_length):
        """Integrates the size-normalised error over size-normalised time."""
        settings = self.settings
        length = length_scale[:, None]
        time_scale = np.sqrt(length_scale)
        distance = np.linalg.norm(position_error, axis=1) / length_scale
        if settings.settle_rule is IntegratorSettleRule.WITHIN_CAPTURE_RADIUS:
            settled = distance < settings.capture_radius
        else:
            target_velocity = self._target_velocity(target_position, control_period, time_scale)
            if settings.settle_rule is IntegratorSettleRule.SLOW_RELATIVE_TO_TARGET:
                settled = ((distance < settings.capture_radius)
                           & (np.linalg.norm(velocity - target_velocity, axis=1) < settings.settled_speed))
            else:
                settled = ((distance < settings.capture_radius)
                           & (np.linalg.norm(velocity, axis=1) / time_scale < settings.settled_speed)
                           & (np.linalg.norm(target_velocity, axis=1) / time_scale < settings.still_target_speed))
        self.value += ((position_error / length) * settled[:, None] * REFERENCE_CONTROL_PERIOD_S
                       * normalised_step_length[:, None])
        np.clip(self.value, -settings.clamp, settings.clamp, out=self.value)

    def _target_velocity(self, target_position, control_period, time_scale):
        """Zero on a waypoint jump, so a jump does not look like a fast-moving target."""
        target_velocity = (target_position - self.previous_target) / control_period[:, None]
        jumped = np.linalg.norm(target_velocity, axis=1) > self.settings.target_jump_speed * time_scale
        target_velocity[jumped] = 0.0
        self.previous_target = target_position.copy()
        return target_velocity
