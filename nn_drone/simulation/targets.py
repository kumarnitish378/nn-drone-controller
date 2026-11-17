"""Training targets: random waypoints near the drone, some of them moving along lines and arcs."""
from dataclasses import dataclass

import numpy as np

from nn_drone.geometry import wrap_angle


@dataclass(frozen=True)
class TargetSampling:
    horizontal_offset_m: float = 4.0
    vertical_offset_m: float = 2.0
    speed_range: tuple = (0.5, 2.0)
    turn_rate: float = 0.5
    turning_share: float = 0.7
    waypoint_switch_probability: float = 1 / 400.0


class TargetTrack:
    def __init__(self, count, rng, moving_targets, moving_share, sampling=TargetSampling()):
        self.rng = rng
        self.moving_targets = moving_targets
        self.moving_share = moving_share
        self.sampling = sampling
        self.position = np.zeros((count, 3))
        self.heading = np.zeros(count)
        self.velocity = np.zeros((count, 3))
        self.turn_rate = np.zeros(count)

    def sample_new(self, indices, vehicle_position, length_scale):
        sampling, rng, count = self.sampling, self.rng, len(indices)
        horizontal, vertical = sampling.horizontal_offset_m, sampling.vertical_offset_m
        offset = np.stack([rng.uniform(-horizontal, horizontal, count), rng.uniform(-horizontal, horizontal, count),
                           rng.uniform(-vertical, vertical, count)], 1)
        length = length_scale[indices]
        self.position[indices] = vehicle_position[indices] + offset * length[:, None]
        self.heading[indices] = rng.uniform(-np.pi, np.pi, count)
        if not self.moving_targets:
            return
        moving = rng.random(count) < self.moving_share
        direction = rng.uniform(-np.pi, np.pi, count)
        speed = rng.uniform(*sampling.speed_range, count) * moving * np.sqrt(length)
        self.velocity[indices] = np.stack([speed * np.cos(direction), speed * np.sin(direction), np.zeros(count)], 1)
        self.turn_rate[indices] = (rng.uniform(-sampling.turn_rate, sampling.turn_rate, count) * moving
                                   * (rng.random(count) < sampling.turning_share) / np.sqrt(length))

    def move(self, control_period):
        cos_turn, sin_turn = np.cos(self.turn_rate * control_period), np.sin(self.turn_rate * control_period)
        velocity_x, velocity_y = self.velocity[:, 0].copy(), self.velocity[:, 1].copy()
        self.velocity[:, 0] = cos_turn * velocity_x - sin_turn * velocity_y
        self.velocity[:, 1] = sin_turn * velocity_x + cos_turn * velocity_y
        self.position += self.velocity * control_period[:, None]
        self.heading = wrap_angle(self.heading + self.turn_rate * control_period)

    def pick_switching(self, eligible):
        """Indices whose waypoint jumps this step, so the network learns transitions."""
        return np.nonzero((self.rng.random(len(eligible)) < self.sampling.waypoint_switch_probability) & eligible)[0]
