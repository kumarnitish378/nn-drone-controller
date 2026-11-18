"""Network inputs.

Every input is size-normalised (lengths / L, times / sqrt(L)) so a heavy drone looks like the 1 kg one.
Layout: gyro(3) accelerometer(3) roll pitch heading-error sin cos position-error(3) velocity(3) battery
previous-motor-outputs(M) [integrated-position-error(3)] [history frames] [mass class].
"""
from dataclasses import dataclass

import numpy as np

from nn_drone.geometry import euler_from_quaternion, rotate_world_to_body, wrap_angle
from nn_drone.simulation.config import history_frame_size
from nn_drone.units import GRAVITY_MPS2

GYRO_SCALE_RAD_S = 8.0
ACCELEROMETER_SCALE = 2 * GRAVITY_MPS2
POSITION_ERROR_SCALE_M = 5.0
VELOCITY_SCALE_MPS = 5.0
BATTERY_CENTER_V = 15.6
BATTERY_SPAN_V = 1.2
MASS_CLASS_REFERENCE_DECADE = 1.0
MIN_ERROR_NORM = 1e-6


@dataclass(frozen=True)
class SensorNoise:
    gyro: float = 0.02
    accelerometer: float = 0.3
    roll_pitch: float = 0.005
    heading: float = 0.01
    position: float = 0.03
    velocity: float = 0.05
    battery: float = 0.03


@dataclass
class SensorFrame:
    """One frame of what the network sees, before history and mass class are appended."""
    features: np.ndarray
    history_features: np.ndarray


class ObservationBuilder:
    def __init__(self, profile, motor_count, position_error_clip, rng, noise_enabled, noise=SensorNoise()):
        self.profile = profile
        self.motor_count = motor_count
        self.position_error_clip = position_error_clip
        self.rng = rng
        self.noise_enabled = noise_enabled
        self.noise = noise

    def sense(self, state, airframe, target_position, target_heading, previous_action, error_integral):
        roll, pitch, heading = euler_from_quaternion(state.attitude)
        position_error = rotate_world_to_body(state.attitude, target_position - state.position)
        velocity = rotate_world_to_body(state.attitude, state.velocity)
        length = airframe.length_scale[:, None]
        time_scale = np.sqrt(length)
        gyro = state.body_rates.copy()
        accelerometer = state.specific_force_body.copy()
        heading_error = wrap_angle(target_heading - heading)
        battery = state.battery_voltage.copy()
        if self.noise_enabled:
            gyro, accelerometer, roll, pitch, heading_error, position_error, velocity, battery = self._add_noise(
                gyro, accelerometer, roll, pitch, heading_error, position_error, velocity, battery)
        gyro = gyro * time_scale
        position_error = position_error / length
        velocity = velocity / time_scale
        error_norm = np.linalg.norm(position_error, axis=1, keepdims=True)
        position_error = position_error * np.minimum(1.0, self.position_error_clip / np.maximum(error_norm, MIN_ERROR_NORM))
        columns = [gyro / GYRO_SCALE_RAD_S, accelerometer / ACCELEROMETER_SCALE,
                   (roll / np.pi)[:, None], (pitch / np.pi)[:, None],
                   np.sin(heading_error)[:, None], np.cos(heading_error)[:, None],
                   position_error / POSITION_ERROR_SCALE_M, velocity / VELOCITY_SCALE_MPS,
                   ((battery - BATTERY_CENTER_V) / BATTERY_SPAN_V)[:, None], previous_action]
        if self.profile.has_integrator_input:
            columns.append(rotate_world_to_body(state.attitude, error_integral) / self.profile.integrator.input_scale)
        features = np.concatenate(columns, axis=1).astype(np.float32)
        return SensorFrame(features=features, history_features=self._history_slice(features))

    def _add_noise(self, gyro, accelerometer, roll, pitch, heading_error, position_error, velocity, battery):
        noise, rng, count = self.noise, self.rng, len(roll)
        gyro += rng.normal(0, noise.gyro, (count, 3))
        accelerometer += rng.normal(0, noise.accelerometer, (count, 3))
        roll = roll + rng.normal(0, noise.roll_pitch, count)
        pitch = pitch + rng.normal(0, noise.roll_pitch, count)
        heading_error = heading_error + rng.normal(0, noise.heading, count)
        position_error += rng.normal(0, noise.position, (count, 3))
        velocity += rng.normal(0, noise.velocity, (count, 3))
        battery += rng.normal(0, noise.battery, count)
        return gyro, accelerometer, roll, pitch, heading_error, position_error, velocity, battery

    def _history_slice(self, features):
        """Gyro, accelerometer, roll, pitch, position error, velocity and motor outputs: no heading, no battery."""
        gyro_to_pitch = slice(0, 8)
        position_and_velocity = slice(10, 16)
        motor_outputs = slice(17, 17 + self.motor_count)
        return np.concatenate([features[:, gyro_to_pitch], features[:, position_and_velocity], features[:, motor_outputs]], axis=1)

    @staticmethod
    def mass_class(airframe):
        return (np.log10(airframe.nominal_mass) - MASS_CLASS_REFERENCE_DECADE).astype(np.float32)[:, None]


class ObservationHistory:
    """The network also sees earlier frames, `spacing` control steps apart (one reference step of size-normalised
    time); after a reset they are filled with the current one. Past frames are kept in a ring buffer."""

    def __init__(self, count, motor_count, frames, max_spacing_steps=1):
        self.frame_count = frames
        self.capacity = max(1, frames * max_spacing_steps)
        self.max_spacing_steps = max_spacing_steps
        self.past_frames = np.zeros((self.capacity, count, history_frame_size(motor_count)), np.float32)
        self.newest_slot = 0
        self.spacing = np.ones(count, np.int64)
        self.needs_fill = np.ones(count, bool)
        self.latest = None
        self.drones = np.arange(count)

    def set_spacing(self, indices, spacing_steps):
        if np.any(spacing_steps > self.max_spacing_steps):
            raise ValueError(f"history spacing {int(np.max(spacing_steps))} exceeds the {self.max_spacing_steps} steps provided")
        self.spacing[indices] = spacing_steps

    @property
    def frames(self):
        """Most recent first."""
        return [self.past_frames[(self.newest_slot - (age * self.spacing - 1)) % self.capacity, self.drones]
                for age in range(1, self.frame_count + 1)]

    def remember(self, sensor_frame):
        if self.frame_count and self.needs_fill.any():
            self.past_frames[:, self.needs_fill] = sensor_frame.history_features[self.needs_fill]
            self.needs_fill[:] = False
        self.latest = sensor_frame

    def advance(self):
        """The frame the network just used becomes the most recent past frame."""
        if self.latest is None or not self.frame_count:
            return
        self.newest_slot = (self.newest_slot + 1) % self.capacity
        self.past_frames[self.newest_slot] = self.latest.history_features

    def mark_reset(self, indices):
        self.needs_fill[indices] = True
