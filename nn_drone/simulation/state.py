"""Batched vehicle state and per-vehicle physical parameters."""
from dataclasses import dataclass

import numpy as np

from nn_drone.units import GRAVITY_MPS2

LEVEL_ATTITUDE = np.array([1.0, 0.0, 0.0, 0.0])
RESTING_SPECIFIC_FORCE = np.array([0.0, 0.0, GRAVITY_MPS2])


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    body_rates: np.ndarray
    specific_force_body: np.ndarray
    motor_throttle: np.ndarray
    battery_voltage: np.ndarray
    on_ground: np.ndarray
    touchdown_speed: np.ndarray

    @classmethod
    def empty(cls, count, motor_count, nominal_voltage):
        attitude = np.zeros((count, 4))
        attitude[:, 0] = 1
        return cls(position=np.zeros((count, 3)), velocity=np.zeros((count, 3)), attitude=attitude,
                   body_rates=np.zeros((count, 3)), specific_force_body=np.zeros((count, 3)),
                   motor_throttle=np.zeros((count, motor_count)), battery_voltage=np.full(count, nominal_voltage),
                   on_ground=np.zeros(count, bool), touchdown_speed=np.zeros(count))


@dataclass
class Airframe:
    nominal_mass: np.ndarray
    length_scale: np.ndarray
    mass: np.ndarray
    inertia: np.ndarray
    max_motor_thrust: np.ndarray
    motor_time_constant: np.ndarray
    control_period: np.ndarray
    normalised_step_length: np.ndarray
    """One control step in reference steps of size-normalised time (exactly 1 when the period scales with size)."""
    yaw_torque_coefficient: np.ndarray
    drag_coefficient: np.ndarray
    rotational_damping: np.ndarray
    start_altitude: np.ndarray
    rest_voltage: np.ndarray
    wind: np.ndarray

    @classmethod
    def empty(cls, count, control_period):
        return cls(nominal_mass=np.ones(count), length_scale=np.ones(count), mass=np.zeros(count),
                   inertia=np.zeros((count, 3)), max_motor_thrust=np.zeros(count), motor_time_constant=np.zeros(count),
                   control_period=np.full(count, control_period), normalised_step_length=np.ones(count),
                   yaw_torque_coefficient=np.zeros(count),
                   drag_coefficient=np.zeros(count), rotational_damping=np.zeros(count), start_altitude=np.zeros(count),
                   rest_voltage=np.zeros(count), wind=np.zeros((count, 3)))

    @property
    def time_scale(self):
        return np.sqrt(self.length_scale)
