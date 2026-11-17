"""Rigid-body dynamics: motors with first-order lag, thrust and reaction torque, drag, wind, battery sag,
thinner air with altitude, and an optional solid ground."""
import numpy as np

from nn_drone.geometry import (euler_from_quaternion, quaternion_derivative, quaternion_from_euler,
                               rotate_body_to_world, rotate_world_to_body)
from nn_drone.simulation.state import RESTING_SPECIFIC_FORCE
from nn_drone.units import GRAVITY_MPS2

BATTERY_SAG_VOLTS = 0.9
ISA_DENSITY_LAPSE_PER_M = 2.2558e-5
ISA_DENSITY_EXPONENT = 4.2559
AIR_DENSITY_RATIO_BOUNDS = (0.3, 1.1)


def air_density_ratio(altitude_m):
    """International Standard Atmosphere density relative to sea level."""
    return np.clip(1.0 - ISA_DENSITY_LAPSE_PER_M * altitude_m, *AIR_DENSITY_RATIO_BOUNDS) ** ISA_DENSITY_EXPONENT


class RigidBodyPhysics:
    def __init__(self, layout, config, air_density_effect):
        self.motor_forward, self.motor_left = layout.arm_offsets(config.arm_length_m)
        self.spin = layout.spin_vector()
        self.nominal_voltage = config.nominal_battery_voltage
        self.solid_ground = config.solid_ground
        self.air_density_effect = air_density_effect

    def advance(self, state, airframe, throttle_command, period):
        state.motor_throttle += ((throttle_command - state.motor_throttle)
                                 * np.minimum(1.0, period / airframe.motor_time_constant)[:, None])
        voltage_factor = (state.battery_voltage / self.nominal_voltage) ** 2
        if self.air_density_effect:
            density = air_density_ratio(airframe.start_altitude + state.position[:, 2])
        else:
            density = np.ones(len(period))
        motor_thrust = airframe.max_motor_thrust[:, None] * (voltage_factor * density)[:, None] * state.motor_throttle ** 2
        total_thrust = motor_thrust.sum(1)
        state.battery_voltage = airframe.rest_voltage - BATTERY_SAG_VOLTS * (state.motor_throttle ** 2).mean(1)
        self._advance_rotation(state, airframe, motor_thrust, period)
        self._advance_translation(state, airframe, total_thrust, density, period)
        if self.solid_ground:
            self._resolve_ground_contact(state)

    def _advance_rotation(self, state, airframe, motor_thrust, period):
        torque = np.stack([(motor_thrust * self.motor_left).sum(1) * airframe.length_scale,
                           -(motor_thrust * self.motor_forward).sum(1) * airframe.length_scale,
                           -(motor_thrust * self.spin).sum(1) * airframe.yaw_torque_coefficient], 1)
        torque -= airframe.rotational_damping[:, None] * state.body_rates
        angular_momentum = airframe.inertia * state.body_rates
        state.body_rates += (torque - np.cross(state.body_rates, angular_momentum)) / airframe.inertia * period[:, None]
        attitude = state.attitude + quaternion_derivative(state.attitude, state.body_rates) * period[:, None]
        state.attitude = attitude / np.linalg.norm(attitude, axis=1, keepdims=True)

    def _advance_translation(self, state, airframe, total_thrust, density, period):
        thrust_body = np.zeros((len(total_thrust), 3))
        thrust_body[:, 2] = total_thrust
        force = (rotate_body_to_world(state.attitude, thrust_body)
                 - (airframe.drag_coefficient * density)[:, None] * (state.velocity - airframe.wind))
        acceleration = force / airframe.mass[:, None]
        state.specific_force_body = rotate_world_to_body(state.attitude, acceleration)
        acceleration[:, 2] -= GRAVITY_MPS2
        state.velocity += acceleration * period[:, None]
        state.position += state.velocity * period[:, None]

    @staticmethod
    def _resolve_ground_contact(state):
        resting = (state.position[:, 2] <= 0) & (state.velocity[:, 2] <= 0)
        touching_down = resting & ~state.on_ground
        state.touchdown_speed[touching_down] = np.maximum(state.touchdown_speed[touching_down],
                                                          -state.velocity[touching_down, 2])
        if resting.any():
            _, _, heading = euler_from_quaternion(state.attitude[resting])
            resting_count = int(resting.sum())
            state.attitude[resting] = quaternion_from_euler(np.zeros(resting_count), np.zeros(resting_count), heading)
            state.position[resting, 2] = 0
            state.velocity[resting] = 0
            state.body_rates[resting] = 0
            state.specific_force_body[resting] = RESTING_SPECIFIC_FORCE
        state.on_ground = resting
