"""Domain randomisation: how each simulated drone differs from the nominal one at reset."""
from dataclasses import dataclass

import numpy as np

from nn_drone.geometry import quaternion_from_euler
from nn_drone.simulation import scaling
from nn_drone.simulation.state import LEVEL_ATTITUDE, RESTING_SPECIFIC_FORCE
from nn_drone.units import GRAVITY_MPS2


@dataclass(frozen=True)
class AirframeVariation:
    mass_factor: tuple = (0.8, 1.25)
    inertia_factor: tuple = (0.8, 1.25)
    thrust_factor: tuple = (0.85, 1.15)
    motor_lag_factor: tuple = (0.7, 1.4)
    yaw_torque_factor: tuple = (0.8, 1.25)
    battery_rest_voltage: tuple = (14.4, 16.8)
    max_wind_speed: float = 3.0


@dataclass(frozen=True)
class StartDisturbance:
    speed: float = 2.0
    roll_pitch_rad: float = 0.7
    body_rate: float = 2.0
    motor_throttle_factor: tuple = (0.6, 1.2)


class AirframeSampler:
    """Fills `Airframe` and the initial `VehicleState` for the vehicles being reset.

    The order of random draws is part of the contract: it keeps training runs reproducible.
    """

    def __init__(self, config, layout, rng, randomize, variation=AirframeVariation(), disturbance=StartDisturbance()):
        self.config = config
        self.layout = layout
        self.rng = rng
        self.randomize = randomize
        self.variation = variation
        self.disturbance = disturbance

    def _factor(self, bounds, count):
        if self.randomize:
            return self.rng.uniform(bounds[0], bounds[1], count)
        return np.ones(count)

    def _nominal_masses(self, count):
        config = self.config
        if config.nominal_mass_range_kg is not None and self.randomize:
            low, high = np.log10(config.nominal_mass_range_kg[0]), np.log10(config.nominal_mass_range_kg[1])
            return 10 ** self.rng.uniform(low, high, count)
        return np.full(count, float(config.nominal_mass_kg))

    def sample_airframe(self, airframe, indices):
        config, variation, count = self.config, self.variation, len(indices)
        nominal_mass = self._nominal_masses(count)
        length = scaling.length_scale(nominal_mass)
        airframe.nominal_mass[indices] = nominal_mass
        airframe.length_scale[indices] = length
        airframe.mass[indices] = nominal_mass * self._factor(variation.mass_factor, count)
        shared_inertia_factor = self._factor(variation.inertia_factor, count)
        airframe.inertia[indices] = (np.array(config.inertia_kg_m2) * scaling.inertia_scale(nominal_mass)[:, None]
                                     * np.stack([shared_inertia_factor] * 3, 1))
        airframe.max_motor_thrust[indices] = (config.thrust_to_weight * nominal_mass * GRAVITY_MPS2 / self.layout.motor_count
                                              * self._factor(variation.thrust_factor, count))
        airframe.motor_time_constant[indices] = (config.motor_time_constant_s * np.sqrt(length)
                                                 * self._factor(variation.motor_lag_factor, count))
        airframe.control_period[indices] = config.control_period_s * np.sqrt(length)
        airframe.yaw_torque_coefficient[indices] = (config.yaw_torque_per_thrust_m * length
                                                    * self._factor(variation.yaw_torque_factor, count))
        airframe.drag_coefficient[indices] = config.drag_n_per_mps * scaling.drag_area_scale(nominal_mass)
        airframe.rotational_damping[indices] = scaling.REFERENCE_ROTATIONAL_DAMPING * scaling.inertia_scale(nominal_mass)
        if config.start_altitude_range_m is not None and self.randomize:
            airframe.start_altitude[indices] = self.rng.uniform(*config.start_altitude_range_m, count)
        else:
            airframe.start_altitude[indices] = config.start_altitude_m
        if self.randomize:
            airframe.rest_voltage[indices] = self.rng.uniform(*variation.battery_rest_voltage, count)
            wind_direction = self.rng.uniform(-np.pi, np.pi, count)
            wind_speed = self.rng.uniform(0, variation.max_wind_speed, count)
            airframe.wind[indices] = np.stack([wind_speed * np.cos(wind_direction), wind_speed * np.sin(wind_direction),
                                               np.zeros(count)], 1)
        else:
            airframe.rest_voltage[indices] = config.nominal_battery_voltage
            airframe.wind[indices] = 0

    def sample_start_state(self, state, airframe, indices, disturbed):
        disturbance, count = self.disturbance, len(indices)
        time_scale = np.sqrt(airframe.length_scale[indices])
        state.battery_voltage[indices] = airframe.rest_voltage[indices]
        state.position[indices] = 0
        if disturbed:
            state.velocity[indices] = self.rng.uniform(-disturbance.speed, disturbance.speed, (count, 3)) * time_scale[:, None]
            tilt = disturbance.roll_pitch_rad
            state.attitude[indices] = quaternion_from_euler(self.rng.uniform(-tilt, tilt, count),
                                                            self.rng.uniform(-tilt, tilt, count),
                                                            self.rng.uniform(-np.pi, np.pi, count))
            state.body_rates[indices] = (self.rng.uniform(-disturbance.body_rate, disturbance.body_rate, (count, 3))
                                         / time_scale[:, None])
        else:
            state.velocity[indices] = 0
            state.attitude[indices] = LEVEL_ATTITUDE
            state.body_rates[indices] = 0
        hover_throttle = np.sqrt(airframe.mass[indices] * GRAVITY_MPS2
                                 / (self.layout.motor_count * airframe.max_motor_thrust[indices]))
        if disturbed:
            spread = self.rng.uniform(*disturbance.motor_throttle_factor, (count, self.layout.motor_count))
        else:
            spread = 1.0
        state.motor_throttle[indices] = hover_throttle[:, None] * spread
        state.specific_force_body[indices] = RESTING_SPECIFIC_FORCE
