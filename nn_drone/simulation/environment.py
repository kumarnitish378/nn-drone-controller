"""Batched multirotor environment: many independent drones stepped together."""
from dataclasses import dataclass

import numpy as np

from nn_drone.geometry import heading_of, tilt_radians, upright_cosine, wrap_angle
from nn_drone.simulation.config import CURRENT_PROFILE, default_config_for
from nn_drone.simulation.frames import layout_named
from nn_drone.simulation.integrator import PositionErrorIntegrator
from nn_drone.simulation.observation import ObservationBuilder, ObservationHistory
from nn_drone.simulation.physics import RigidBodyPhysics
from nn_drone.simulation.randomization import AirframeSampler
from nn_drone.simulation.reward import CRASH_REWARD, REWARD_SCALE, flight_reward, has_crashed
from nn_drone.simulation.state import Airframe, VehicleState
from nn_drone.simulation.targets import TargetTrack
from nn_drone.units import action_to_pwm, clip_action, pwm_to_throttle

DEFAULT_EPISODE_STEPS = 1000
MOTORS_STOPPED_ACTION = -1.0


@dataclass
class StepResult:
    observation: np.ndarray
    reward: np.ndarray
    crashed: np.ndarray
    timed_out: np.ndarray
    final_observation: np.ndarray
    distance_to_target: np.ndarray
    pwm: np.ndarray


class MultirotorEnv:
    """`step(..., training=True)` moves the targets and resets crashed or timed-out drones (PPO training).
    With `training=False` the caller drives the targets and nothing is reset (flight tests, ground link)."""

    def __init__(self, count=1, frame="quad_x", profile=CURRENT_PROFILE, config=None, randomize=True, noise=True,
                 episode_steps=DEFAULT_EPISODE_STEPS, seed=0):
        self.count = count
        self.layout = layout_named(frame)
        self.profile = profile
        self.config = config or default_config_for(profile)
        self.episode_steps = episode_steps
        self.rng = np.random.default_rng(seed)
        motor_count = self.layout.motor_count
        self.observation_size = profile.input_size(motor_count)
        self.action_size = motor_count

        self.state = VehicleState.empty(count, motor_count, self.config.nominal_battery_voltage)
        self.airframe = Airframe.empty(count, self.config.control_period_s)
        self.targets = TargetTrack(count, self.rng, profile.moving_targets, self.config.moving_target_share)
        self.integrator = PositionErrorIntegrator(count, profile.integrator, self.config.control_period_s)
        self.history = ObservationHistory(count, motor_count, profile.history_frames)
        self.sampler = AirframeSampler(self.config, self.layout, self.rng, randomize)
        self.physics = RigidBodyPhysics(self.layout, self.config, profile.air_density_effect)
        self.sensors = ObservationBuilder(profile, motor_count, self.config.position_error_clip_m, self.rng, noise)
        self.previous_action = np.zeros((count, motor_count))
        self.step_count = np.zeros(count, dtype=np.int64)
        self.elapsed_time = np.zeros(count)
        self.reset(np.arange(count))

    @property
    def motor_count(self):
        return self.layout.motor_count

    @property
    def control_period(self):
        """Seconds per network step of the first drone (all drones share it when the mass is not randomised)."""
        return float(self.airframe.control_period[0])

    def reset(self, indices, disturbed=True):
        if len(indices) == 0:
            return
        self.sampler.sample_airframe(self.airframe, indices)
        self.sampler.sample_start_state(self.state, self.airframe, indices, disturbed)
        self.previous_action[indices] = 2 * self.state.motor_throttle[indices] - 1
        self.step_count[indices] = 0
        self.elapsed_time[indices] = 0
        self.integrator.reset(indices)
        self.history.mark_reset(indices)
        self._new_targets(indices)

    def reset_calm(self):
        """Every drone level, still and at hover throttle: the start of a flight test."""
        self.reset(np.arange(self.count), disturbed=False)

    def park_on_ground(self):
        """Motors stopped, sitting on the ground."""
        self.state.motor_throttle[:] = 0.0
        self.previous_action[:] = MOTORS_STOPPED_ACTION
        self.state.on_ground[:] = True

    def set_target(self, position, heading):
        self.targets.position[:] = position
        self.targets.heading[:] = heading

    def observe(self):
        return self._observation(remember=True)

    def step(self, action, training=True):
        action = clip_action(action)
        self.history.advance()
        if training and self.profile.moving_targets:
            self.targets.move(self.airframe.control_period)
        pwm = action_to_pwm(action)
        throttle_command = pwm_to_throttle(pwm)
        substep_period = self.airframe.control_period / self.config.physics_substeps
        for _ in range(self.config.physics_substeps):
            self.physics.advance(self.state, self.airframe, throttle_command, substep_period)
        self.step_count += 1
        self.elapsed_time += self.airframe.control_period

        reward, crashed, distance_m = self._score(action)
        timed_out = (self.step_count >= self.episode_steps) & ~crashed
        self.previous_action = action.copy()

        final_observation = None
        if training:
            self._new_targets(self.targets.pick_switching(~crashed & ~timed_out))
            if timed_out.any():
                final_observation = self._observation(remember=False)
            self.reset(np.nonzero(crashed | timed_out)[0])
        return StepResult(observation=self.observe(), reward=(reward * REWARD_SCALE).astype(np.float32), crashed=crashed,
                          timed_out=timed_out, final_observation=final_observation, distance_to_target=distance_m, pwm=pwm)

    def _score(self, action):
        state, airframe = self.state, self.airframe
        length = airframe.length_scale
        time_scale = np.sqrt(length)
        position_error = self.targets.position - state.position
        distance_m = np.linalg.norm(position_error, axis=1)
        distance = distance_m / length
        heading_error = np.abs(wrap_angle(self.targets.heading - heading_of(state.attitude)))
        uprightness = upright_cosine(state.attitude)
        action_change = action - self.previous_action
        if self.profile.has_integrator_input:
            self.integrator.update(position_error, state.velocity, self.targets.position, length, airframe.control_period)
        speed = np.linalg.norm(state.velocity, axis=1) / time_scale
        body_rate = np.linalg.norm(state.body_rates, axis=1) * time_scale
        reward = flight_reward(self.config.reward, distance, heading_error, body_rate, action_change, speed,
                               tilt_radians(state.attitude))
        crashed = has_crashed(uprightness, distance, speed)
        return np.where(crashed, CRASH_REWARD, reward), crashed, distance_m

    def _new_targets(self, indices):
        self.targets.sample_new(indices, self.state.position, self.airframe.length_scale)
        if self.profile.moving_targets:
            self.integrator.set_reference_target(indices, self.targets.position)

    def _observation(self, remember):
        frame = self.sensors.sense(self.state, self.airframe, self.targets.position, self.targets.heading,
                                   self.previous_action, self.integrator.value)
        if remember:
            self.history.remember(frame)
        parts = [frame.features, *self.history.frames]
        if self.profile.has_mass_class_input:
            parts.append(self.sensors.mass_class(self.airframe))
        return np.concatenate(parts, axis=1) if len(parts) > 1 else frame.features
