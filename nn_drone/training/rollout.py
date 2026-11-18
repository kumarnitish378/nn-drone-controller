"""Experience collection on the batched simulator and generalised advantage estimation (GAE).

The discount and GAE factors are per reference step of size-normalised time; a drone stepped more often
discounts each of its steps by the matching fraction, so the horizon in seconds stays the same."""
from dataclasses import dataclass

import numpy as np
import torch

from nn_drone.training.ppo import TrainingBatch


@dataclass
class RolloutStatistics:
    mean_reward: float
    crashes: int
    mean_distance_m: float


class RolloutCollector:
    def __init__(self, env, agent, steps_per_rollout):
        self.env = env
        self.agent = agent
        self.steps = steps_per_rollout
        self.device = agent.device
        self.observation = self._to_device(env.observe())

    def _to_device(self, array):
        return torch.from_numpy(np.ascontiguousarray(array, np.float32)).to(self.device)

    def collect(self):
        env, agent, steps, count = self.env, self.agent, self.steps, self.env.count
        buffer = lambda *shape: torch.zeros((steps, count, *shape), device=self.device)
        observations, actions = buffer(env.observation_size), buffer(env.action_size)
        log_probabilities, rewards, values = buffer(), buffer(), torch.zeros((steps + 1, count), device=self.device)
        step_lengths = buffer()
        episode_ended = torch.zeros((steps, count), dtype=torch.bool, device=self.device)
        crashes = np.zeros((), np.int64)
        distance_sum = 0.0
        for step in range(steps):
            action, log_probability = agent.act(self.observation)
            values[step] = agent.value(self.observation)
            observations[step], actions[step], log_probabilities[step] = self.observation, action, log_probability
            result = env.step(action.cpu().numpy())
            reward = self._to_device(result.reward)
            step_lengths[step] = self._to_device(result.normalised_step_length)
            if result.final_observation is not None:
                time_limited = self._to_device(result.timed_out)
                discount = agent.settings.discount ** step_lengths[step]
                reward = reward + discount * agent.value(self._to_device(result.final_observation)) * time_limited
            rewards[step] = reward
            episode_ended[step] = torch.from_numpy(result.crashed | result.timed_out).to(self.device)
            crashes += result.crashed.sum()
            distance_sum += float(result.distance_to_target.mean())
            self.observation = self._to_device(result.observation)
        values[steps] = agent.value(self.observation)
        advantage = self._advantages(rewards, values, episode_ended, step_lengths)
        next_observations = torch.cat([observations[1:], self.observation[None]], 0)
        flat = lambda tensor: tensor.reshape(steps * count, *tensor.shape[2:])
        batch = TrainingBatch(observation=flat(observations), action=flat(actions), log_probability=flat(log_probabilities),
                              advantage=flat(advantage), value_target=flat(advantage + values[:-1]),
                              next_observation=flat(next_observations), continues=flat((~episode_ended).float()))
        statistics = RolloutStatistics(mean_reward=float(rewards.mean()), crashes=int(crashes),
                                       mean_distance_m=distance_sum / steps)
        return batch, statistics

    def _advantages(self, rewards, values, episode_ended, step_lengths):
        settings = self.agent.settings
        discounts = settings.discount ** step_lengths
        discounts_with_lambda = (settings.discount * settings.gae_lambda) ** step_lengths
        advantage = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        for step in reversed(range(self.steps)):
            continues = 1.0 - episode_ended[step].float()
            temporal_difference = rewards[step] + discounts[step] * values[step + 1] * continues - values[step]
            running = temporal_difference + discounts_with_lambda[step] * continues * running
            advantage[step] = running
        return advantage
