"""Proximal Policy Optimisation with optional action-smoothness regularisers (CAPS, Mysore et al. 2021):
the network's output should change little between consecutive steps (temporal) and under small input
noise (spatial)."""
from dataclasses import dataclass

import torch

from nn_drone.policy.networks import GaussianPolicy, ValueNetwork

LOG_RATIO_LIMIT = 20.0
ADVANTAGE_EPSILON = 1e-8


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_weight: float = 0.0
    epochs: int = 5
    minibatch_size: int = 8192
    max_gradient_norm: float = 0.5
    min_log_std: float = -3.0
    max_log_std: float = 0.0
    temporal_smoothness: float = 0.0
    spatial_smoothness: float = 0.0
    spatial_noise_std: float = 0.01


@dataclass
class TrainingBatch:
    observation: torch.Tensor
    action: torch.Tensor
    log_probability: torch.Tensor
    advantage: torch.Tensor
    value_target: torch.Tensor
    next_observation: torch.Tensor
    continues: torch.Tensor

    def __len__(self):
        return len(self.observation)

    def subset(self, indices):
        return TrainingBatch(*(getattr(self, name)[indices] for name in self.__dataclass_fields__))


class PPOAgent:
    def __init__(self, observation_size, action_size, settings=PPOSettings(), device="cpu", seed=0):
        self.settings = settings
        self.device = torch.device(device)
        initializer = torch.Generator().manual_seed(seed)
        self.policy = GaussianPolicy(observation_size, action_size, generator=initializer).to(self.device)
        self.value_network = ValueNetwork(observation_size, generator=initializer).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=settings.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=settings.learning_rate)
        self.sampling_generator = torch.Generator(device=self.device).manual_seed(seed)

    def set_policy_learning_rate(self, learning_rate):
        for group in self.policy_optimizer.param_groups:
            group["lr"] = learning_rate

    def clamp_log_std(self):
        with torch.no_grad():
            self.policy.log_std.clamp_(self.settings.min_log_std, self.settings.max_log_std)

    @torch.no_grad()
    def act(self, observation):
        return self.policy.sample(observation, generator=self.sampling_generator)

    @torch.no_grad()
    def value(self, observation):
        return self.value_network(observation)

    def update(self, batch):
        settings = self.settings
        advantage = batch.advantage
        batch.advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + ADVANTAGE_EPSILON)
        for _ in range(settings.epochs):
            order = torch.randperm(len(batch), generator=self.sampling_generator, device=self.device)
            for start in range(0, len(batch), settings.minibatch_size):
                minibatch = batch.subset(order[start:start + settings.minibatch_size])
                self._update_policy(minibatch)
                self._update_value(minibatch)

    def _update_policy(self, minibatch):
        settings, policy = self.settings, self.policy
        mean = policy(minibatch.observation)
        log_probability = policy.log_probability(minibatch.action, mean)
        ratio = torch.exp(torch.clamp(log_probability - minibatch.log_probability, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT))
        clipped_ratio = torch.clamp(ratio, 1 - settings.clip_range, 1 + settings.clip_range)
        surrogate = torch.minimum(ratio * minibatch.advantage, clipped_ratio * minibatch.advantage)
        loss = -surrogate.mean() - settings.entropy_weight * policy.log_std.sum()
        if settings.temporal_smoothness > 0:
            next_mean = policy(minibatch.next_observation)
            step_change = ((next_mean - mean.detach()) ** 2 * minibatch.continues[:, None]).sum(1)
            loss = loss + settings.temporal_smoothness * step_change.mean()
        if settings.spatial_smoothness > 0:
            noise = torch.randn(minibatch.observation.shape, generator=self.sampling_generator, device=self.device)
            perturbed_mean = policy(minibatch.observation + settings.spatial_noise_std * noise)
            loss = loss + settings.spatial_smoothness * ((perturbed_mean - mean.detach()) ** 2).sum(1).mean()
        self._optimizer_step(self.policy_optimizer, policy.parameters(), loss)
        self.clamp_log_std()

    def _update_value(self, minibatch):
        value = self.value_network(minibatch.observation)
        loss = 0.5 * ((value - minibatch.value_target) ** 2).mean()
        self._optimizer_step(self.value_optimizer, self.value_network.parameters(), loss)

    def _optimizer_step(self, optimizer, parameters, loss):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters), self.settings.max_gradient_norm)
        optimizer.step()
