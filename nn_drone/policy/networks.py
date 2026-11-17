"""PyTorch networks: the flight policy (Gaussian with a tanh-squashed mean) and the value function."""
import math

import torch
from torch import nn

POLICY_HIDDEN_LAYERS = (64, 64)
VALUE_HIDDEN_LAYERS = (128, 128)
POLICY_OUTPUT_GAIN = 0.1
INITIAL_LOG_STD = -0.7


class TanhMlp(nn.Module):
    """Tanh hidden layers; optionally tanh on the output. Weights start at N(0, 1/fan_in), biases at zero."""

    def __init__(self, layer_sizes, squash_output, output_gain=1.0, generator=None):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(fan_in, fan_out) for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]))
        self.squash_output = squash_output
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                gain = output_gain if index == len(self.layers) - 1 else 1.0
                std = math.sqrt(1.0 / layer.in_features) * gain
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * std)
                layer.bias.zero_()

    @property
    def input_size(self):
        return self.layers[0].in_features

    @property
    def output_size(self):
        return self.layers[-1].out_features

    def forward(self, inputs):
        hidden = inputs
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < last or self.squash_output:
                hidden = torch.tanh(hidden)
        return hidden


class GaussianPolicy(nn.Module):
    def __init__(self, observation_size, action_size, hidden_layers=POLICY_HIDDEN_LAYERS, generator=None):
        super().__init__()
        self.mean_network = TanhMlp([observation_size, *hidden_layers, action_size], squash_output=True,
                                    output_gain=POLICY_OUTPUT_GAIN, generator=generator)
        self.log_std = nn.Parameter(torch.full((action_size,), INITIAL_LOG_STD))

    def forward(self, observation):
        return self.mean_network(observation)

    def log_probability(self, action, mean):
        standardized = (action - mean) / torch.exp(self.log_std)
        return (-0.5 * standardized ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def sample(self, observation, generator=None):
        mean = self(observation)
        noise = torch.randn(mean.shape, generator=generator, device=mean.device)
        action = mean + torch.exp(self.log_std) * noise
        return action, self.log_probability(action, mean)


class ValueNetwork(nn.Module):
    def __init__(self, observation_size, hidden_layers=VALUE_HIDDEN_LAYERS, generator=None):
        super().__init__()
        self.network = TanhMlp([observation_size, *hidden_layers, 1], squash_output=False, generator=generator)

    def forward(self, observation):
        return self.network(observation)[..., 0]
