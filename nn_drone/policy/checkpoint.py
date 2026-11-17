"""Weights files (.npz).

The format is shared with the C header export, the web tool and the CI model package:
`pi_W{n}` / `pi_b{n}` are the flight network's layers with weights stored [inputs][outputs],
`vf_W{n}` / `vf_b{n}` the value network, `logstd` the exploration noise, and `simulator_profile`
the simulator generation the network flies in (files without it fly in the current one). A file with
fewer inputs than its profile provides is padded with zero weights: the extra inputs are ignored.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nn_drone.simulation.config import CURRENT_PROFILE, profile_named
from nn_drone.simulation.frames import LAYOUT_BY_MOTOR_COUNT

POLICY_PREFIX = "pi"
VALUE_PREFIX = "vf"
LOG_STD_KEY = "logstd"
PROFILE_KEY = "simulator_profile"


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    input_size: int
    motor_count: int
    profile_name: str

    @property
    def frame(self):
        return LAYOUT_BY_MOTOR_COUNT[self.motor_count].name

    @property
    def profile(self):
        return profile_named(self.profile_name)


def layer_keys(prefix, index):
    return f"{prefix}_W{index}", f"{prefix}_b{index}"


def read_layers(arrays, prefix):
    """[(weights [in][out], bias), ...] for one network."""
    layers, index = [], 0
    while layer_keys(prefix, index)[0] in arrays:
        weight_key, bias_key = layer_keys(prefix, index)
        layers.append((np.asarray(arrays[weight_key], np.float32), np.asarray(arrays[bias_key], np.float32)))
        index += 1
    return layers


def inspect_checkpoint(path):
    with np.load(path) as arrays:
        layers = read_layers(arrays, POLICY_PREFIX)
        input_size, motor_count = layers[0][0].shape[0], layers[-1][0].shape[1]
        profile_name = str(arrays[PROFILE_KEY]) if PROFILE_KEY in arrays else CURRENT_PROFILE.name
    expected_inputs = profile_named(profile_name).input_size(motor_count)
    if input_size > expected_inputs:
        raise ValueError(f"{path}: {input_size} inputs, but simulator profile {profile_name} provides {expected_inputs}")
    return CheckpointInfo(Path(path), input_size, motor_count, profile_name)


def label_profile(path, profile_name):
    """Record the simulator generation in an existing weights file."""
    with np.load(path) as arrays:
        contents = {key: arrays[key] for key in arrays.files}
    contents[PROFILE_KEY] = np.array(profile_name)
    np.savez(path, **contents)


def load_network(network, layers):
    """Copy [in][out] layers into a TanhMlp. Inputs the file does not have start with zero weight (warm start)."""
    with torch.no_grad():
        for linear, (weight, bias) in zip(network.layers, layers):
            stored = torch.from_numpy(weight.T.copy())
            linear.weight.zero_()
            linear.weight[:, :stored.shape[1]] = stored
            linear.bias.copy_(torch.from_numpy(bias))


def network_layers(network):
    return [(linear.weight.detach().cpu().numpy().T.astype(np.float32), linear.bias.detach().cpu().numpy().astype(np.float32))
            for linear in network.layers]


def load_policy_weights(path, policy, value_network=None):
    with np.load(path) as arrays:
        load_network(policy.mean_network, read_layers(arrays, POLICY_PREFIX))
        if value_network is not None and layer_keys(VALUE_PREFIX, 0)[0] in arrays:
            load_network(value_network.network, read_layers(arrays, VALUE_PREFIX))
        with torch.no_grad():
            policy.log_std.copy_(torch.from_numpy(np.asarray(arrays[LOG_STD_KEY], np.float32)))


def save_checkpoint(path, policy, value_network, profile_name):
    arrays = {}
    for prefix, network in ((POLICY_PREFIX, policy.mean_network), (VALUE_PREFIX, value_network.network)):
        for index, (weight, bias) in enumerate(network_layers(network)):
            weight_key, bias_key = layer_keys(prefix, index)
            arrays[weight_key], arrays[bias_key] = weight, bias
    arrays[LOG_STD_KEY] = policy.log_std.detach().cpu().numpy().astype(np.float32)
    arrays[PROFILE_KEY] = np.array(profile_name)
    np.savez(path, **arrays)
