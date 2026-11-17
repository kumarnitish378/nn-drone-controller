"""The PyTorch networks and PPO update must match the original NumPy implementation."""
from pathlib import Path

import numpy as np
import pytest
import torch

from nn_drone.paths import MODELS_DIR
from nn_drone.policy import NeuralPilot
from nn_drone.policy.checkpoint import load_network, network_layers, read_layers
from nn_drone.training.ppo import PPOAgent, PPOSettings, TrainingBatch

GOLDEN = Path(__file__).parent / "golden"
SHIPPED_WEIGHTS = {"quad_x_v4": "quad_x_policy_v4.npz", "hex_x_v1": "hex_x_policy_v1.npz",
                   "quad_v1": "legacy/quad_x_v1.npz", "quad_x_v2": "legacy/quad_x_v2.npz", "quad_x_v3": "legacy/quad_x_v3.npz"}
EXPECTED_PROFILE = {"quad_x_v4": "v4", "hex_x_v1": "v4", "quad_v1": "v1", "quad_x_v2": "v2", "quad_x_v3": "v3"}


@pytest.mark.parametrize("name", sorted(SHIPPED_WEIGHTS))
def test_pilot_matches_numpy_forward(name):
    reference = np.load(GOLDEN / "policy_forward.npz")
    pilot = NeuralPilot.from_checkpoint(MODELS_DIR / SHIPPED_WEIGHTS[name])
    assert pilot.profile.name == EXPECTED_PROFILE[name]
    observation = reference[f"{name}_obs"]
    padding = pilot.policy.mean_network.input_size - observation.shape[1]
    observation = np.pad(observation, ((0, 0), (0, padding)))
    np.testing.assert_allclose(pilot.command(observation), reference[f"{name}_action"], atol=2e-6)


def test_ppo_update_matches_numpy_reference():
    reference = np.load(GOLDEN / "ppo_update.npz")
    batch_size, observation_size = reference["batch_obs"].shape
    action_size = reference["batch_act"].shape[1]
    settings = PPOSettings(epochs=3, minibatch_size=batch_size, temporal_smoothness=0.5)
    agent = PPOAgent(observation_size, action_size, settings=settings)
    before = {key.removeprefix("before_"): reference[key] for key in reference.files if key.startswith("before_")}
    load_network(agent.policy.mean_network, read_layers(before, "pi"))
    load_network(agent.value_network.network, read_layers(before, "vf"))
    with torch.no_grad():
        agent.policy.log_std.copy_(torch.from_numpy(before["logstd"]))
    tensor = lambda key: torch.from_numpy(reference[f"batch_{key}"])
    agent.update(TrainingBatch(observation=tensor("obs"), action=tensor("act"), log_probability=tensor("logp_old"),
                               advantage=tensor("adv"), value_target=tensor("ret"), next_observation=tensor("obs_next"),
                               continues=tensor("cont")))
    for prefix, network in (("pi", agent.policy.mean_network), ("vf", agent.value_network.network)):
        for index, (weight, bias) in enumerate(network_layers(network)):
            np.testing.assert_allclose(weight, reference[f"after_{prefix}_W{index}"], atol=2e-6)
            np.testing.assert_allclose(bias, reference[f"after_{prefix}_b{index}"], atol=2e-6)
    np.testing.assert_allclose(agent.policy.log_std.detach().numpy(), reference["after_logstd"], atol=2e-6)
