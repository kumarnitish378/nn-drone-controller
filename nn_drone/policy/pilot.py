"""A trained network flying a simulated drone: deterministic (mean) motor commands, on the CPU."""
import numpy as np
import torch

from nn_drone.policy.checkpoint import inspect_checkpoint, load_policy_weights
from nn_drone.policy.networks import GaussianPolicy
from nn_drone.simulation import MultirotorEnv, default_config_for


class NeuralPilot:
    def __init__(self, policy, checkpoint_info):
        self.policy = policy.eval()
        self.info = checkpoint_info

    @classmethod
    def from_checkpoint(cls, path):
        info = inspect_checkpoint(path)
        policy = GaussianPolicy(info.profile.input_size(info.motor_count), info.motor_count)
        load_policy_weights(path, policy)
        return cls(policy, info)

    @property
    def frame(self):
        return self.info.frame

    @property
    def profile(self):
        return self.info.profile

    def command(self, observation):
        with torch.inference_mode():
            return self.policy(torch.from_numpy(np.ascontiguousarray(observation, np.float32))).numpy()

    def make_environment(self, count, randomize, seed, noise=True, **config_overrides):
        """Environment in this network's simulator generation, every drone starting calm."""
        env = MultirotorEnv(count, frame=self.frame, profile=self.profile,
                            config=default_config_for(self.profile, **config_overrides),
                            randomize=randomize, noise=noise, seed=seed)
        env.reset_calm()
        return env

    def fly_step(self, env, training=False):
        return env.step(self.command(env.observe()), training=training)
