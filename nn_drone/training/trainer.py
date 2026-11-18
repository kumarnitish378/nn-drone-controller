"""PPO training run: collect, update, log, checkpoint, until the time or iteration budget is spent."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nn_drone.policy.checkpoint import inspect_checkpoint, load_policy_weights, save_checkpoint
from nn_drone.simulation import CURRENT_PROFILE, MultirotorEnv, default_config_for, profile_named
from nn_drone.training.ppo import PPOAgent, PPOSettings
from nn_drone.training.rollout import RolloutCollector

PROGRESS_EVERY_ITERATIONS = 10
CHECKPOINT_EVERY_ITERATIONS = 20
SNAPSHOT_EVERY_ITERATIONS = 30
SECONDS_PER_MINUTE = 60
REWARD_REPORT_SCALE = 10
CRASH_RATE_PER = 1000
SAMPLES_PER_MILLION = 1e6
REFERENCE_STEPS_PER_ROLLOUT = 128


@dataclass(frozen=True)
class TrainingPlan:
    frame: str = "quad_x"
    profile_name: str = CURRENT_PROFILE.name
    parallel_drones: int = 1024
    steps_per_rollout: int = None
    """Defaults to 128 reference steps of size-normalised time (512 steps at 400 Hz for a 1 kg drone)."""
    time_budget_minutes: float = 8.0
    max_iterations: int = 100000
    output_stem: str = "policy"
    resume_from: str = ""
    seed: int = 0
    critic_warmup_iterations: int = 0
    config_overrides: dict = field(default_factory=dict)
    ppo: PPOSettings = PPOSettings()
    device: str = "auto"


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a GPU")
    return requested


class Trainer:
    def __init__(self, plan, log=print):
        self.plan = plan
        self.log = log
        self.device = resolve_device(plan.device)
        self.profile = profile_named(plan.profile_name)
        self.env = MultirotorEnv(plan.parallel_drones, frame=plan.frame, profile=self.profile,
                                 config=default_config_for(self.profile, **plan.config_overrides), seed=plan.seed)
        self.agent = PPOAgent(self.env.observation_size, self.env.action_size, settings=plan.ppo, device=self.device,
                              seed=plan.seed)
        self.output = Path(plan.output_stem)
        self.history_path = self.output.with_name(self.output.name + "_log.json")
        self.history = []
        if plan.resume_from:
            self._resume(plan.resume_from)
        self.agent.clamp_log_std()

    def _resume(self, path):
        info = inspect_checkpoint(path)
        if info.motor_count != self.env.motor_count:
            raise ValueError(f"{path} drives {info.motor_count} motors, the {self.plan.frame} frame has {self.env.motor_count}")
        load_policy_weights(path, self.agent.policy, self.agent.value_network)
        if self.history_path.exists():
            self.history = json.loads(self.history_path.read_text())

    def run(self):
        plan = self.plan
        steps_per_rollout = plan.steps_per_rollout or self._default_rollout_steps()
        collector = RolloutCollector(self.env, self.agent, steps_per_rollout)
        samples_before = self.history[-1]["samples"] if self.history else 0
        samples_per_iteration = steps_per_rollout * plan.parallel_drones
        self.log(f"device: {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else "")
                 + f", profile {self.profile.name} at {self.profile.control_timing.rate_hz:g} Hz, {steps_per_rollout} steps per rollout")
        started = time.time()
        for iteration in range(plan.max_iterations):
            in_warmup = iteration < plan.critic_warmup_iterations
            self.agent.set_policy_learning_rate(0.0 if in_warmup else plan.ppo.learning_rate)
            batch, statistics = collector.collect()
            self.agent.update(batch)
            elapsed = time.time() - started
            record = dict(samples=samples_before + (iteration + 1) * samples_per_iteration,
                          reward=statistics.mean_reward * REWARD_REPORT_SCALE,
                          crash_per_1k=statistics.crashes / samples_per_iteration * CRASH_RATE_PER,
                          dist=statistics.mean_distance_m,
                          std=float(torch.exp(self.agent.policy.log_std.detach()).mean()))
            self.history.append(record)
            out_of_time = elapsed > plan.time_budget_minutes * SECONDS_PER_MINUTE
            if iteration % PROGRESS_EVERY_ITERATIONS == 0:
                self._report(iteration, elapsed, record)
            if iteration % CHECKPOINT_EVERY_ITERATIONS == 0 or out_of_time:
                self._save(self.output.with_suffix(".npz"))
                self.history_path.write_text(json.dumps(self.history))
            if iteration % SNAPSHOT_EVERY_ITERATIONS == 0 and not in_warmup:
                self._save(self.output.with_name(f"{self.output.name}_snap{iteration:03d}.npz"))
            if out_of_time:
                break
        self._save(self.output.with_suffix(".npz"))
        self.history_path.write_text(json.dumps(self.history))
        self.log("done")

    def _default_rollout_steps(self):
        typical_step_length = float(self.env.airframe.normalised_step_length.mean())
        return int(round(REFERENCE_STEPS_PER_ROLLOUT / typical_step_length))

    def _save(self, path):
        save_checkpoint(path, self.agent.policy, self.agent.value_network, self.profile.name)

    def _report(self, iteration, elapsed, record):
        self.log(f"it {iteration:4d} {elapsed:5.0f}s samples {record['samples'] / SAMPLES_PER_MILLION:6.2f}M "
                 f"rew {record['reward']:6.3f} crash/1k {record['crash_per_1k']:6.3f} dist {record['dist']:5.2f} "
                 f"std {record['std']:.3f}")
