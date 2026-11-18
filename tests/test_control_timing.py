"""Profile v5 runs the network in a fixed 400 Hz main loop; everything time-based must still count
size-normalised time, so it means the same as in the 100 Hz profiles."""
import numpy as np
import pytest

from nn_drone.simulation import MultirotorEnv, default_config_for, profile_named
from nn_drone.simulation.config import REFERENCE_CONTROL_PERIOD_S, history_frame_size
from nn_drone.simulation.scaling import time_scale

MAIN_LOOP_HZ = 400
HISTORY_FRAMES = 2
EPISODE_REFERENCE_STEPS = 50
HOVER_ACTION = 0.18


def main_loop_env(mass_kg=1.0, count=2, **options):
    profile = profile_named("v5")
    return MultirotorEnv(count, profile=profile, config=default_config_for(profile, nominal_mass_kg=mass_kg), seed=3,
                         **options)


@pytest.mark.parametrize("mass_kg", [1.0, 4.0, 95.0])
def test_every_size_runs_at_the_main_loop_rate(mass_kg):
    env = main_loop_env(mass_kg)
    assert env.control_period == pytest.approx(1 / MAIN_LOOP_HZ)
    reference_period = REFERENCE_CONTROL_PERIOD_S * time_scale(mass_kg)
    np.testing.assert_allclose(env.airframe.normalised_step_length, env.control_period / reference_period)


@pytest.mark.parametrize("mass_kg, spacing", [(1.0, 4), (4.0, 5), (95.0, 9)])
def test_history_frames_are_one_reference_step_apart(mass_kg, spacing):
    env = main_loop_env(mass_kg, count=1, randomize=False, noise=False)
    env.reset_calm()
    env.set_target([1.0, 0.5, 2.0], 0.3)
    frame_size = history_frame_size(env.motor_count)
    seen = []
    for _ in range(3 * spacing * HISTORY_FRAMES):
        observation = env.observe()
        seen.append(env.history.latest.history_features[0].copy())
        env.step(np.full((1, env.action_size), HOVER_ACTION, np.float32), training=False)
    observation = env.observe()
    history_start = env.observation_size - 1 - HISTORY_FRAMES * frame_size
    for age in range(1, HISTORY_FRAMES + 1):
        frame = observation[0, history_start + (age - 1) * frame_size: history_start + age * frame_size]
        np.testing.assert_array_equal(frame, seen[-age * spacing])


def test_episode_lasts_the_same_simulated_time():
    env = main_loop_env(episode_steps=EPISODE_REFERENCE_STEPS, randomize=False, noise=False)
    env.reset_calm()
    steps_per_reference_step = round(1 / env.airframe.normalised_step_length[0])
    action = np.full((env.count, env.action_size), HOVER_ACTION, np.float32)
    for _ in range(EPISODE_REFERENCE_STEPS * steps_per_reference_step - 1):
        result = env.step(action)
        assert not result.timed_out.any()
    assert env.step(action).timed_out.all()


def test_reward_per_second_does_not_depend_on_the_rate():
    """A calm hover collects the same reward per simulated second at 100 Hz and at 400 Hz."""
    rewards_per_second = []
    for profile_name in ("v4", "v5"):
        profile = profile_named(profile_name)
        env = MultirotorEnv(1, profile=profile, config=default_config_for(profile), randomize=False, noise=False, seed=1)
        env.reset_calm()
        env.set_target(env.state.position[0].copy(), 0.0)
        steps = round(1.0 / env.control_period)
        total = sum(float(env.step(np.full((1, env.action_size), HOVER_ACTION, np.float32), training=False).reward[0])
                    for _ in range(steps))
        rewards_per_second.append(total)
    assert rewards_per_second[1] == pytest.approx(rewards_per_second[0], rel=0.02)


def test_waypoints_jump_as_often_per_second_at_any_rate():
    from nn_drone.simulation.targets import TargetSampling, TargetTrack
    drones, seconds = 20000, 1.0
    switches_per_second = []
    for step_length, period in ((1.0, REFERENCE_CONTROL_PERIOD_S), (0.25, 1 / MAIN_LOOP_HZ)):
        track = TargetTrack(drones, np.random.default_rng(0), moving_targets=False, moving_share=0.0)
        eligible, lengths = np.ones(drones, bool), np.full(drones, step_length)
        steps = round(seconds / period)
        switches_per_second.append(sum(len(track.pick_switching(eligible, lengths)) for _ in range(steps)) / drones / seconds)
    expected = TargetSampling().waypoint_switch_probability / REFERENCE_CONTROL_PERIOD_S
    np.testing.assert_allclose(switches_per_second, expected, rtol=0.05)
