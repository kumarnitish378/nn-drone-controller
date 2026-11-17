"""The simulator must reproduce, step for step, the four simulator generations the shipped networks were
trained in. Reference data was recorded from the original implementation (tests/golden)."""
import json
from pathlib import Path

import numpy as np
import pytest

from nn_drone.simulation import MultirotorEnv, default_config_for, profile_named

GOLDEN = Path(__file__).parent / "golden"
CASES = json.loads((GOLDEN / "env_cases.json").read_text())
LEGACY_CONFIG_NAMES = {"mass": "nominal_mass_kg", "mass_range": "nominal_mass_range_kg", "alt_range": "start_altitude_range_m",
                       "ground": "solid_ground", "alt0": "start_altitude_m", "err_clip": "position_error_clip_m"}
DRONES = 8
TRAINING_STEPS = 260
EPISODE_STEPS = 120
FIXED_DRONES = 3
FIXED_STEPS = 200
TOLERANCE = 1e-9


def build_env(case, count, seed, **options):
    profile = profile_named(case["version"])
    overrides = {LEGACY_CONFIG_NAMES[name]: value for name, value in case["overrides"].items()}
    return MultirotorEnv(count, frame=case["frame"], profile=profile, config=default_config_for(profile, **overrides),
                         seed=seed, **options)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_training_steps_match_reference(case_name):
    case, reference = CASES[case_name], np.load(GOLDEN / f"env_{case_name}.npz")
    case_index = int(case_name.removeprefix("case"))
    env = build_env(case, DRONES, seed=11 + case_index, episode_steps=EPISODE_STEPS)
    action_rng = np.random.default_rng(100 + case_index)
    np.testing.assert_allclose(env.observe(), reference["obs"][0], atol=TOLERANCE)
    for step in range(TRAINING_STEPS):
        action = np.tanh(action_rng.normal(0.2, 0.6, (DRONES, env.action_size))).astype(np.float32)
        result = env.step(action)
        np.testing.assert_allclose(result.observation, reference["obs"][step + 1], atol=1e-5, err_msg=f"step {step}")
        np.testing.assert_allclose(result.reward, reference["reward"][step], atol=1e-6)
        np.testing.assert_array_equal(result.crashed, reference["crashed"][step])
        np.testing.assert_array_equal(result.timed_out, reference["timeout"][step])
        assert (result.final_observation is not None) == reference["final_obs_present"][step]
        np.testing.assert_allclose(result.distance_to_target, reference["dist"][step], atol=TOLERANCE)
        np.testing.assert_allclose(result.pwm, reference["pwm"][step], atol=TOLERANCE)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_flight_with_fixed_target_matches_reference(case_name):
    case, reference = CASES[case_name], np.load(GOLDEN / f"fixed_{case_name}.npy")
    env = build_env(case, FIXED_DRONES, seed=5, randomize=False, noise=False)
    env.reset_calm()
    for step in range(FIXED_STEPS):
        env.set_target([1.0, -0.5, 1.5], 0.3)
        action = np.full((FIXED_DRONES, env.action_size), 0.18, np.float32)
        action[1] += 0.05
        action[2, 0] -= 0.05
        env.step(action, training=False)
        state = env.state
        flat = np.concatenate([state.position.ravel(), state.velocity.ravel(), state.attitude.ravel(), state.body_rates.ravel()])
        np.testing.assert_allclose(flat, reference[step], atol=TOLERANCE, err_msg=f"step {step}")
