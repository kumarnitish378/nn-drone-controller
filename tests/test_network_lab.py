"""The network lab (web/network_core.js) must build the same inputs as the simulator and give the same
outputs as the Python network. Needs node."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from nn_drone.export.web_page import browser_network
from nn_drone.geometry import quaternion_from_euler, rotate_world_to_body, wrap_angle
from nn_drone.paths import WEB_DIR, default_weights
from nn_drone.policy import NeuralPilot

ERROR_CLIP_M = 2.5
TOLERANCE = 1e-5
JS_RUNNER = r"""
const fs = require('fs'); const lab = require(process.argv[2]); const c = JSON.parse(fs.readFileSync(process.argv[3]));
const network = lab.prepare(c.network);
const results = c.states.map((state) => {
  const built = lab.buildInputs(state, null);
  const activations = lab.forward(network, built.inputs);
  return { inputs: built.inputs, outputs: activations[activations.length - 1] };
});
console.log(JSON.stringify(results));
"""


def random_case(rng, pilot, mass_kg):
    env = pilot.make_environment(1, randomize=False, seed=0, noise=False, nominal_mass_kg=mass_kg,
                                 position_error_clip_m=ERROR_CLIP_M)
    roll, pitch, yaw = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-np.pi, np.pi)
    state = env.state
    state.attitude[0] = quaternion_from_euler(np.array([roll]), np.array([pitch]), np.array([yaw]))[0]
    state.position[0], state.velocity[0] = rng.uniform(-5, 5, 3), rng.uniform(-3, 3, 3)
    state.body_rates[0], state.specific_force_body[0] = rng.uniform(-2, 2, 3), rng.uniform(-5, 15, 3)
    state.battery_voltage[0] = rng.uniform(14.5, 16.8)
    target, target_heading = rng.uniform(-8, 8, 3), rng.uniform(-np.pi, np.pi)
    env.set_target(target, target_heading)
    env.previous_action[0] = rng.uniform(-0.5, 0.8, env.motor_count)
    env.integrator.value[0] = rng.uniform(-3, 3, 3)
    env.history.mark_reset(np.arange(1))
    observation = env.observe()[0]
    attitude = state.attitude[:1]
    body = lambda vector: rotate_world_to_body(attitude, np.asarray(vector, float)[None])[0].tolist()
    lab_state = dict(massKg=mass_kg, batteryV=float(state.battery_voltage[0]), errorClip=ERROR_CLIP_M,
                     rollDeg=float(np.degrees(roll)), pitchDeg=float(np.degrees(pitch)),
                     headingErrorDeg=float(np.degrees(wrap_angle(target_heading - yaw))),
                     gyroDegS=np.degrees(state.body_rates[0]).tolist(), accel=state.specific_force_body[0].tolist(),
                     target=body(target - state.position[0]), velocity=body(state.velocity[0]),
                     integral=body(env.integrator.value[0]), previous=env.previous_action[0].tolist())
    return lab_state, observation, pilot.command(observation[None])[0]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("frame", ["quad_x", "hex_x"])
def test_lab_inputs_and_outputs_match_python(frame, tmp_path):
    pilot = NeuralPilot.from_checkpoint(default_weights(frame))
    rng = np.random.default_rng(4)
    cases = [random_case(rng, pilot, mass_kg) for mass_kg in (1.0, 4.0, 95.0, 1.0, 20.0)]
    case_file = tmp_path / "case.json"
    case_file.write_text(json.dumps(dict(network=browser_network(default_weights(frame)), states=[case[0] for case in cases])))
    runner = tmp_path / "run.js"
    runner.write_text(JS_RUNNER)
    output = subprocess.run(["node", str(runner), str(WEB_DIR / "network_core.js"), str(case_file)],
                            check=True, capture_output=True, text=True)
    for (_, observation, action), result in zip(cases, json.loads(output.stdout)):
        np.testing.assert_allclose(result["inputs"], observation, atol=TOLERANCE)
        np.testing.assert_allclose(result["outputs"], action, atol=TOLERANCE)
