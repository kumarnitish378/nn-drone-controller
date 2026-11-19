"""The C++ flight library (ardupilot/libraries/AP_NNControl) must fly like the Python network: a recorded Python
flight (ground-link vehicle, sensor noise off) is replayed through NN_Pilot and the inputs and motor commands
compared at every 400 Hz step. Needs a C++ compiler (g++)."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.export.cpp_library import CPP_LIBRARY_DIR, CPP_MODELS_DIR, model_header
from nn_drone.geometry import rotate_body_to_world
from nn_drone.ground_link.geodesy import HomePoint
from nn_drone.ground_link.mission_items import MissionItem
from nn_drone.ground_link.vehicle import CopterMode, SimulatedVehicle
from nn_drone.paths import DEFAULT_WEIGHTS, REPOSITORY_ROOT

HOME = HomePoint(28.5355, 77.3910, 0.0)
RELATIVE = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
RECORDED_STEPS = 6000                     # 15 s at 400 Hz: idle, spool-up, take-off and the start of a waypoint leg
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])   # also North-West-Up to North-East-Down
INPUT_TOLERANCE = 1e-5                     # measured: 7.5e-7 (float32 vs Python float64 sensing)
COMMAND_TOLERANCE = {"exact": 1e-5, "fast": 1e-5, "cordic": 1e-5}   # measured: 2-6e-7
FAST_TANH_TOLERANCE = 1e-5
CORDIC_TANH_TOLERANCE = 1e-6              # emulated peripheral (exact cosh/sinh, q1.31); real one at 24 iterations ~1e-6
TEST_SOURCES = REPOSITORY_ROOT / "ardupilot" / "tests"
COMPILER = shutil.which("g++")
needs_compiler = pytest.mark.skipif(COMPILER is None, reason="needs g++")


def record_flight(frame, mass_kg):
    """Python flight with the state in ArduPilot conventions (NED, FRD) at every control step."""
    vehicle = SimulatedVehicle(frame, DEFAULT_WEIGHTS[frame], HOME, mass_kg, log=lambda *_: None)
    vehicle.set_parameter("SIM_NOISE", 0.0)

    def item(command, north=0.0, east=0.0, height=0.0):
        latitude, longitude = vehicle.home.to_global([north, -east, 0]) if (north or east) else (0, 0)
        return MissionItem(seq=0, frame=RELATIVE, command=command, param1=0, param3=0, x=latitude, y=longitude, z=height)
    vehicle.mission = [item(mavlink.MAV_CMD_NAV_WAYPOINT), item(mavlink.MAV_CMD_NAV_TAKEOFF, height=10),
                       item(mavlink.MAV_CMD_NAV_WAYPOINT, 30, 20, 15)]
    vehicle.mode = CopterMode.AUTO
    vehicle.arm()
    vehicle.start_mission()

    env, steps = vehicle.env, []
    network_command = vehicle.pilot.command

    def recording_command(observation):
        state, targets = env.state, env.targets
        body_to_world = rotate_body_to_world(np.repeat(state.attitude[:1], 3, axis=0), np.eye(3)).T
        commands = network_command(observation)
        steps.append(dict(
            attitude=(FLU_TO_FRD @ body_to_world @ FLU_TO_FRD).reshape(-1),
            state=np.concatenate([FLU_TO_FRD @ state.body_rates[0], FLU_TO_FRD @ state.specific_force_body[0],
                                  FLU_TO_FRD @ state.position[0], FLU_TO_FRD @ state.velocity[0], state.battery_voltage[:1]]),
            target=np.concatenate([FLU_TO_FRD @ targets.position[0], -targets.heading[:1]]),
            commands=commands[0].astype(np.float64), inputs=observation[0].astype(np.float64)))
        return commands

    vehicle.pilot.command = recording_command
    initial_command = float(env.previous_action[0, 0])
    for _ in range(RECORDED_STEPS):
        vehicle.step()
        steps[-1]["applied"] = env.previous_action[0].astype(np.float64)
    assert not vehicle.crashed and vehicle.position[2] > 5, "the recorded Python flight must take off"
    header = [env.motor_count, len(steps), steps[0]["inputs"].size, mass_kg, float(vehicle.parameters["NN_ERR_CLIP"]),
              initial_command]
    records = [np.concatenate([step["attitude"], step["state"], step["target"], step["commands"], step["applied"], step["inputs"]])
               for step in steps]
    return np.concatenate([np.asarray(header, np.float64), *records]).astype("<f8")


@pytest.fixture(scope="module")
def host_test(tmp_path_factory):
    if COMPILER is None:
        pytest.skip("needs g++")
    executable = tmp_path_factory.mktemp("cpp") / "nn_host_test"
    sources = [TEST_SOURCES / "nn_host_test.cpp", *(CPP_LIBRARY_DIR / name for name in ("NN_Network.cpp", "NN_Pilot.cpp", "NN_Stm32Cordic.cpp"))]
    subprocess.run([COMPILER, "-std=gnu++11", "-O2", "-Wall", "-Wextra", "-Werror", "-DNN_USE_STM32_CORDIC=1",
                    '-DNN_CORDIC_DEVICE_HEADER="fake_stm32_cordic.h"', f"-I{CPP_LIBRARY_DIR}", f"-I{TEST_SOURCES}",
                    *map(str, sources), "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


def run_host_test(executable, *arguments):
    output = subprocess.run([str(executable), *arguments], check=True, capture_output=True, text=True)
    return json.loads(output.stdout)


@pytest.mark.parametrize("frame, mass_kg", [("quad_x", 4.0), ("hex_x", 95.0), ("quad_x", 1.0)])
def test_cpp_flies_like_python(host_test, tmp_path, frame, mass_kg):
    recording = tmp_path / "flight.bin"
    record_flight(frame, mass_kg).tofile(recording)
    for activation, tolerance in COMMAND_TOLERANCE.items():
        result = run_host_test(host_test, "replay", str(recording), activation)
        assert result["steps"] == RECORDED_STEPS
        assert result["largest_input_error"] < INPUT_TOLERANCE, (activation, result)
        assert result["largest_command_error"] < tolerance, (activation, result)


def test_tanh_backends_are_accurate(host_test):
    result = run_host_test(host_test, "tanh")
    assert result["fast_tanh_error"] < FAST_TANH_TOLERANCE, result
    assert result["cordic_tanh_error"] < CORDIC_TANH_TOLERANCE, result


@pytest.mark.parametrize("frame", list(DEFAULT_WEIGHTS))
def test_model_headers_match_shipped_networks(frame):
    committed = (CPP_MODELS_DIR / f"NN_Model_{frame}.h").read_text(encoding="utf-8")
    assert committed == model_header(frame, DEFAULT_WEIGHTS[frame]), "run: python -m nn_drone export-cpp"
