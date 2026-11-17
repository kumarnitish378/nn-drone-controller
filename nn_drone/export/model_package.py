"""Package the flight network (policy only, no critic) after the tests have passed: weights as .npz,
a C header with the weights as float arrays, and a model card with the test report."""
import json
from pathlib import Path

import numpy as np

from nn_drone.policy.checkpoint import POLICY_PREFIX, inspect_checkpoint, layer_keys, read_layers

C_VALUES_PER_LINE = 8


def flight_network_arrays(weights_path):
    """Policy layers padded to the inputs the simulator feeds (older weights: missing inputs get zero weight)."""
    info = inspect_checkpoint(weights_path)
    with np.load(weights_path) as stored:
        layers = read_layers(stored, POLICY_PREFIX)
    input_size = info.profile.input_size(info.motor_count)
    first_weight, first_bias = layers[0]
    padding = np.zeros((input_size - first_weight.shape[0], first_weight.shape[1]), np.float32)
    layers[0] = (np.concatenate([first_weight, padding]), first_bias)
    arrays = {}
    for index, (weight, bias) in enumerate(layers):
        weight_key, bias_key = layer_keys(POLICY_PREFIX, index)
        arrays[weight_key], arrays[bias_key] = weight, bias
    return info, arrays


def c_header(frame, arrays, layer_sizes, parameter_count):
    guard = f"{frame.upper()}_MODEL_H"
    lines = [f"/* {frame} flight network: {' -> '.join(map(str, layer_sizes))}, tanh, {parameter_count} float32 parameters.",
             "   y = tanh(tanh(tanh(x*W0 + b0)*W1 + b1)*W2 + b2);  PWM = 1500 + 500*y. Weights are row-major [in][out]. */",
             f"#ifndef {guard}", f"#define {guard}"]
    for key, values in arrays.items():
        dimensions = "".join(f"[{size}]" for size in values.shape)
        lines.append(f"static const float {frame}_{key}{dimensions} = {{")
        flat = values.reshape(-1)
        for start in range(0, len(flat), C_VALUES_PER_LINE):
            lines.append("  " + ", ".join(f"{value:.8e}f" for value in flat[start:start + C_VALUES_PER_LINE]) + ",")
        lines.append("};")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def export_model(weights_path, output_dir, test_report_path=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info, arrays = flight_network_arrays(weights_path)
    frame = info.frame
    np.savez(output_dir / f"{frame}_model.npz", **arrays)
    parameter_count = int(sum(values.size for values in arrays.values()))
    weights = [values for key, values in arrays.items() if "_W" in key]
    layer_sizes = [int(weights[0].shape[0])] + [int(weight.shape[1]) for weight in weights]
    (output_dir / f"{frame}_model.h").write_text(c_header(frame, arrays, layer_sizes, parameter_count))
    card = dict(frame=frame, source_weights=Path(weights_path).name, simulator_profile=info.profile_name, layers=layer_sizes,
                activation="tanh", parameters=parameter_count, size_bytes_float32=parameter_count * np.dtype(np.float32).itemsize,
                output="per motor, -1..1; PWM = 1500 + 500 * output",
                note="simulation-trained only; inputs must be built and scaled as in nn_drone.simulation.observation")
    if test_report_path and Path(test_report_path).exists():
        card["tests"] = json.loads(Path(test_report_path).read_text())
    (output_dir / f"{frame}_model_card.json").write_text(json.dumps(card, indent=1))
    return card
