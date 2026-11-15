"""Package the flight network (policy only, no critic) after the tests have passed.
usage: python3 export_model.py <weights.npz> <frame> <out_dir>
writes <frame>_model.npz, <frame>_model.h (C arrays, float32) and <frame>_model_card.json"""
import os, sys, json
import numpy as np
weights, frame, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
d = np.load(weights)
keys = ["pi_W0", "pi_b0", "pi_W1", "pi_b1", "pi_W2", "pi_b2"]
net = {k: d[k].astype(np.float32) for k in keys}
n_mot = net["pi_W2"].shape[1]; n_in = 21 + n_mot + 2 * (14 + n_mot)      # inputs the simulator feeds
if net["pi_W0"].shape[0] < n_in:                                           # older weights: missing inputs have zero weight
    net["pi_W0"] = np.concatenate([net["pi_W0"], np.zeros((n_in - net["pi_W0"].shape[0], net["pi_W0"].shape[1]), np.float32)])
np.savez(os.path.join(out, f"{frame}_model.npz"), **net)
params = int(sum(v.size for v in net.values()))
layers = [int(net["pi_W0"].shape[0]), int(net["pi_W0"].shape[1]), int(net["pi_W1"].shape[1]), int(net["pi_W2"].shape[1])]
with open(os.path.join(out, f"{frame}_model.h"), "w") as f:
    g = f"{frame.upper()}_MODEL_H"
    f.write(f"/* {frame} flight network: {' -> '.join(map(str, layers))}, tanh, {params} float32 parameters.\n"
            f"   y = tanh(tanh(tanh(x*W0 + b0)*W1 + b1)*W2 + b2);  PWM = 1500 + 500*y. Weights are row-major [in][out]. */\n"
            f"#ifndef {g}\n#define {g}\n")
    for k, v in net.items():
        dims = "".join(f"[{n}]" for n in v.shape)
        f.write(f"static const float {frame}_{k}{dims} = {{\n")
        flat = v.reshape(-1)
        for i in range(0, len(flat), 8):
            f.write("  " + ", ".join(f"{x:.8e}f" for x in flat[i:i + 8]) + ",\n")
        f.write("};\n")
    f.write(f"#endif\n")
card = dict(frame=frame, source_weights=os.path.basename(weights), layers=layers, activation="tanh",
            parameters=params, size_bytes_float32=params * 4,
            output="per motor, -1..1; PWM = 1500 + 500 * output",
            note="simulation-trained only; inputs must be built and scaled as in sim.py")
rep = f"ci_report_{frame}.json"
if os.path.exists(rep):
    card["tests"] = json.load(open(rep))
json.dump(card, open(os.path.join(out, f"{frame}_model_card.json"), "w"), indent=1)
print(json.dumps({k: card[k] for k in ("frame", "layers", "parameters", "size_bytes_float32")}))
