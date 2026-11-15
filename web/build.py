"""Build web/index.html: one self-contained page with the simulator core and both networks inside.
usage (from the repository root): python3 web/build.py"""
import json, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)

def weights(npz):
    d = np.load(os.path.join(root, npz)); W0 = d["pi_W0"]; M = d["pi_W2"].shape[1]; nin = 21 + M + 2 * (14 + M)
    if W0.shape[0] < nin:                          # older weights: missing inputs have zero weight
        W0 = np.concatenate([W0, np.zeros((nin - W0.shape[0], W0.shape[1]), np.float32)])
    r = lambda a: [[round(float(x), 6) for x in row] for row in a] if a.ndim == 2 else [round(float(x), 6) for x in a]
    return json.dumps(dict(W0=r(W0), b0=r(d["pi_b0"]), W1=r(d["pi_W1"]), b1=r(d["pi_b1"]), W2=r(d["pi_W2"]), b2=r(d["pi_b2"])), separators=(",", ":"))

t = open(os.path.join(here, "template.html")).read()
t = t.replace("/*__CORE__*/", open(os.path.join(here, "core.js")).read())
t = t.replace("/*__WQ__*/", weights("quad_x_policy_v4.npz")).replace("/*__WH__*/", weights("hex_x_policy_v1.npz"))
open(os.path.join(here, "index.html"), "w").write(t)
print("web/index.html", len(t) // 1024, "kB")
