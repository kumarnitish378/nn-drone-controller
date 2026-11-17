"""Repository locations and the default network for each frame."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPOSITORY_ROOT / "models"
RESULTS_DIR = REPOSITORY_ROOT / "results"
WEB_DIR = REPOSITORY_ROOT / "web"

DEFAULT_WEIGHTS = {"quad_x": MODELS_DIR / "quad_x_policy_v4.npz", "hex_x": MODELS_DIR / "hex_x_policy_v1.npz"}


def default_weights(frame):
    return DEFAULT_WEIGHTS[frame]
