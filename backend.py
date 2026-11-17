"""Array backend: NumPy on the CPU (default) or CuPy on an NVIDIA GPU.

sim.py and ppo.py do `from backend import xp as np`, so the backend is fixed when they are first
imported. It is chosen by the NN_DEVICE environment variable:
    NN_DEVICE=cpu   NumPy (default; what every tool except train.py uses)
    NN_DEVICE=cuda  CuPy  (pip install "cupy-cuda12x[ctk]")
    NN_DEVICE=auto  CuPy if a working GPU is found, otherwise NumPy
train.py sets it from its --device option before importing sim and ppo.
"""
import os
import warnings
import numpy


def _load_cupy():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # "CUDA path could not be detected": not needed with [ctk]
        import cupy
    a = cupy.ones((8, 8), cupy.float32)
    float((a @ a).sum())                          # fails here if the CUDA libraries are missing
    return cupy


_req = os.environ.get("NN_DEVICE", "cpu").lower()
if _req in ("cuda", "gpu"):
    xp = _load_cupy()
elif _req == "auto":
    try:
        xp = _load_cupy()
    except Exception:
        xp = numpy
elif _req == "cpu":
    xp = numpy
else:
    raise ValueError(f"NN_DEVICE must be cpu, cuda or auto, not {_req!r}")

DEVICE = "cpu" if xp is numpy else "cuda"


class _CupyRNG:
    """The subset of numpy.random.Generator that sim.py and ppo.py use, on the GPU."""

    def __init__(self, seed):
        self.g = xp.random.default_rng(seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.g.uniform(low, high, size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return loc + scale * self.g.standard_normal(size)

    def standard_normal(self, size=None, dtype=numpy.float64):
        return self.g.standard_normal(size, dtype=dtype)

    def random(self, size=None):
        return self.g.random(size)

    def permutation(self, n):
        return xp.argsort(self.g.random(n))


def default_rng(seed=None):
    return numpy.random.default_rng(seed) if xp is numpy else _CupyRNG(seed)


def to_numpy(a):
    """Copy an array to host memory (no-op for NumPy arrays)."""
    return a.get() if hasattr(a, "get") and not isinstance(a, numpy.ndarray) else numpy.asarray(a)


def device_name():
    if xp is numpy:
        return "cpu"
    return "cuda: " + xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
