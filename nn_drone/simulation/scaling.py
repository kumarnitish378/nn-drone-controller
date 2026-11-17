"""Scaling laws from the 1 kg reference airframe, at constant thrust-to-weight.

length ~ m^(1/3), inertia ~ m^(5/3), drag area ~ m^(2/3), motor lag ~ sqrt(length).
A drone scaled this way behaves like the 1 kg one with lengths x L and time x sqrt(L) (Froude
similarity), so the network sees size-normalised inputs and runs with a period of 10 ms x sqrt(L).
"""
import numpy as np

REFERENCE_ROTATIONAL_DAMPING = 0.002


def length_scale(nominal_mass_kg):
    return nominal_mass_kg ** (1 / 3)


def inertia_scale(nominal_mass_kg):
    return nominal_mass_kg ** (5 / 3)


def drag_area_scale(nominal_mass_kg):
    return nominal_mass_kg ** (2 / 3)


def time_scale(nominal_mass_kg):
    return np.sqrt(length_scale(nominal_mass_kg))


def speed_scale(nominal_mass_kg):
    """Speeds scale with sqrt(length)."""
    return nominal_mass_kg ** (1 / 6)
