"""Landing detection and the motor sequences around take-off and landing, shared by the full-mission test
and the ground-station link.

A drone counts as landed when it has stopped moving vertically while its target is well below it, for a
confirmation time, and the network already commands less than lift-off throttle (so a slow drone still
hovering in the air is never taken for landed).
"""
from dataclasses import dataclass

import numpy as np

PWM_ARMED_IDLE_US = 1150.0
PWM_SPOOL_END_US = 1450.0
"""Spool-up ends below hover throttle; the network then takes over."""


@dataclass(frozen=True)
class MotorSequenceTiming:
    disarmed_s: float = 1.0
    armed_idle_s: float = 2.0
    spool_up_s: float = 2.0
    landing_confirm_s: float = 1.0
    ramp_to_idle_s: float = 1.0
    idle_before_disarm_s: float = 2.0


@dataclass(frozen=True)
class LandingCriteria:
    max_vertical_speed_mps: float = 0.15
    target_below_by_m: float = 0.5
    max_mean_pwm_us: float = PWM_SPOOL_END_US


class LandingDetector:
    def __init__(self, count, period, criteria=LandingCriteria(), timing=MotorSequenceTiming()):
        self.period = period
        self.criteria = criteria
        self.timing = timing
        self.confirmed_for = np.zeros(count)

    def looks_landed(self, vertical_speed, height, target_height, mean_pwm):
        criteria = self.criteria
        return ((np.abs(vertical_speed) < criteria.max_vertical_speed_mps)
                & (target_height < height - criteria.target_below_by_m)
                & (mean_pwm < criteria.max_mean_pwm_us))

    def update(self, candidate):
        """`candidate`: drones that currently look landed. Returns the drones whose landing is confirmed."""
        self.confirmed_for = np.where(candidate, self.confirmed_for + self.period, 0.0)
        return self.confirmed_for >= self.timing.landing_confirm_s

    def clear(self, mask=slice(None)):
        self.confirmed_for[mask] = 0.0
