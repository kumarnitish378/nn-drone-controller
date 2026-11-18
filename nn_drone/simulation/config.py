"""Simulator settings.

`SimulatorConfig` holds what a user may change (airframe, randomisation, reward weights).
`SimulatorProfile` fixes what a trained network depends on: which inputs it sees, how often it runs
and how the position-error integrator behaves. Profiles v1..v4 reproduce the four generations of the
experiment at 100 Hz, so every network can be flown in the simulator it was trained in; v5 runs the
network in the flight controller's 400 Hz main loop.
"""
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum


@dataclass(frozen=True)
class RewardWeights:
    action_change_squared: float = 1.0
    action_change_absolute: float = 0.3
    excess_speed: float = 0.05
    free_speed: float = 3.0
    excess_tilt: float = 0.5
    free_tilt_rad: float = 0.6


LEGACY_REWARD_WEIGHTS = RewardWeights(action_change_squared=0.3, action_change_absolute=0.0, excess_speed=0.03,
                                      free_speed=4.0, excess_tilt=0.0)


@dataclass(frozen=True)
class SimulatorConfig:
    nominal_mass_kg: float = 1.0
    nominal_mass_range_kg: tuple = None
    solid_ground: bool = False
    start_altitude_m: float = 0.0
    start_altitude_range_m: tuple = None
    arm_length_m: float = 0.17
    inertia_kg_m2: tuple = (0.010, 0.010, 0.018)
    thrust_to_weight: float = 2.8
    yaw_torque_per_thrust_m: float = 0.016
    motor_time_constant_s: float = 0.05
    drag_n_per_mps: float = 0.25
    nominal_battery_voltage: float = 16.0
    physics_substeps: int = 2
    position_error_clip_m: float = 5.0
    moving_target_share: float = 0.4
    reward: RewardWeights = field(default_factory=RewardWeights)


class IntegratorSettleRule(Enum):
    """When the position-error integrator is allowed to accumulate."""
    WITHIN_CAPTURE_RADIUS = "within_capture_radius"
    SLOW_RELATIVE_TO_TARGET = "slow_relative_to_target"
    VEHICLE_AND_TARGET_STILL = "vehicle_and_target_still"


@dataclass(frozen=True)
class IntegratorSettings:
    settle_rule: IntegratorSettleRule
    capture_radius: float
    clamp: float
    input_scale: float = 1.5
    settled_speed: float = 0.5
    still_target_speed: float = 0.05
    target_jump_speed: float = 5.0


REFERENCE_CONTROL_PERIOD_S = 0.01
"""Control period of the 1 kg airframe in v1..v4. Size-normalised time is counted in these reference steps:
episode length, discounting, history spacing and the per-step rewards are defined per reference step."""


@dataclass(frozen=True)
class ControlTiming:
    """How often the network runs. `scales_with_size`: the period grows with sqrt(L) (Froude similarity),
    otherwise every drone runs at the same rate, as in a flight controller's fixed main loop."""
    period_s: float
    scales_with_size: bool
    physics_substeps: int

    @property
    def rate_hz(self):
        return 1.0 / self.period_s


FROUDE_SCALED_100_HZ = ControlTiming(period_s=REFERENCE_CONTROL_PERIOD_S, scales_with_size=True, physics_substeps=2)
MAIN_LOOP_400_HZ = ControlTiming(period_s=1 / 400, scales_with_size=False, physics_substeps=1)


@dataclass(frozen=True)
class SimulatorProfile:
    name: str
    has_integrator_input: bool
    history_frames: int
    has_mass_class_input: bool
    moving_targets: bool
    air_density_effect: bool
    integrator: IntegratorSettings
    default_reward: RewardWeights
    control_timing: ControlTiming = FROUDE_SCALED_100_HZ

    def input_size(self, motor_count):
        size = SENSOR_FEATURES + motor_count
        if self.has_integrator_input:
            size += VECTOR_FEATURES
        size += self.history_frames * history_frame_size(motor_count)
        if self.has_mass_class_input:
            size += 1
        return size


VECTOR_FEATURES = 3
SENSOR_FEATURES = 17
"""gyro, accelerometer, position error, velocity (3 each); roll, pitch, heading sin/cos, battery (1 each)."""
HISTORY_SENSOR_FEATURES = 14
"""gyro, accelerometer, position error, velocity (3 each); roll, pitch (1 each)."""


def history_frame_size(motor_count):
    return HISTORY_SENSOR_FEATURES + motor_count


PROFILE_V1 = SimulatorProfile(
    name="v1", has_integrator_input=False, history_frames=0, has_mass_class_input=False, moving_targets=False,
    air_density_effect=False, default_reward=LEGACY_REWARD_WEIGHTS,
    integrator=IntegratorSettings(IntegratorSettleRule.WITHIN_CAPTURE_RADIUS, capture_radius=1.0, clamp=1.5))
PROFILE_V2 = replace(PROFILE_V1, name="v2", has_integrator_input=True)
PROFILE_V3 = SimulatorProfile(
    name="v3", has_integrator_input=True, history_frames=2, has_mass_class_input=False, moving_targets=True,
    air_density_effect=False, default_reward=RewardWeights(),
    integrator=IntegratorSettings(IntegratorSettleRule.SLOW_RELATIVE_TO_TARGET, capture_radius=0.5, clamp=1.5))
PROFILE_V4 = replace(
    PROFILE_V3, name="v4", has_mass_class_input=True, air_density_effect=True,
    integrator=IntegratorSettings(IntegratorSettleRule.VEHICLE_AND_TARGET_STILL, capture_radius=1.0, clamp=4.0))

PROFILE_V5 = replace(PROFILE_V4, name="v5", control_timing=MAIN_LOOP_400_HZ)

PROFILES = {profile.name: profile for profile in (PROFILE_V1, PROFILE_V2, PROFILE_V3, PROFILE_V4, PROFILE_V5)}
CURRENT_PROFILE = PROFILE_V5
UNLABELLED_FILE_PROFILE = PROFILE_V4
"""Weights files written before profiles were recorded were all trained in v1..v4; their input count picks v4."""


def profile_named(name):
    if name not in PROFILES:
        raise ValueError(f"unknown simulator profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]


def default_config_for(profile, **overrides):
    defaults = SimulatorConfig(reward=profile.default_reward, physics_substeps=profile.control_timing.physics_substeps)
    return apply_overrides(defaults, overrides)


def apply_overrides(settings, overrides):
    """Return a copy of a (nested) frozen dataclass with fields replaced; nested dataclasses take dicts."""
    known = {item.name: item for item in fields(settings)}
    changes = {}
    for name, value in overrides.items():
        if name not in known:
            raise ValueError(f"{type(settings).__name__} has no setting {name!r}; known: {sorted(known)}")
        current = getattr(settings, name)
        if is_dataclass(current) and isinstance(value, dict):
            value = apply_overrides(current, value)
        elif isinstance(value, list):
            value = tuple(value)
        changes[name] = value
    return replace(settings, **changes)
