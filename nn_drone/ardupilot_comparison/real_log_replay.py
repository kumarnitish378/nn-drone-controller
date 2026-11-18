"""Run the quad network against a REAL ArduCopter flight log.

1. Shadow replay (open loop): at every instant the network is given what the real autopilot knew (gyro,
   accelerometer, attitude, position error to ArduPilot's own target, velocity) and its motor PWM is compared
   with the PWM ArduPilot actually sent. The network does not fly the drone, so this shows whether it reacts
   in the same direction and how strongly, not whether it would fly.
2. Closed loop in our simulator: the network follows the same target track ArduPilot was following, and its
   tracking error is compared with the real flight's.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymavlink import mavutil

from nn_drone.geometry import quaternion_from_euler, wrap_angle
from nn_drone.paths import default_weights
from nn_drone.policy import NeuralPilot
from nn_drone.units import PWM_DISARMED_US, PWM_FULL_THROTTLE_US, action_to_pwm, clip_action

DEFAULT_LABEL = ("erlerobot/logs: pxf/copter/5-12-2014/218.BIN (ArduCopter 3.3-dev, Erle-Copter quad X, real flight)")
MODE_NAMES = {0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED", 5: "LOITER", 6: "RTL", 9: "LAND"}
NEUTRAL_BATTERY_V = 15.6
POSITION_ERROR_CLIP = 2.5
SELF_TEST_STEPS = 1500
SELF_TEST_MASS_KG = 1.5
SELF_TEST_LIMIT_US = 1.0
CENTIMETRES = 100.0
MICROSECONDS = 1e-6
MILLISECONDS = 1e-3
LOG_GAP_S = 1.0
MIN_STRETCH_S = 8
STRETCH_TRIM_START_S, STRETCH_TRIM_END_S = 1.0, 0.5
MODE_LOOKUP_DELAY_S = 0.5
THROTTLE_FRACTION_TO_PERMILLE = 1000.0
QUAD_MOTORS = 4
FLYING_THROTTLE = 200
THROTTLE_SAMPLES = 50
SETTLE_BEFORE_MEASURING_S = 3.0
COMPARISON_PERIOD_S = 0.1
TARGET_STILL_MPS, VEHICLE_STILL_MPS = 0.3, 0.7
MIN_STRETCH_SAMPLES = 20
LARGE_MOVE_SHARE_OF_STD = 0.5
PWM_LIMIT_MARGIN_US = 10
SIMULATOR_SEED = 5
LOGGED_TYPES = ("ATT", "IMU", "NTUN", "CTUN", "RCOU", "GPS", "MODE", "CMD", "PARM", "PSCN", "PSCE", "PSCD")


class ShadowPilot:
    """Feeds the network externally supplied states (simulator axes: world x north, y west, z up; body x
    forward, y left, z up) through the simulator's own input builder, keeping the network's memory."""

    def __init__(self, mass_kg, weights=None, position_error_clip=POSITION_ERROR_CLIP, integrate=True):
        self.pilot = NeuralPilot.from_checkpoint(weights or default_weights("quad_x"))
        self.env = self.pilot.make_environment(1, randomize=False, seed=0, noise=False, nominal_mass_kg=mass_kg,
                                               position_error_clip_m=position_error_clip)
        self.integrate = integrate
        self.reset()

    @property
    def period(self):
        return self.env.control_period

    def reset(self):
        env = self.env
        env.previous_action[:] = 0.0
        env.integrator.reset(np.arange(1))
        env.history.mark_reset(np.arange(1))
        env.history.latest = None
        self.last_target = None
        self.has_reference = False

    def step(self, attitude, body_rates, specific_force, position, velocity, battery_voltage, target, target_heading,
             applied_action=None):
        """One network step. `applied_action`: the motor command (-1..1) really sent on the previous step, if known;
        otherwise the network's own previous output is used. Returns PWM per motor."""
        env, state = self.env, self.env.state
        position, velocity, target = (np.asarray(vector, float) for vector in (position, velocity, target))
        if self.last_target is not None and self.integrate:
            if not self.has_reference:
                env.integrator.set_reference_target(np.arange(1), self.last_target[None])
                self.has_reference = True
            env.integrator.update((self.last_target - position)[None], velocity[None], self.last_target[None],
                                  env.airframe.length_scale, env.airframe.control_period,
                                  env.airframe.normalised_step_length)
        self.last_target = target.copy()
        if applied_action is not None:
            env.previous_action[0] = applied_action
        state.attitude[0], state.body_rates[0], state.specific_force_body[0] = attitude, body_rates, specific_force
        state.position[0], state.velocity[0], state.battery_voltage[0] = position, velocity, battery_voltage
        env.set_target(target, target_heading)
        action = clip_action(self.pilot.command(env.observe())[0])
        env.history.advance()
        env.previous_action[0] = action
        return action_to_pwm(action)


def shadow_self_test(log=print):
    """States from our own simulator (noise off) through the shadow builder must give the simulator's outputs."""
    shadow = ShadowPilot(SELF_TEST_MASS_KG)
    pilot = shadow.pilot
    env = pilot.make_environment(1, randomize=False, seed=3, noise=False, nominal_mass_kg=SELF_TEST_MASS_KG,
                                 position_error_clip_m=POSITION_ERROR_CLIP)
    shadow.env.previous_action[0] = env.previous_action[0]
    worst = 0.0
    for step in range(SELF_TEST_STEPS):
        target = np.array([3.0 * (step > 200), -2.0 * (step > 700), 2.0 + 1.0 * (step > 1100)])
        heading = 0.4 * (step > 500)
        env.set_target(target, heading)
        action = pilot.command(env.observe())
        state = env.state
        pwm = shadow.step(state.attitude[0], state.body_rates[0], state.specific_force_body[0], state.position[0],
                          state.velocity[0], state.battery_voltage[0], target, heading)
        worst = max(worst, float(np.abs(pwm - action_to_pwm(np.clip(action[0], -1, 1))).max()))
        env.step(action, training=False)
    log(f"self-test: largest PWM difference between the shadow builder and the simulator = {worst:.3f} us")
    return worst


@dataclass
class FlightLog:
    attitude: np.ndarray
    imu: np.ndarray
    navigation: np.ndarray
    climb: np.ndarray
    motor_output: np.ndarray
    gps: np.ndarray
    modes: list
    commands: list


def read_flight_log(path):
    """Same arrays for old logs (ArduCopter 3.3: NTUN/CTUN, TimeMS) and current ones (PSCN/PSCE/PSCD, TimeUS).
    navigation: t, target N, target E, N, E, velocity N, velocity E (cm). climb: t, target alt, alt, climb rate, throttle."""
    log = mavutil.mavlink_connection(str(path))
    rows = {name: [] for name in ("ATT", "IMU", "NTUN", "CTUN", "RCOU", "GPS")}
    position_control = {"PSCN": [], "PSCE": [], "PSCD": []}
    modes, commands, throttle, last_time = [], [], [], 0.0
    while True:
        message = log.recv_match()
        if message is None:
            break
        kind, current_format = message.get_type(), hasattr(message, "TimeUS")
        if current_format:
            time = message.TimeUS * MICROSECONDS
        elif hasattr(message, "TimeMS") and kind != "GPS":
            time = message.TimeMS * MILLISECONDS
        else:
            time = None
        if time is not None:
            last_time = time
        if kind == "ATT":
            rows[kind].append([time, message.Roll, message.Pitch, message.Yaw, message.DesRoll, message.DesPitch, message.DesYaw])
        elif kind == "IMU" and (not current_format or message.I == 0):
            rows[kind].append([time, message.GyrX, message.GyrY, message.GyrZ, message.AccX, message.AccY, message.AccZ])
        elif kind == "NTUN":
            rows[kind].append([time, message.DPosX, message.DPosY, message.PosX, message.PosY, message.VelX, message.VelY])
        elif kind == "CTUN":
            if current_format:
                throttle.append([time, message.ThO * THROTTLE_FRACTION_TO_PERMILLE])
            else:
                rows[kind].append([time, message.DAlt, message.Alt, message.CRt, message.ThrOut])
        elif kind in position_control:
            axis = kind[-1]
            position_control[kind].append([time, getattr(message, f"TP{axis}"), getattr(message, f"P{axis}"),
                                           getattr(message, f"V{axis}")])
        elif kind == "RCOU":
            channels = [message.C1, message.C2, message.C3, message.C4] if current_format else \
                [message.Ch1, message.Ch2, message.Ch3, message.Ch4]
            rows[kind].append([time, *channels])
        elif kind == "GPS":
            rows[kind].append([time if current_format else message.T * MILLISECONDS, message.NSats, message.HDop, message.Spd])
        elif kind == "MODE":
            modes.append([last_time, int(message.ModeNum) if current_format else int(message.Mode)])
        elif kind == "CMD":
            commands.append([time, message.CNum, message.CId, message.Alt])
    if position_control["PSCN"]:
        rows["NTUN"], rows["CTUN"] = _from_position_control(position_control, throttle)
    arrays = {name: np.array(values, float) for name, values in rows.items()}
    return FlightLog(arrays["ATT"], arrays["IMU"], arrays["NTUN"], arrays["CTUN"], arrays["RCOU"], arrays["GPS"], modes, commands)


def _from_position_control(position_control, throttle):
    """Current format (metres, down positive) -> the old layout (centimetres, up positive)."""
    north, east, down = (np.array(position_control[name], float) for name in ("PSCN", "PSCE", "PSCD"))
    length = min(len(north), len(east), len(down))
    north, east, down = north[:length], east[:length], down[:length]
    navigation = np.stack([north[:, 0], north[:, 1] * CENTIMETRES, east[:, 1] * CENTIMETRES, north[:, 2] * CENTIMETRES,
                           east[:, 2] * CENTIMETRES, north[:, 3] * CENTIMETRES, east[:, 3] * CENTIMETRES], 1)
    throttle = np.array(throttle, float)
    climb = np.stack([down[:, 0], -down[:, 1], -down[:, 2], -down[:, 3] * CENTIMETRES,
                      np.interp(down[:, 0], throttle[:, 0], throttle[:, 1])], 1)
    return navigation.tolist(), climb.tolist()


def interpolate(times, rows, column, angle=False):
    values = np.unwrap(np.radians(rows[:, column])) if angle else rows[:, column]
    return np.interp(times, rows[:, 0], values)


def control_channels(pwm):
    """4 motor PWMs (ArduPilot quad X order: front-right, rear-left, front-left, rear-right) -> the four control directions."""
    front_right, rear_left, front_left, rear_right = pwm[:, 0], pwm[:, 1], pwm[:, 2], pwm[:, 3]
    return dict(collective=(front_right + rear_left + front_left + rear_right) / QUAD_MOTORS,
                roll=((rear_left + front_left) - (front_right + rear_right)) / QUAD_MOTORS,
                pitch=((front_right + front_left) - (rear_left + rear_right)) / QUAD_MOTORS,
                yaw=((front_right + rear_left) - (front_left + rear_right)) / QUAD_MOTORS)


def position_controlled_stretches(flight_log):
    """Stretches where ArduPilot was controlling position (NTUN logged) and the drone was flying, split at mode changes."""
    navigation, climb, modes = flight_log.navigation, flight_log.climb, flight_log.modes
    gaps = np.nonzero(np.diff(navigation[:, 0]) > LOG_GAP_S)[0]
    starts, ends = np.r_[0, gaps + 1], np.r_[gaps, len(navigation) - 1]

    def mode_at(time):
        earlier = [mode for mode_time, mode in modes if mode_time <= time]
        return MODE_NAMES.get(earlier[-1], str(earlier[-1])) if earlier else "?"
    stretches = []
    for start, end in zip(starts, ends):
        cuts = ([navigation[start, 0]] + [time for time, _ in modes if navigation[start, 0] < time < navigation[end, 0]]
                + [navigation[end, 0]])
        for begin, finish in zip(cuts[:-1], cuts[1:]):
            if finish - begin < MIN_STRETCH_S:
                continue
            throttle = np.interp(np.linspace(begin, finish, THROTTLE_SAMPLES), climb[:, 0], climb[:, 4])
            if np.median(throttle) < FLYING_THROTTLE:
                continue
            stretches.append((begin + STRETCH_TRIM_START_S, finish - STRETCH_TRIM_END_S, mode_at(begin + MODE_LOOKUP_DELAY_S)))
    return stretches


@dataclass
class StretchSignals:
    """One stretch of the log resampled to the network's rate, in ArduPilot's north-east-down conventions."""
    time: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    yaw: np.ndarray
    desired_roll: np.ndarray
    desired_pitch: np.ndarray
    desired_yaw: np.ndarray
    gyro: np.ndarray
    accelerometer: np.ndarray
    target: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    ardupilot_pwm: np.ndarray

    @classmethod
    def resample(cls, flight_log, start, end, period):
        times = np.arange(start, end, period)
        attitude, imu, navigation, climb = flight_log.attitude, flight_log.imu, flight_log.navigation, flight_log.climb
        column = lambda rows, index, angle=False: interpolate(times, rows, index, angle)
        target = np.stack([column(navigation, 1) / CENTIMETRES, column(navigation, 2) / CENTIMETRES, column(climb, 1)], 1)
        position = np.stack([column(navigation, 3) / CENTIMETRES, column(navigation, 4) / CENTIMETRES, column(climb, 2)], 1)
        velocity = np.stack([column(navigation, 5) / CENTIMETRES, column(navigation, 6) / CENTIMETRES,
                             column(climb, 3) / CENTIMETRES], 1)
        return cls(times, column(attitude, 1, True), column(attitude, 2, True), column(attitude, 3, True),
                   column(attitude, 4, True), column(attitude, 5, True), column(attitude, 6, True),
                   np.stack([column(imu, index) for index in (1, 2, 3)], 1), np.stack([column(imu, index) for index in (4, 5, 6)], 1),
                   target, position, velocity, np.stack([column(flight_log.motor_output, index) for index in (1, 2, 3, 4)], 1))


def ned_to_simulator(vector):
    """North-east-down (or forward-right-down) -> the simulator's north-west-up (forward-left-up)."""
    return np.array([vector[0], -vector[1], -vector[2]])


def shadow_replay(shadow, signals):
    shadow.reset()
    network_pwm = np.zeros((len(signals.time), QUAD_MOTORS))
    for step in range(len(signals.time)):
        attitude = quaternion_from_euler(np.array([signals.roll[step]]), np.array([-signals.pitch[step]]),
                                         np.array([-signals.yaw[step]]))[0]
        target, position, velocity = signals.target[step], signals.position[step], signals.velocity[step]
        network_pwm[step] = shadow.step(attitude, ned_to_simulator(signals.gyro[step]), ned_to_simulator(signals.accelerometer[step]),
                                        [position[0], -position[1], position[2]], [velocity[0], -velocity[1], velocity[2]],
                                        NEUTRAL_BATTERY_V, [target[0], -target[1], target[2]], -signals.desired_yaw[step])
    return network_pwm


def closed_loop_replay(pilot, signals, mass_kg):
    env = pilot.make_environment(1, randomize=False, seed=SIMULATOR_SEED, nominal_mass_kg=mass_kg,
                                 position_error_clip_m=POSITION_ERROR_CLIP)
    state = env.state
    state.position[0] = [signals.position[0, 0], -signals.position[0, 1], signals.position[0, 2]]
    state.velocity[0] = [signals.velocity[0, 0], -signals.velocity[0, 1], signals.velocity[0, 2]]
    state.attitude[0] = quaternion_from_euler(np.array([0.0]), np.array([0.0]), np.array([-signals.yaw[0]]))[0]
    steps = len(signals.time)
    error, pwm, positions = np.zeros((steps, 3)), np.zeros((steps, QUAD_MOTORS)), np.zeros((steps, 3))
    for step in range(steps):
        target = signals.target[step]
        env.set_target([target[0], -target[1], target[2]], -signals.desired_yaw[step])
        result = pilot.fly_step(env)
        error[step] = env.targets.position[0] - state.position[0]
        pwm[step] = result.pwm[0]
        positions[step] = state.position[0]
    return error, pwm, positions


def _demeaned_by_stretch(values, stretch_index, mask, stretch_count):
    pieces = []
    for stretch in range(stretch_count):
        selected = (stretch_index == stretch) & mask
        if selected.sum() > MIN_STRETCH_SAMPLES:
            pieces.append(values[selected] - values[selected].mean())
    return np.concatenate(pieces)


def replay_log(log_path, output_dir, mass_kg=1.5, tag="real_log_replay", label=DEFAULT_LABEL, log=print):
    from nn_drone.charts.comparison_charts import real_log_chart
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shadow_self_test(log) >= SELF_TEST_LIMIT_US:
        raise RuntimeError("the shadow input builder does not reproduce the simulator")
    flight_log = read_flight_log(log_path)
    stretches = position_controlled_stretches(flight_log)
    shadow = ShadowPilot(mass_kg, integrate=False)
    period = shadow.period
    every = max(1, int(round(COMPARISON_PERIOD_S / period)))
    samples, stretch_stats, tracks = [], [], []
    settle = int(SETTLE_BEFORE_MEASURING_S / period)
    for index, (start, end, mode) in enumerate(stretches):
        signals = StretchSignals.resample(flight_log, start, end, period)
        network_pwm = shadow_replay(shadow, signals)
        target_speed = np.hypot(np.gradient(signals.target[:, 0], signals.time), np.gradient(signals.target[:, 1], signals.time))
        vehicle_speed = np.hypot(signals.velocity[:, 0], signals.velocity[:, 1])
        for step in range(0, len(signals.time), every):
            error = signals.target[step] - signals.position[step]
            samples.append([signals.time[step], index, *signals.ardupilot_pwm[step], *network_pwm[step], *error,
                            np.degrees(signals.desired_roll[step] - signals.roll[step]),
                            np.degrees(signals.desired_pitch[step] - signals.pitch[step]),
                            np.degrees(wrap_angle(signals.desired_yaw[step] - signals.yaw[step])),
                            float(target_speed[step] < TARGET_STILL_MPS and vehicle_speed[step] < VEHICLE_STILL_MPS)])
        simulated_error, simulated_pwm, simulated_position = closed_loop_replay(shadow.pilot, signals, mass_kg)
        real_horizontal = np.hypot(signals.target[settle:, 0] - signals.position[settle:, 0],
                                   signals.target[settle:, 1] - signals.position[settle:, 1])
        real_vertical = np.abs(signals.target[settle:, 2] - signals.position[settle:, 2])
        simulated_horizontal = np.hypot(simulated_error[settle:, 0], simulated_error[settle:, 1])
        stretch_stats.append(dict(
            mode=mode, start_s=round(float(start), 1), length_s=round(float(end - start), 1),
            target_speed_mean_mps=round(float(target_speed.mean()), 2),
            real_horizontal_error_mean_m=round(float(real_horizontal.mean()), 3),
            real_horizontal_error_max_m=round(float(real_horizontal.max()), 3),
            real_height_error_mean_m=round(float(real_vertical.mean()), 3),
            network_sim_horizontal_error_mean_m=round(float(simulated_horizontal.mean()), 3),
            network_sim_horizontal_error_max_m=round(float(simulated_horizontal.max()), 3),
            network_sim_height_error_mean_m=round(float(np.abs(simulated_error[settle:, 2]).mean()), 3)))
        tracks.append(dict(time=signals.time, target=signals.target, real=signals.position,
                           simulated=np.stack([simulated_position[:, 0], -simulated_position[:, 1], simulated_position[:, 2]], 1),
                           mode=mode, simulated_pwm=simulated_pwm))
    samples = np.array(samples)
    report = _report(samples, flight_log, stretches, stretch_stats, mass_kg, period, label)
    (output_dir / f"{tag}.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    real_log_chart(output_dir / f"{tag}.png", samples, stretches, stretch_stats, tracks)
    return report


SAMPLE_COLUMNS = dict(stretch=1, ardupilot_pwm=slice(2, 6), network_pwm=slice(6, 10), roll_demand=13, pitch_demand=14,
                      yaw_demand=15, hovering=16)


def _report(samples, flight_log, stretches, stretch_stats, mass_kg, period, label):
    columns = SAMPLE_COLUMNS
    ardupilot_pwm, network_pwm = samples[:, columns["ardupilot_pwm"]], samples[:, columns["network_pwm"]]
    stretch_index = samples[:, columns["stretch"]]
    ardupilot_channels, network_channels = control_channels(ardupilot_pwm), control_channels(network_pwm)
    hovering = samples[:, columns["hovering"]] > 0.5
    everything = np.ones(len(samples), bool)
    demeaned = lambda values, mask: _demeaned_by_stretch(values, stretch_index, mask, len(stretches))

    def agreement(mask):
        result = {}
        for name in ("collective", "roll", "pitch", "yaw"):
            ardupilot, network = demeaned(ardupilot_channels[name], mask), demeaned(network_channels[name], mask)
            large = np.abs(ardupilot) > ardupilot.std() * LARGE_MOVE_SHARE_OF_STD
            result[name] = dict(correlation=round(float(np.corrcoef(ardupilot, network)[0, 1]), 2),
                                same_direction_share=round(float((np.sign(ardupilot) == np.sign(network))[large].mean()), 2),
                                ardupilot_rms_us=round(float(ardupilot.std()), 1), network_rms_us=round(float(network.std()), 1))
        return result
    shadow = dict(whole_replay=agreement(everything), hovering_only=agreement(hovering),
                  hovering_share=round(float(hovering.mean()), 2))
    demand_agreement = {}
    for name in ("roll", "pitch", "yaw"):
        demand = demeaned(samples[:, columns[f"{name}_demand"]], everything)
        demand_agreement[name] = dict(
            ardupilot=round(float(np.corrcoef(demand, demeaned(ardupilot_channels[name], everything))[0, 1]), 2),
            network=round(float(np.corrcoef(demand, demeaned(network_channels[name], everything))[0, 1]), 2))
    shadow["correlation_with_ardupilots_attitude_demand"] = demand_agreement
    motor_output, imu, gps = flight_log.motor_output, flight_log.imu, flight_log.gps
    pwm_low, pwm_high = PWM_DISARMED_US + PWM_LIMIT_MARGIN_US, PWM_FULL_THROTTLE_US - PWM_LIMIT_MARGIN_US
    return dict(
        log=label, assumed_mass_kg=mass_kg, network_rate_hz=round(1 / period, 1), compared_at_hz=round(1 / COMPARISON_PERIOD_S),
        log_facts=dict(flight_s=round(float(motor_output[-1, 0] - motor_output[0, 0]), 0),
                       gps_satellites_median=float(np.median(gps[:, 1])), gps_hdop_median=float(np.median(gps[:, 2])),
                       pwm_rate_hz=round(len(motor_output) / (motor_output[-1, 0] - motor_output[0, 0]), 1),
                       imu_rate_hz=round(len(imu) / (imu[-1, 0] - imu[0, 0]), 1), battery_voltage_logged=False,
                       mission_items=[[int(command[1]), int(command[2]), command[3]] for command in flight_log.commands]),
        seconds_replayed=round(float(sum(end - start for start, end, _ in stretches)), 0),
        pwm=dict(ardupilot_mean_us=round(float(ardupilot_pwm.mean()), 0), network_mean_us=round(float(network_pwm.mean()), 0),
                 ardupilot_range_us=[float(ardupilot_pwm.min()), float(ardupilot_pwm.max())],
                 network_range_us=[round(float(network_pwm.min())), round(float(network_pwm.max()))],
                 ardupilot_change_per_0p1s_us=round(float(np.abs(np.diff(ardupilot_pwm, axis=0)).mean()), 1),
                 network_change_per_0p1s_us=round(float(np.abs(np.diff(network_pwm, axis=0)).mean()), 1),
                 network_time_at_limits_share=round(float(((network_pwm < pwm_low) | (network_pwm > pwm_high)).mean()), 3)),
        shadow_replay=shadow, closed_loop_same_target_track=stretch_stats)
