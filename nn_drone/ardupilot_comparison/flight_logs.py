"""Flight logs in one shape for both sides: ArduPilot DataFlash (.BIN) logs and the network simulator's .npz logs."""
from dataclasses import dataclass, field

import numpy as np
from pymavlink import mavutil

from nn_drone.ardupilot_comparison.mission_definition import HOME
from nn_drone.ground_link.geodesy import METRES_PER_DEGREE_LATITUDE

SIM2_RATE_DIVIDER = 8
"""SITL logs its true state (SIM2) at 400 Hz; every 8th sample gives 50 Hz."""
MICROSECONDS = 1e-6
ARMED = 0.5
NETWORK_COLUMNS = dict(time=0, north=1, east=2, height=3, roll=4, pitch=5, heading=6, battery=7)
NETWORK_FIRST_PWM_COLUMN = 8
ARDUPILOT_NAME = "ArduCopter 4.7.2 (PID)"
LOGGED_STATUS_MESSAGES = 12


@dataclass
class Flight:
    name: str
    time: np.ndarray
    north: np.ndarray
    east: np.ndarray
    height: np.ndarray
    attitude_time: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    pwm_time: np.ndarray
    pwm: np.ndarray
    duration: float
    heading: np.ndarray = None
    hover_throttle: float = None
    ekf_error_max: float = None
    messages: list = field(default_factory=list)

    @property
    def log_rates(self):
        return dict(position_hz=len(self.time) / self.duration, pwm_hz=len(self.pwm_time) / self.duration,
                    attitude_hz=len(self.attitude_time) / self.duration)


def read_dataflash(path, message_types):
    """{message type: [message, ...]} for the requested types, in log order."""
    log = mavutil.mavlink_connection(str(path))
    messages = {message_type: [] for message_type in message_types}
    while True:
        message = log.recv_match(type=list(message_types))
        if message is None:
            return messages
        messages[message.get_type()].append(message)


def seconds(message):
    return message.TimeUS * MICROSECONDS


def true_positions(sim2_messages, first_sample=SIM2_RATE_DIVIDER - 1):
    """(time, north, east, up) from SITL's own physics, decimated to 50 Hz."""
    every = sim2_messages[first_sample::SIM2_RATE_DIVIDER]
    return np.array([[seconds(message), message.PN, message.PE, -message.PD] for message in every], float)


def _within(rows, start, end):
    return rows[(rows[:, 0] >= start) & (rows[:, 0] <= end)] if len(rows) else rows


def read_ardupilot_flight(path):
    logged = read_dataflash(path, ("POS", "ATT", "RCOU", "ARM", "MSG", "CTUN", "SIM2"))
    arming = np.array([[seconds(message), message.ArmState] for message in logged["ARM"]], float)
    pwm_rows = np.array([[seconds(message), message.C1, message.C2, message.C3, message.C4] for message in logged["RCOU"]], float)
    estimated = np.array([[seconds(message), message.Lat, message.Lng, message.RelHomeAlt] for message in logged["POS"]], float)
    armed_rows = arming[arming[:, 1] > ARMED] if len(arming) else []
    armed_at = armed_rows[0, 0] if len(armed_rows) else pwm_rows[0, 0]
    disarm_rows = arming[(arming[:, 1] < ARMED) & (arming[:, 0] > armed_at)] if len(arming) else []
    disarmed_at = disarm_rows[0, 0] if len(disarm_rows) else estimated[-1, 0]
    truth = _within(true_positions(logged["SIM2"]), armed_at, disarmed_at)
    north, east, height = truth[:, 1] - truth[0, 1], truth[:, 2] - truth[0, 2], truth[:, 3] - truth[0, 3]
    estimated = _within(estimated, armed_at, disarmed_at)
    estimated_north = (estimated[:, 1] - HOME.latitude) * METRES_PER_DEGREE_LATITUDE
    estimated_east = (estimated[:, 2] - HOME.longitude) * HOME.metres_per_degree_longitude
    ekf_error = np.hypot(np.interp(estimated[:, 0], truth[:, 0], north) - (estimated_north - estimated_north[0]),
                         np.interp(estimated[:, 0], truth[:, 0], east) - (estimated_east - estimated_east[0]))
    attitude = _within(np.array([[seconds(message), message.Roll, message.Pitch, message.Yaw] for message in logged["ATT"]],
                                float), armed_at, disarmed_at)
    pwm_rows = _within(pwm_rows, armed_at, disarmed_at)
    throttle = _within(np.array([[seconds(message), message.ThO, message.ThH] for message in logged["CTUN"]], float),
                       armed_at, disarmed_at)
    return Flight(name=ARDUPILOT_NAME, time=truth[:, 0] - armed_at, north=north, east=east, height=height,
                  attitude_time=attitude[:, 0] - armed_at, roll=attitude[:, 1], pitch=attitude[:, 2],
                  pwm_time=pwm_rows[:, 0] - armed_at, pwm=pwm_rows[:, 1:], duration=disarmed_at - armed_at,
                  heading=attitude[:, 3], hover_throttle=float(np.median(throttle[:, 2])) if len(throttle) else None,
                  ekf_error_max=float(np.max(ekf_error)),
                  messages=[message.Message for message in logged["MSG"]][:LOGGED_STATUS_MESSAGES])


def read_network_flight(path, name):
    data = np.load(path)["data"].astype(float)
    columns = NETWORK_COLUMNS
    time = data[:, columns["time"]]
    return Flight(name=name, time=time, north=data[:, columns["north"]], east=data[:, columns["east"]],
                  height=data[:, columns["height"]], attitude_time=time, roll=data[:, columns["roll"]],
                  pitch=data[:, columns["pitch"]], pwm_time=time, pwm=data[:, NETWORK_FIRST_PWM_COLUMN:], duration=time[-1],
                  heading=data[:, columns["heading"]])

