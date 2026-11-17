"""The simulated vehicle behind the ground-station link: arming, flight modes, mission execution and the motor
sequences around take-off and landing. The neural network flies it; this class only decides where to."""
import math
from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.flight_tests.landing import PWM_SPOOL_END_US, LandingDetector, MotorSequenceTiming
from nn_drone.geometry import euler_from_quaternion, upright_cosine, wrap_angle
from nn_drone.ground_link.geodesy import HomePoint
from nn_drone.ground_link.navigation import CarrotGuidance, Goal, GoalKind, NavigationSpeeds
from nn_drone.ground_link.parameters import WIND_PARAMETERS, default_parameters, rebuilds_airframe
from nn_drone.policy import NeuralPilot
from nn_drone.units import CENTIMETRES_PER_METRE, PWM_DISARMED_US, PWM_SPAN_US, clip_action, pwm_to_action

STATUS_TEXT_LIMIT = 50
MASS_LIMITS_KG = (0.2, 500)
MAX_IDLE_THROTTLE = 0.3
CRASH_TOUCHDOWN_MPS = 3.0
MIN_TAKEOFF_ALTITUDE_M = 2.0
DEFAULT_TAKEOFF_ALTITUDE_M = 10.0
MIN_CIRCLE_RADIUS_PARAM = 0.1
MIN_DIRECTION_NORM = 1e-3
NOISE_ON = 0.5
SIMULATION_SEED = 1
PERCENT = 100
ALWAYS = float("inf")


class CopterMode(IntEnum):
    """ArduCopter custom-mode numbers."""
    STABILIZE = 0
    AUTO = 3
    GUIDED = 4
    LOITER = 5
    RTL = 6
    LAND = 9


SUPPORTED_MODES = (CopterMode.AUTO, CopterMode.GUIDED, CopterMode.LOITER, CopterMode.RTL, CopterMode.LAND)
ARMABLE_MODES = (CopterMode.GUIDED, CopterMode.LOITER, CopterMode.AUTO)


class Severity(IntEnum):
    """MAVLink STATUSTEXT severities."""
    CRITICAL = 2
    ERROR = 3
    WARNING = 4
    INFO = 6


class Phase(Enum):
    DISARMED = "disarmed"
    IDLE = "idle"
    SPOOL_UP = "spool"
    FLYING = "fly"
    RAMP_TO_IDLE = "ramp"
    IDLE_BEFORE_DISARM = "idle_end"


@dataclass
class Telemetry:
    latitude: float
    longitude: float
    height_above_home: float
    altitude_asl: float
    velocity_north: float
    velocity_east: float
    velocity_down: float
    roll: float
    pitch: float
    yaw: float
    heading_deg: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float
    battery_voltage: float
    ground_speed: float
    throttle_percent: float


class SimulatedVehicle:
    def __init__(self, frame, weights_path, home, mass_kg, log=print, timing=MotorSequenceTiming()):
        self.frame = frame
        self.pilot = NeuralPilot.from_checkpoint(weights_path)
        self.home = home if isinstance(home, HomePoint) else HomePoint(*home)
        self.log = log
        self.timing = timing
        self.parameters = default_parameters(frame, mass_kg)
        self.mission = []
        self.mode = CopterMode.GUIDED
        self.armed = False
        self.status_texts = []
        self.reached_items = []
        self.time = 0.0
        self.rebuild()

    def rebuild(self):
        """New airframe from the SIM_ parameters; simulated time keeps running."""
        parameters = self.parameters
        self.env = self.pilot.make_environment(
            1, randomize=False, seed=SIMULATION_SEED, noise=parameters["SIM_NOISE"] >= NOISE_ON, solid_ground=True,
            position_error_clip_m=float(parameters["NN_ERR_CLIP"]),
            nominal_mass_kg=float(np.clip(parameters["SIM_MASS"], *MASS_LIMITS_KG)),
            thrust_to_weight=float(parameters["SIM_TW_RATIO"]), start_altitude_m=float(self.home.altitude_asl))
        self.env.park_on_ground()
        self.env.airframe.rest_voltage[:] = parameters["SIM_BATT_VOLT"]
        self.env.state.battery_voltage[:] = parameters["SIM_BATT_VOLT"]
        self.period = self.env.control_period
        self.guidance = CarrotGuidance(float(self.env.airframe.length_scale[0]))
        self.landing_detector = LandingDetector(1, self.period, timing=self.timing)
        self.phase, self.phase_started = Phase.DISARMED, self.time
        self.goals, self.seq, self.mission_running, self.jumps_done = [], 0, False, {}
        self.pwm = np.full(self.env.motor_count, PWM_DISARMED_US)
        self.pwm_at_landing = self.pwm.copy()
        self.crashed = False
        self.takeoff_altitude = None
        self._apply_wind()

    def _apply_wind(self):
        direction = math.radians(self.parameters["SIM_WIND_DIR"])
        speed = self.parameters["SIM_WIND_SPD"]
        self.env.airframe.wind[0] = [-speed * math.cos(direction), speed * math.sin(direction), 0.0]

    def say(self, text, severity=Severity.INFO):
        self.status_texts.append((severity, text[:STATUS_TEXT_LIMIT]))
        self.log(f"[{self.time:7.1f}s] {text}")

    @property
    def position(self):
        return self.env.state.position[0].copy()

    @property
    def airborne(self):
        return not bool(self.env.state.on_ground[0])

    @property
    def carrot(self):
        return self.guidance.carrot

    def _set_phase(self, phase):
        self.phase, self.phase_started = phase, self.time

    def _time_in_phase(self):
        return self.time - self.phase_started

    def set_parameter(self, name, value):
        if name not in self.parameters:
            return False
        self.parameters[name] = float(value)
        if rebuilds_airframe(name):
            if name in WIND_PARAMETERS:
                self._apply_wind()
            elif not self.armed:
                self.rebuild()
                self.say(f"Airframe rebuilt: {self.parameters['SIM_MASS']:g} kg")
            else:
                self.say(f"{name} applies after disarm", Severity.WARNING)
        return True

    def set_mode(self, mode):
        if mode not in SUPPORTED_MODES:
            self.say(f"Mode {mode} not supported", Severity.WARNING)
            return False
        mode = CopterMode(mode)
        if mode == self.mode:
            return True
        self.mode = mode
        self.say(f"Mode {mode.name}")
        if self.phase is not Phase.FLYING:
            return True
        self.mission_running = False
        if mode in (CopterMode.LOITER, CopterMode.GUIDED):
            self.goals = [Goal.hold(self.position)]
            self.guidance.carrot = self.position
        elif mode is CopterMode.RTL:
            self.goals = self._return_to_launch_goals()
        elif mode is CopterMode.LAND:
            self.goals = [Goal.land(self.position)]
            self.guidance.carrot = self.position
        elif mode is CopterMode.AUTO:
            self.start_mission()
        return True

    def arm(self, force=False):
        if self.armed:
            return True
        if self.crashed:
            self.say("Arm refused: crashed, change a SIM_ parameter to reset", Severity.ERROR)
            return False
        if self.mode not in ARMABLE_MODES:
            self.say("Arm refused: use GUIDED, LOITER or AUTO", Severity.WARNING)
            return False
        self.armed = True
        self._set_phase(Phase.IDLE)
        self.say("Armed")
        return True

    def disarm(self, force=False):
        if not self.armed:
            return True
        if self.airborne and not force:
            self.say("Disarm refused: flying", Severity.WARNING)
            return False
        self.armed = False
        self.phase = Phase.DISARMED
        self.goals, self.mission_running = [], False
        self.say("Disarmed")
        return True

    def takeoff(self, altitude):
        if not self.armed:
            self.say("Take-off refused: not armed", Severity.WARNING)
            return False
        if self.phase is Phase.FLYING:
            here = self.position
            self.goals = [Goal.go_to([here[0], here[1], altitude]), Goal.hold()]
            return True
        self.takeoff_altitude = float(altitude)
        self._set_phase(Phase.SPOOL_UP)
        self.say(f"Take-off to {altitude:.0f} m")
        return True

    def go_to(self, latitude, longitude, height_above_home):
        if self.phase is not Phase.FLYING:
            self.say("Go-to refused: not flying", Severity.WARNING)
            return False
        if self.mode is not CopterMode.GUIDED:
            self.set_mode(CopterMode.GUIDED)
        self.goals = [Goal.go_to(self.home.to_local(latitude, longitude, height_above_home)), Goal.hold()]
        return True

    def _return_to_launch_goals(self):
        here = self.position
        altitude = max(here[2], self.parameters["RTL_ALT"] / CENTIMETRES_PER_METRE)
        slow_below = self.parameters["LAND_ALT_LOW"] / CENTIMETRES_PER_METRE
        return [Goal.go_to([here[0], here[1], altitude]), Goal.go_to([0, 0, altitude]),
                Goal.go_to([0, 0, min(altitude, slow_below)]), Goal.land([0, 0, 0])]

    def start_mission(self):
        if len(self.mission) < 2:
            self.say("No mission loaded", Severity.WARNING)
            return False
        if not self.armed:
            self.say("Mission start refused: not armed", Severity.WARNING)
            return False
        self.mode = CopterMode.AUTO
        self.mission_running, self.jumps_done, self.seq, self.goals = True, {}, 0, []
        self.say("Mission started")
        if self.phase is not Phase.FLYING:
            navigation = [item for item in self.mission[1:] if item.is_navigation]
            first_altitude = self._item_height(navigation[0]) if navigation else DEFAULT_TAKEOFF_ALTITUDE_M
            self.takeoff_altitude = max(first_altitude, MIN_TAKEOFF_ALTITUDE_M)
            self._set_phase(Phase.SPOOL_UP)
        self.next_mission_item()
        return True

    def _item_height(self, item):
        return self.home.height_above_home(item.z, item.frame)

    def next_mission_item(self):
        """Load the goals of the next mission item; called when the current goals are finished."""
        while True:
            self.seq += 1
            if self.seq >= len(self.mission):
                self.mission_running = False
                self.goals = [Goal.hold()]
                self.say("Mission complete, holding position")
                return
            if self._load_item(self.mission[self.seq]):
                return

    def _load_item(self, item):
        """Returns False for items that only change settings, so the next item is loaded straight away."""
        command = item.command
        here = self.goals[-1].position if self.goals and self.goals[-1].position is not None else self.carrot
        horizontal = self.home.to_local(item.x, item.y, 0)[:2] if item.has_position else np.array(here[:2])
        height = self._item_height(item)
        if command == mavlink.MAV_CMD_NAV_TAKEOFF:
            self.goals = [Goal.go_to([self.position[0], self.position[1], height])]
        elif command in (mavlink.MAV_CMD_NAV_WAYPOINT, mavlink.MAV_CMD_NAV_LOITER_TIME):
            self.goals = [Goal.go_to([horizontal[0], horizontal[1], height], hold_s=item.param1)]
        elif command == mavlink.MAV_CMD_NAV_LOITER_UNLIM:
            self.goals = [Goal.go_to([horizontal[0], horizontal[1], height], hold_s=ALWAYS)]
        elif command == mavlink.MAV_CMD_NAV_LOITER_TURNS:
            self.goals = self._circle_goals(item, horizontal, height, here)
        elif command == mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            self.goals = self._return_to_launch_goals()
        elif command == mavlink.MAV_CMD_NAV_LAND:
            slow_below = self.parameters["LAND_ALT_LOW"] / CENTIMETRES_PER_METRE
            self.goals = [Goal.go_to([horizontal[0], horizontal[1], here[2]]),
                          Goal.go_to([horizontal[0], horizontal[1], min(here[2], slow_below)]),
                          Goal.land([horizontal[0], horizontal[1], 0])]
        elif command == mavlink.MAV_CMD_DO_CHANGE_SPEED:
            if item.param2 > 0:
                self.parameters["WPNAV_SPEED"] = item.param2 * CENTIMETRES_PER_METRE
                self.say(f"Speed {item.param2:.1f} m/s")
            return False
        elif command == mavlink.MAV_CMD_DO_JUMP:
            self._jump(item)
            return False
        else:
            self.say(f"Item {self.seq}: command {command} not supported, skipped", Severity.WARNING)
            return False
        return True

    def _circle_goals(self, item, horizontal, height, here):
        radius = item.param3 if abs(item.param3) > MIN_CIRCLE_RADIUS_PARAM else self.parameters["CIRCLE_RADIUS"] / CENTIMETRES_PER_METRE
        center = np.array([horizontal[0], horizontal[1], height])
        outward = np.array(here[:2]) - center[:2]
        distance = np.linalg.norm(outward)
        outward = outward / distance if distance > MIN_DIRECTION_NORM else np.array([1.0, 0.0])
        edge = np.array([center[0] + abs(radius) * outward[0], center[1] + abs(radius) * outward[1], height])
        return [Goal.go_to(edge), Goal.circle(edge, center=center, radius=radius, turns=max(item.param1, 1.0))]

    def _jump(self, item):
        target, repeats = int(item.param1), int(item.param2)
        done = self.jumps_done.get(self.seq, 0)
        if (repeats < 0 or done < repeats) and 1 <= target < len(self.mission):
            self.jumps_done[self.seq] = done + 1
            self.seq = target - 1

    def jump_to_item(self, seq):
        self.seq = seq - 1
        if self.mission_running:
            self.goals = []
            self.next_mission_item()

    def _advance_goals(self):
        if not self.goals:
            return
        goal = self.goals[0]
        speeds = NavigationSpeeds.from_parameters(self.parameters)
        progress = self.guidance.advance(goal, speeds, self.position, self.time, self.period)
        if progress.reached_waypoint and self.mission_running and len(self.goals) == 1:
            self.reached_items.append(self.seq)
        if progress.finished:
            self._finish_goal()

    def _finish_goal(self):
        self.goals.pop(0)
        if self.goals:
            return
        if self.mission_running:
            self.next_mission_item()
        else:
            self.goals = [Goal.hold(self.carrot.copy())]

    @property
    def idle_pwm(self):
        return PWM_DISARMED_US + PWM_SPAN_US * float(np.clip(self.parameters["MOT_SPIN_ARM"], 0.0, MAX_IDLE_THROTTLE))

    def step(self):
        env = self.env
        landing = bool(self.goals) and self.goals[0].kind is GoalKind.LAND
        if self.phase is Phase.FLYING:
            self._advance_goals()
        elif self.phase in (Phase.DISARMED, Phase.IDLE, Phase.SPOOL_UP):
            self.guidance.carrot = self.position
        env.set_target(self.carrot, self.guidance.target_heading)
        action = clip_action(self.pilot.command(env.observe()))
        fixed_pwm = self._sequenced_pwm()
        if fixed_pwm is not None:
            action[0] = pwm_to_action(fixed_pwm)
        result = env.step(action, training=False)
        self.pwm = result.pwm[0].copy()
        self.time += self.period
        if self.phase is Phase.FLYING:
            self._watch_flight(landing)

    def _sequenced_pwm(self):
        """Motor PWM while the network is not in control (None while it is)."""
        timing, motor_count = self.timing, self.env.motor_count
        if self.phase is Phase.DISARMED:
            return np.full(motor_count, PWM_DISARMED_US)
        if self.phase is Phase.IDLE:
            if self._time_in_phase() > self.parameters["DISARM_DELAY"] > 0:
                self.say("Auto-disarm: idle on the ground")
                self.disarm()
            return np.full(motor_count, self.idle_pwm)
        if self.phase is Phase.SPOOL_UP:
            progress = min(1.0, self._time_in_phase() / timing.spool_up_s)
            pwm = np.full(motor_count, self.idle_pwm + progress * (PWM_SPOOL_END_US - self.idle_pwm))
            if progress >= 1.0:
                self._begin_flight()
            return pwm
        if self.phase is Phase.RAMP_TO_IDLE:
            progress = min(1.0, self._time_in_phase() / timing.ramp_to_idle_s)
            pwm = (1 - progress) * self.pwm_at_landing + progress * self.idle_pwm
            if progress >= 1.0:
                self._set_phase(Phase.IDLE_BEFORE_DISARM)
            return pwm
        if self.phase is Phase.IDLE_BEFORE_DISARM:
            if self._time_in_phase() >= timing.idle_before_disarm_s:
                self.say("Landed")
                self.disarm(force=True)
            return np.full(motor_count, self.idle_pwm)
        return None

    def _begin_flight(self):
        self.phase = Phase.FLYING
        self.guidance.carrot = self.position
        self.guidance.target_heading = float(euler_from_quaternion(self.env.state.attitude)[2][0])
        if not self.mission_running:
            here = self.position
            self.goals = [Goal.go_to([here[0], here[1], self.takeoff_altitude]), Goal.hold()]

    def _watch_flight(self, landing):
        state = self.env.state
        if upright_cosine(state.attitude)[0] < 0 or state.touchdown_speed[0] > CRASH_TOUCHDOWN_MPS:
            self.crashed = True
            self.say("CRASH: disarming", Severity.CRITICAL)
            self.disarm(force=True)
        elif landing:
            candidate = self.landing_detector.looks_landed(state.velocity[0, 2], state.position[0, 2], self.carrot[2],
                                                           self.pwm.mean())
            if self.landing_detector.update(np.array([candidate]))[0]:
                self._set_phase(Phase.RAMP_TO_IDLE)
                self.pwm_at_landing = self.pwm.copy()
                self.landing_detector.clear()
                self.mission_running = False

    def telemetry(self):
        state = self.env.state
        position, velocity, rates = state.position[0], state.velocity[0], state.body_rates[0]
        roll, pitch, yaw = (angle[0] for angle in euler_from_quaternion(state.attitude))
        latitude, longitude = self.home.to_global(position)
        return Telemetry(latitude=latitude, longitude=longitude, height_above_home=float(position[2]),
                         altitude_asl=float(position[2] + self.home.altitude_asl),
                         velocity_north=float(velocity[0]), velocity_east=float(-velocity[1]), velocity_down=float(-velocity[2]),
                         roll=float(roll), pitch=float(-pitch), yaw=float(wrap_angle(-yaw)),
                         heading_deg=math.degrees(-float(yaw)) % 360.0,
                         roll_rate=float(rates[0]), pitch_rate=float(-rates[1]), yaw_rate=float(-rates[2]),
                         battery_voltage=float(state.battery_voltage[0]), ground_speed=float(np.hypot(velocity[0], velocity[1])),
                         throttle_percent=float(np.clip((self.pwm.mean() - PWM_DISARMED_US) / PWM_SPAN_US * PERCENT, 0, PERCENT)))
