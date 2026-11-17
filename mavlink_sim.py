"""MAVLink front-end for the neural-network drone simulator.

Lets QGroundControl or Mission Planner connect to the simulator, change parameters, upload missions
and fly them. It presents itself as an ArduCopter-type vehicle so the ground station shows the usual
flight modes, but it is NOT ArduPilot: the vehicle is flown by the neural network in this repository.

    pip install pymavlink numpy
    python3 mavlink_sim.py                          # quad, sends to a ground station on this PC (UDP 14550)
    python3 mavlink_sim.py --frame hex_x --mass 4   # hexacopter, 4 kg
    python3 mavlink_sim.py --gcs 192.168.1.20:14550 --home 28.5355,77.3910,200

Supported modes: GUIDED, AUTO, LOITER, RTL, LAND.
Supported mission commands: TAKEOFF, WAYPOINT, LOITER_UNLIM, LOITER_TURNS (circle), LOITER_TIME,
RETURN_TO_LAUNCH, LAND, DO_CHANGE_SPEED, DO_JUMP.
"""
import argparse, math, os, socket, time
import numpy as np

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mav

from sim import MultirotorEnv, euler_from_quat, wrap
from ppo import PPO

# ArduCopter flight-mode numbers
STABILIZE, AUTO, GUIDED, LOITER, RTL, LAND = 0, 3, 4, 5, 6, 9
MODE_NAMES = {AUTO: "AUTO", GUIDED: "GUIDED", LOITER: "LOITER", RTL: "RTL", LAND: "LAND"}
PWM_OFF, PWM_SPOOL = 1000.0, 1450.0
T_SPOOL, T_CONFIRM, T_RAMP, T_IDLE_END = 2.0, 1.0, 1.0, 2.0
EARTH = 111319.5          # metres per degree of latitude

DEFAULT_WEIGHTS = {"quad_x": "quad_x_policy_v4.npz", "hex_x": "hex_x_policy_v1.npz"}


def default_params(frame, mass):
    rl = mass ** (1 / 6)                     # speed scale: bigger drones fly faster for the same network
    return {
        # --- simulated airframe (applied when disarmed) ---
        "SIM_MASS": mass,                    # kg, 1..100; airframe is scaled with mass
        "SIM_TW_RATIO": 2.8,                 # thrust-to-weight at full throttle
        "SIM_WIND_SPD": 0.0,                 # m/s
        "SIM_WIND_DIR": 0.0,                 # deg, direction the wind comes FROM
        "SIM_BATT_VOLT": 16.0,               # V at rest (4S: 14.4..16.8)
        "SIM_NOISE": 1.0,                    # 1 = sensor noise on
        "SIM_SPEEDUP": 1.0,                  # simulation speed relative to real time
        # --- network ---
        "NN_ERR_CLIP": 2.5,                  # position-error limit fed to the network (size-normalised m)
        # --- navigation, ArduCopter names and units ---
        "WPNAV_SPEED": round(250 * rl),      # cm/s horizontal
        "WPNAV_SPEED_UP": round(250 * rl),   # cm/s
        "WPNAV_SPEED_DN": round(150 * rl),   # cm/s
        "WPNAV_RADIUS": round(100 * mass ** (1 / 3)),   # cm, waypoint reached radius
        "LAND_SPEED": 50.0,                  # cm/s final descent
        "LAND_ALT_LOW": 1000.0,              # cm, below this height the descent uses LAND_SPEED
        "RTL_ALT": 1500.0,                   # cm, minimum height for the return leg
        "CIRCLE_RADIUS": 1000.0,             # cm, used when a LOITER_TURNS item gives no radius
        "WP_YAW_BEHAVIOR": 1.0,              # 0 = keep heading, 1 = face the next waypoint
        "MOT_SPIN_ARM": 0.15,                # idle throttle when armed (0.15 -> 1150 us)
        "DISARM_DELAY": 10.0,                # s, auto-disarm if armed on the ground and idle
        "AUTO_OPTIONS": 3.0,                 # allow arming and take-off in AUTO
        "FRAME_CLASS": 1.0 if frame == "quad_x" else 2.0,
        "FRAME_TYPE": 1.0,                   # X
        "SYSID_THISMAV": 1.0,
    }


class Goal:
    """A point the moving target heads for, plus what to do there."""
    def __init__(self, kind, pos=None, hold=0.0, center=None, radius=0.0, turns=0.0, speed_dn=None):
        self.kind = kind            # goto | hold | circle | land
        self.pos = None if pos is None else np.array(pos, float)
        self.hold = hold; self.center = center; self.radius = radius; self.turns = turns
        self.speed_dn = speed_dn; self.t_arrived = None; self.angle0 = None; self.angle = 0.0


class Vehicle:
    def __init__(self, frame, weights, home, mass, log=print):
        self.frame, self.weights_path, self.log = frame, weights, log
        self.home_lat, self.home_lon, self.home_alt = home
        self.P = default_params(frame, mass)
        self.mission = []           # list of dicts; item 0 is home (ArduPilot convention)
        self.mode = GUIDED; self.armed = False
        self.statustexts = []       # (severity, text) waiting to be sent
        self.reached = []           # mission seq numbers reached, waiting to be sent
        self.rebuild()

    # ------------------------------------------------------------------ simulation set-up
    def rebuild(self):
        P = self.P
        self.env = MultirotorEnv(1, frame=self.frame, randomize=False, noise=P["SIM_NOISE"] >= 0.5, ground=True,
                                 err_clip=float(P["NN_ERR_CLIP"]), mass=float(np.clip(P["SIM_MASS"], 0.2, 500)),
                                 thrust_to_weight=float(P["SIM_TW_RATIO"]), alt0=float(self.home_alt), seed=1)
        e = self.env
        self.agent = PPO(e.obs_dim, e.act_dim); self.agent.load(self.weights_path)
        e.reset_idx(np.arange(1), hard=False)
        e.mot[:] = 0.0; e.prev_a[:] = -1.0; e.on_ground[:] = True
        e.v0[:] = P["SIM_BATT_VOLT"]; e.volt[:] = P["SIM_BATT_VOLT"]
        self.dt = float(e.dt_env[0]); self.L = float(e.arm_s[0])
        self.t = getattr(self, "t", 0.0)       # simulated time keeps running across rebuilds
        self.phase = "disarmed"; self.t_phase = self.t
        self.carrot = np.zeros(3); self.yaw_t = 0.0
        self.goals = []; self.seq = 0; self.mission_running = False; self.jumps = {}
        self.confirm = 0.0; self.pwm = np.full(e.M, PWM_OFF); self.pwm_ramp0 = self.pwm.copy()
        self.crashed = False; self.takeoff_alt = None
        self._set_wind()

    def _set_wind(self):
        d = math.radians(self.P["SIM_WIND_DIR"]); s = self.P["SIM_WIND_SPD"]
        # wind FROM direction d (compass) blows toward d + 180; x = north, y = west
        self.env.wind[0] = [-s * math.cos(d), s * math.sin(d), 0.0]

    def say(self, text, sev=6):
        self.statustexts.append((sev, text[:50])); self.log(f"[{self.t:7.1f}s] {text}")

    # ------------------------------------------------------------------ coordinates
    def to_local(self, lat, lon, alt_rel):
        x = (lat - self.home_lat) * EARTH
        y = -(lon - self.home_lon) * EARTH * math.cos(math.radians(self.home_lat))
        return np.array([x, y, alt_rel])

    def to_global(self, p):
        lat = self.home_lat + p[0] / EARTH
        lon = self.home_lon - p[1] / (EARTH * math.cos(math.radians(self.home_lat)))
        return lat, lon

    def item_alt(self, it):
        # frames 0 / 5 are altitude above sea level, everything else is treated as relative to home
        return it["z"] - self.home_alt if it["frame"] in (0, 5) else it["z"]

    @property
    def pos(self):
        return self.env.p[0].copy()

    @property
    def airborne(self):
        return not bool(self.env.on_ground[0])

    # ------------------------------------------------------------------ commands
    def set_param(self, name, value):
        if name not in self.P:
            return False
        self.P[name] = float(value)
        if name.startswith("SIM_") and name != "SIM_SPEEDUP" or name in ("NN_ERR_CLIP",):
            if name in ("SIM_WIND_SPD", "SIM_WIND_DIR"):
                self._set_wind()
            elif not self.armed:
                old = self.P.copy(); self.rebuild(); self.P = old
                self.say(f"Airframe rebuilt: {self.P['SIM_MASS']:g} kg")
            else:
                self.say(f"{name} applies after disarm", 4)
        return True

    def set_mode(self, mode):
        if mode not in MODE_NAMES:
            self.say(f"Mode {mode} not supported", 4); return False
        if mode == self.mode:
            return True
        self.mode = mode; self.say(f"Mode {MODE_NAMES[mode]}")
        if self.phase != "fly":
            return True
        self.mission_running = False
        if mode in (LOITER, GUIDED):
            self.goals = [Goal("hold", self.pos)]; self.carrot = self.pos
        elif mode == RTL:
            self.goals = self._rtl_goals()
        elif mode == LAND:
            self.goals = [Goal("land", self.pos)]; self.carrot = self.pos
        elif mode == AUTO:
            self.start_mission()
        return True

    def arm(self, force=False):
        if self.armed:
            return True
        if self.crashed:
            self.say("Arm refused: crashed, change a SIM_ parameter to reset", 3); return False
        if self.mode not in (GUIDED, LOITER, AUTO):
            self.say("Arm refused: use GUIDED, LOITER or AUTO", 4); return False
        self.armed = True; self.phase = "idle"; self.t_phase = self.t; self.say("Armed")
        return True

    def disarm(self, force=False):
        if not self.armed:
            return True
        if self.airborne and not force:
            self.say("Disarm refused: flying", 4); return False
        self.armed = False; self.phase = "disarmed"; self.goals = []; self.mission_running = False
        self.say("Disarmed")
        return True

    def takeoff(self, alt):
        if not self.armed:
            self.say("Take-off refused: not armed", 4); return False
        if self.phase == "fly":
            self.goals = [Goal("goto", [self.pos[0], self.pos[1], alt]), Goal("hold", None)]
            return True
        self.takeoff_alt = float(alt); self.phase = "spool"; self.t_phase = self.t
        self.say(f"Take-off to {alt:.0f} m")
        return True

    def goto(self, lat, lon, alt_rel):
        if self.phase != "fly":
            self.say("Go-to refused: not flying", 4); return False
        if self.mode != GUIDED:
            self.set_mode(GUIDED)
        self.goals = [Goal("goto", self.to_local(lat, lon, alt_rel)), Goal("hold", None)]
        return True

    def _rtl_goals(self):
        p = self.pos; alt = max(p[2], self.P["RTL_ALT"] / 100.0)
        low = self.P["LAND_ALT_LOW"] / 100.0
        return [Goal("goto", [p[0], p[1], alt]), Goal("goto", [0, 0, alt]), Goal("goto", [0, 0, min(alt, low)]),
                Goal("land", [0, 0, 0])]

    # ------------------------------------------------------------------ mission
    def start_mission(self):
        if len(self.mission) < 2:
            self.say("No mission loaded", 4); return False
        if not self.armed:
            self.say("Mission start refused: not armed", 4); return False
        self.mode = AUTO; self.mission_running = True; self.jumps = {}
        self.seq = 0; self.goals = []
        self.say("Mission started")
        if self.phase != "fly":
            nav = [it for it in self.mission[1:] if it["command"] < 100]
            alt = self.item_alt(nav[0]) if nav else 10.0
            self.takeoff_alt = max(alt, 2.0); self.phase = "spool"; self.t_phase = self.t
        self._next_item()
        return True

    def _next_item(self):
        """Load goals for the next mission item; called when the current goals are finished."""
        while True:
            self.seq += 1
            if self.seq >= len(self.mission):
                self.mission_running = False
                self.goals = [Goal("hold", None)]
                self.say("Mission complete, holding position")
                return
            it = self.mission[self.seq]; c = it["command"]
            has_pos = abs(it["x"]) > 1e-9 or abs(it["y"]) > 1e-9
            here = self.goals[-1].pos if self.goals and self.goals[-1].pos is not None else self.carrot
            xy = self.to_local(it["x"], it["y"], 0)[:2] if has_pos else np.array(here[:2])
            alt = self.item_alt(it)
            if c == mav.MAV_CMD_NAV_TAKEOFF:
                self.goals = [Goal("goto", [self.pos[0], self.pos[1], alt])]
            elif c == mav.MAV_CMD_NAV_WAYPOINT:
                self.goals = [Goal("goto", [xy[0], xy[1], alt], hold=it["p1"])]
            elif c == mav.MAV_CMD_NAV_LOITER_TIME:
                self.goals = [Goal("goto", [xy[0], xy[1], alt], hold=it["p1"])]
            elif c == mav.MAV_CMD_NAV_LOITER_UNLIM:
                self.goals = [Goal("goto", [xy[0], xy[1], alt], hold=1e12)]
            elif c == mav.MAV_CMD_NAV_LOITER_TURNS:
                r = it["p3"] if abs(it["p3"]) > 0.1 else self.P["CIRCLE_RADIUS"] / 100.0
                cen = np.array([xy[0], xy[1], alt]); d = np.array(here[:2]) - cen[:2]
                n = np.linalg.norm(d); d = d / n if n > 1e-3 else np.array([1.0, 0.0])
                edge = np.array([cen[0] + abs(r) * d[0], cen[1] + abs(r) * d[1], alt])
                self.goals = [Goal("goto", edge), Goal("circle", edge, center=cen, radius=r, turns=max(it["p1"], 1.0))]
            elif c == mav.MAV_CMD_NAV_RETURN_TO_LAUNCH:
                self.goals = self._rtl_goals()
            elif c == mav.MAV_CMD_NAV_LAND:
                low = self.P["LAND_ALT_LOW"] / 100.0
                top = [xy[0], xy[1], here[2]]
                self.goals = [Goal("goto", top), Goal("goto", [xy[0], xy[1], min(here[2], low)]), Goal("land", [xy[0], xy[1], 0])]
            elif c == mav.MAV_CMD_DO_CHANGE_SPEED:
                if it["p2"] > 0:
                    self.P["WPNAV_SPEED"] = it["p2"] * 100.0; self.say(f"Speed {it['p2']:.1f} m/s")
                continue
            elif c == mav.MAV_CMD_DO_JUMP:
                target, repeat = int(it["p1"]), int(it["p2"])
                done = self.jumps.get(self.seq, 0)
                if (repeat < 0 or done < repeat) and 1 <= target < len(self.mission):
                    self.jumps[self.seq] = done + 1; self.seq = target - 1
                continue
            else:
                self.say(f"Item {self.seq}: command {c} not supported, skipped", 4)
                continue
            return

    # ------------------------------------------------------------------ one control step
    def _advance_goals(self):
        """Move the target point (carrot) toward the current goal; returns nothing."""
        P, dt = self.P, self.dt
        if not self.goals:
            return
        g = self.goals[0]
        if g.pos is None:
            g.pos = self.carrot.copy()
        vh, vu, vd = P["WPNAV_SPEED"] / 100, P["WPNAV_SPEED_UP"] / 100, P["WPNAV_SPEED_DN"] / 100
        if g.kind in ("goto", "hold"):
            d = g.pos - self.carrot
            T = max(np.hypot(d[0], d[1]) / max(vh, 0.1), d[2] / max(vu, 0.1) if d[2] > 0 else -d[2] / max(vd, 0.1))
            if T > dt:
                self.carrot = self.carrot + d * (dt / T)
                if P["WP_YAW_BEHAVIOR"] >= 0.5 and np.hypot(d[0], d[1]) > 3.0 * self.L:
                    self.yaw_t = math.atan2(d[1], d[0])
            else:
                self.carrot = g.pos.copy()
                near = np.linalg.norm(self.pos - g.pos) < P["WPNAV_RADIUS"] / 100.0
                if g.kind == "goto" and near:
                    if g.t_arrived is None:
                        g.t_arrived = self.t
                        if self.mission_running and len(self.goals) == 1:
                            self.reached.append(self.seq)
                    if self.t - g.t_arrived >= g.hold:
                        self._goal_done()
        elif g.kind == "circle":
            if g.angle0 is None:
                e = self.carrot - g.center; g.angle0 = math.atan2(e[1], e[0]); g.angle = 0.0
            sgn = -1.0 if g.radius > 0 else 1.0          # positive radius = clockwise seen from above
            g.angle += vh * dt / abs(g.radius)
            a = g.angle0 + sgn * g.angle
            self.carrot = np.array([g.center[0] + abs(g.radius) * math.cos(a), g.center[1] + abs(g.radius) * math.sin(a), g.center[2]])
            if P["WP_YAW_BEHAVIOR"] >= 0.5:
                self.yaw_t = a + sgn * math.pi / 2
            if g.angle >= 2 * math.pi * g.turns:
                if self.mission_running:
                    self.reached.append(self.seq)
                self._goal_done()
        elif g.kind == "land":
            self.carrot[:2] = g.pos[:2]
            rate = P["LAND_SPEED"] / 100.0 if self.pos[2] < P["LAND_ALT_LOW"] / 100.0 else vd
            if self.carrot[2] > -3.0 * self.L:
                self.carrot[2] -= rate * dt

    def _goal_done(self):
        self.goals.pop(0)
        if not self.goals:
            if self.mission_running:
                self._next_item()
            else:
                self.goals = [Goal("hold", self.carrot.copy())]

    def step(self):
        e, P = self.env, self.P
        idle = 1000.0 + 1000.0 * float(np.clip(P["MOT_SPIN_ARM"], 0.0, 0.3))
        landing = bool(self.goals) and self.goals[0].kind == "land"
        if self.phase == "fly":
            self._advance_goals()
        elif self.phase in ("disarmed", "idle", "spool"):
            self.carrot = np.array([self.pos[0], self.pos[1], self.pos[2]])
        e.target[0] = self.carrot; e.target_yaw[0] = self.yaw_t
        a, _ = self.agent.act(e.observe(), deterministic=True)
        a = np.clip(a, -1, 1).astype(np.float64)
        ph = self.phase
        if ph == "disarmed":
            pwm = np.full(e.M, PWM_OFF)
        elif ph == "idle":
            pwm = np.full(e.M, idle)
            if self.t - self.t_phase > P["DISARM_DELAY"] > 0:
                self.say("Auto-disarm: idle on the ground"); self.disarm()
        elif ph == "spool":
            f = min(1.0, (self.t - self.t_phase) / T_SPOOL)
            pwm = np.full(e.M, idle + f * (PWM_SPOOL - idle))
            if f >= 1.0:
                self.phase = "fly"; self.carrot = self.pos
                _, _, yaw = euler_from_quat(e.q); self.yaw_t = float(yaw[0])
                if not self.mission_running:
                    self.goals = [Goal("goto", [self.pos[0], self.pos[1], self.takeoff_alt]), Goal("hold", None)]
        elif ph == "ramp":
            f = min(1.0, (self.t - self.t_phase) / T_RAMP)
            pwm = (1 - f) * self.pwm_ramp0 + f * idle
            if f >= 1.0:
                self.phase = "idle_end"; self.t_phase = self.t
        elif ph == "idle_end":
            pwm = np.full(e.M, idle)
            if self.t - self.t_phase >= T_IDLE_END:
                self.say("Landed"); self.disarm(force=True)
        else:
            pwm = None
        if pwm is not None:
            a[0] = (pwm - 1500.0) / 500.0
        _, _, _, _, _, info = e.step(a, auto_target=False)
        self.pwm = info["pwm"][0].copy(); self.t += self.dt

        if self.phase == "fly":
            up = 1 - 2 * (e.q[0, 1] ** 2 + e.q[0, 2] ** 2)
            if up < 0 or e.impact[0] > 3.0:
                self.crashed = True; self.say("CRASH: disarming", 2); self.disarm(force=True)
            elif landing:
                # landed = not moving vertically, target well below, throttle already under lift-off level
                cond = abs(e.v[0, 2]) < 0.15 and self.carrot[2] < e.p[0, 2] - 0.5 and self.pwm.mean() < PWM_SPOOL
                self.confirm = self.confirm + self.dt if cond else 0.0
                if self.confirm >= T_CONFIRM:
                    self.phase = "ramp"; self.t_phase = self.t; self.pwm_ramp0 = self.pwm.copy(); self.confirm = 0.0
                    self.mission_running = False

    # ------------------------------------------------------------------ telemetry values
    def telemetry(self):
        e = self.env; p = e.p[0]; v = e.v[0]
        r, pt, y = euler_from_quat(e.q)
        lat, lon = self.to_global(p)
        hdg = math.degrees(-float(y[0])) % 360.0
        return dict(lat=lat, lon=lon, alt_rel=float(p[2]), alt_amsl=float(p[2] + self.home_alt),
                    vn=float(v[0]), ve=float(-v[1]), vd=float(-v[2]), roll=float(r[0]), pitch=float(-pt[0]),
                    yaw=float(wrap(-y[0])), hdg=hdg, rollspeed=float(e.w[0, 0]), pitchspeed=float(-e.w[0, 1]),
                    yawspeed=float(-e.w[0, 2]), volt=float(e.volt[0]), gs=float(np.hypot(v[0], v[1])),
                    throttle=float(np.clip((self.pwm.mean() - 1000) / 10, 0, 100)))


# ====================================================================== MAVLink server
class Server:
    def __init__(self, veh, gcs, out=print):
        self.v = veh
        # pymavlink's "udpout" binds the destination port locally on Windows, which collides with a
        # ground station on the same PC. Instead bind an ephemeral local port and send to the GCS.
        host, port = gcs.rsplit(":", 1)
        self.m = mavutil.mavlink_connection("udpin:0.0.0.0:0", source_system=int(veh.P["SYSID_THISMAV"]),
                                            source_component=1, dialect="ardupilotmega")
        dest = (socket.gethostbyname(host), int(port))
        self.m.clients.add(dest); self.m.clients_last_alive[dest] = time.time()
        self.mav = self.m.mav
        self.param_queue = []; self.last = {}
        self.up = None            # mission upload in progress: dict(count, items, next, t)
        self.gcs_sys, self.gcs_comp = 255, 190

    # ---------------------------------------------------------------- helpers
    def due(self, key, period):
        t = self.v.t
        if t - self.last.get(key, -1e9) >= period:
            self.last[key] = t; return True
        return False

    def ms(self):
        return int(self.v.t * 1000) & 0xFFFFFFFF

    def ack(self, cmd, ok=True, result=None):
        res = result if result is not None else (mav.MAV_RESULT_ACCEPTED if ok else mav.MAV_RESULT_FAILED)
        self.mav.command_ack_send(cmd, res)

    def send_param(self, name):
        names = list(self.v.P)
        self.mav.param_value_send(name.encode(), float(self.v.P[name]), mav.MAV_PARAM_TYPE_REAL32,
                                  len(names), names.index(name))

    def send_home(self):
        v = self.v
        self.mav.home_position_send(int(v.home_lat * 1e7), int(v.home_lon * 1e7), int(v.home_alt * 1000),
                                    0, 0, 0, [1, 0, 0, 0], 0, 0, 0)

    def send_version(self):
        caps = (mav.MAV_PROTOCOL_CAPABILITY_MISSION_INT | mav.MAV_PROTOCOL_CAPABILITY_COMMAND_INT |
                mav.MAV_PROTOCOL_CAPABILITY_PARAM_FLOAT | mav.MAV_PROTOCOL_CAPABILITY_MAVLINK2 |
                mav.MAV_PROTOCOL_CAPABILITY_SET_POSITION_TARGET_GLOBAL_INT)
        ver = (4 << 24) | (5 << 16) | (0 << 8) | 255          # reported as 4.5.0 so ground stations enable Copter features
        self.mav.autopilot_version_send(caps, ver, 0, 0, 0, [0] * 8, [0] * 8, [0] * 8, 0, 0, 0)

    def home_item(self):
        v = self.v
        return dict(seq=0, frame=0, command=16, current=0, autocontinue=1, p1=0, p2=0, p3=0, p4=0,
                    x=v.home_lat, y=v.home_lon, z=v.home_alt)

    def send_item(self, seq, as_int):
        items = self.v.mission if self.v.mission else [self.home_item()]
        it = items[seq]; cur = 1 if seq == self.v.seq and self.v.mission_running else 0
        if as_int:
            self.mav.mission_item_int_send(self.gcs_sys, self.gcs_comp, seq, it["frame"], it["command"], cur,
                                           it["autocontinue"], it["p1"], it["p2"], it["p3"], it["p4"],
                                           int(round(it["x"] * 1e7)), int(round(it["y"] * 1e7)), it["z"], 0)
        else:
            self.mav.mission_item_send(self.gcs_sys, self.gcs_comp, seq, it["frame"], it["command"], cur,
                                       it["autocontinue"], it["p1"], it["p2"], it["p3"], it["p4"],
                                       it["x"], it["y"], it["z"], 0)

    # ---------------------------------------------------------------- incoming
    def handle(self, msg):
        v = self.v; t = msg.get_type()
        if t == "BAD_DATA":
            return
        if t == "HEARTBEAT":
            if msg.type == mav.MAV_TYPE_GCS:
                self.gcs_sys, self.gcs_comp = msg.get_srcSystem(), msg.get_srcComponent()
        elif t == "PARAM_REQUEST_LIST":
            self.param_queue = list(v.P)
        elif t == "PARAM_REQUEST_READ":
            name = msg.param_id.rstrip("\x00") if isinstance(msg.param_id, str) else msg.param_id.decode().rstrip("\x00")
            names = list(v.P)
            if name not in v.P and 0 <= msg.param_index < len(names):
                name = names[msg.param_index]
            if name in v.P:
                self.send_param(name)
        elif t == "PARAM_SET":
            name = msg.param_id.rstrip("\x00")
            if v.set_param(name, msg.param_value):
                self.send_param(name)
        elif t == "MISSION_COUNT":
            if getattr(msg, "mission_type", 0) != 0:
                self.mav.mission_ack_send(self.gcs_sys, self.gcs_comp, mav.MAV_MISSION_ACCEPTED, msg.mission_type)
            elif msg.count == 0:
                v.mission = []; self.mav.mission_ack_send(self.gcs_sys, self.gcs_comp, mav.MAV_MISSION_ACCEPTED, 0)
            else:
                self.up = dict(count=msg.count, items=[], t=time.time())
                self.mav.mission_request_int_send(self.gcs_sys, self.gcs_comp, 0, 0)
        elif t in ("MISSION_ITEM_INT", "MISSION_ITEM"):
            sc = 1e-7 if t == "MISSION_ITEM_INT" else 1.0
            it = dict(seq=msg.seq, frame=msg.frame, command=msg.command, current=msg.current, autocontinue=msg.autocontinue,
                      p1=msg.param1, p2=msg.param2, p3=msg.param3, p4=msg.param4, x=msg.x * sc, y=msg.y * sc, z=msg.z)
            if msg.current == 2 and self.up is None:            # "fly to here" in GUIDED
                v.goto(it["x"], it["y"], v.item_alt(it))
                self.mav.mission_ack_send(self.gcs_sys, self.gcs_comp, mav.MAV_MISSION_ACCEPTED, 0)
            elif self.up is not None and msg.seq == len(self.up["items"]):
                self.up["items"].append(it); self.up["t"] = time.time()
                if len(self.up["items"]) == self.up["count"]:
                    v.mission = self.up["items"]; self.up = None
                    self.mav.mission_ack_send(self.gcs_sys, self.gcs_comp, mav.MAV_MISSION_ACCEPTED, 0)
                    v.say(f"Mission received: {len(v.mission) - 1} items")
                else:
                    self.mav.mission_request_int_send(self.gcs_sys, self.gcs_comp, len(self.up["items"]), 0)
        elif t == "MISSION_REQUEST_LIST":
            if getattr(msg, "mission_type", 0) != 0:
                self.mav.mission_count_send(self.gcs_sys, self.gcs_comp, 0, msg.mission_type)
            else:
                self.mav.mission_count_send(self.gcs_sys, self.gcs_comp, max(len(v.mission), 1), 0)
        elif t in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            if getattr(msg, "mission_type", 0) == 0 and msg.seq < max(len(v.mission), 1):
                self.send_item(msg.seq, t == "MISSION_REQUEST_INT")
        elif t == "MISSION_CLEAR_ALL":
            v.mission = []; self.mav.mission_ack_send(self.gcs_sys, self.gcs_comp, mav.MAV_MISSION_ACCEPTED, 0)
        elif t == "MISSION_SET_CURRENT":
            if 0 < msg.seq < len(v.mission):
                v.seq = msg.seq - 1
                if v.mission_running:
                    v.goals = []; v._next_item()
        elif t == "SET_MODE":
            v.set_mode(int(msg.custom_mode))
        elif t == "SET_POSITION_TARGET_GLOBAL_INT":
            alt = msg.alt - v.home_alt if msg.coordinate_frame in (0, 5) else msg.alt
            v.goto(msg.lat_int / 1e7, msg.lon_int / 1e7, alt)
        elif t == "TIMESYNC":
            if msg.tc1 == 0:
                self.mav.timesync_send(int(time.time() * 1e9), msg.ts1)
        elif t == "COMMAND_INT":
            if msg.command == mav.MAV_CMD_DO_REPOSITION:
                alt = msg.z - v.home_alt if msg.frame in (0, 5) else msg.z
                self.ack(msg.command, v.goto(msg.x / 1e7, msg.y / 1e7, alt))
            else:
                self.ack(msg.command, result=mav.MAV_RESULT_UNSUPPORTED)
        elif t == "COMMAND_LONG":
            self.command(msg)

    def command(self, msg):
        v = self.v; c = msg.command
        if c == mav.MAV_CMD_COMPONENT_ARM_DISARM:
            force = abs(msg.param2 - 21196) < 1
            self.ack(c, v.arm(force) if msg.param1 >= 0.5 else v.disarm(force))
        elif c == mav.MAV_CMD_NAV_TAKEOFF:
            self.ack(c, v.takeoff(msg.param7))
        elif c == mav.MAV_CMD_DO_SET_MODE:
            self.ack(c, v.set_mode(int(msg.param2)))
        elif c == mav.MAV_CMD_NAV_LAND:
            self.ack(c, v.set_mode(LAND))
        elif c == mav.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            self.ack(c, v.set_mode(RTL))
        elif c == mav.MAV_CMD_MISSION_START:
            self.ack(c, v.start_mission())
        elif c == mav.MAV_CMD_DO_CHANGE_SPEED:
            if msg.param2 > 0:
                v.P["WPNAV_SPEED"] = msg.param2 * 100.0; self.send_param("WPNAV_SPEED")
            self.ack(c, True)
        elif c == mav.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES:
            self.ack(c, True); self.send_version()
        elif c == mav.MAV_CMD_REQUEST_MESSAGE:
            mid = int(msg.param1)
            if mid == mav.MAVLINK_MSG_ID_AUTOPILOT_VERSION:
                self.ack(c, True); self.send_version()
            elif mid == mav.MAVLINK_MSG_ID_HOME_POSITION:
                self.ack(c, True); self.send_home()
            else:
                self.ack(c, result=mav.MAV_RESULT_UNSUPPORTED)
        elif c == mav.MAV_CMD_GET_HOME_POSITION:
            self.ack(c, True); self.send_home()
        elif c in (mav.MAV_CMD_SET_MESSAGE_INTERVAL, mav.MAV_CMD_DO_SET_HOME):
            self.ack(c, True)
        else:
            self.ack(c, result=mav.MAV_RESULT_UNSUPPORTED)

    # ---------------------------------------------------------------- outgoing
    def send_telemetry(self):
        v = self.v; P = v.P
        if self.due("hb", 1.0):
            base = (mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mav.MAV_MODE_FLAG_STABILIZE_ENABLED |
                    mav.MAV_MODE_FLAG_MANUAL_INPUT_ENABLED | mav.MAV_MODE_FLAG_GUIDED_ENABLED)
            if v.armed:
                base |= mav.MAV_MODE_FLAG_SAFETY_ARMED
            typ = mav.MAV_TYPE_QUADROTOR if v.frame == "quad_x" else mav.MAV_TYPE_HEXAROTOR
            st = mav.MAV_STATE_CRITICAL if v.crashed else (mav.MAV_STATE_ACTIVE if v.armed else mav.MAV_STATE_STANDBY)
            self.mav.heartbeat_send(typ, mav.MAV_AUTOPILOT_ARDUPILOTMEGA, base, v.mode, st)
        tm = None
        if self.due("fast", 0.1):
            tm = v.telemetry()
            self.mav.attitude_send(self.ms(), tm["roll"], tm["pitch"], tm["yaw"], tm["rollspeed"], tm["pitchspeed"], tm["yawspeed"])
            self.mav.global_position_int_send(self.ms(), int(tm["lat"] * 1e7), int(tm["lon"] * 1e7), int(tm["alt_amsl"] * 1000),
                                              int(tm["alt_rel"] * 1000), int(tm["vn"] * 100), int(tm["ve"] * 100),
                                              int(tm["vd"] * 100), int(tm["hdg"] * 100) % 36000)
            self.mav.vfr_hud_send(tm["gs"], tm["gs"], int(tm["hdg"]), int(tm["throttle"]), tm["alt_amsl"], -tm["vd"])
            pw = [int(x) for x in v.pwm] + [0] * 16
            self.mav.servo_output_raw_send(self.ms() * 1000 & 0xFFFFFFFF, 0, *pw[:8])
        if self.due("slow", 0.5):
            tm = tm or v.telemetry()
            self.mav.gps_raw_int_send(int(v.t * 1e6), 3, int(tm["lat"] * 1e7), int(tm["lon"] * 1e7), int(tm["alt_amsl"] * 1000),
                                      80, 120, int(tm["gs"] * 100), int(tm["hdg"] * 100) % 36000, 14)
            sens = 0x0FFFFFFF & ~(1 << 16)
            self.mav.sys_status_send(sens, sens, sens, 200, int(tm["volt"] * 1000), -1, -1, 0, 0, 0, 0, 0, 0)
            landed = mav.MAV_LANDED_STATE_IN_AIR if v.airborne else mav.MAV_LANDED_STATE_ON_GROUND
            self.mav.extended_sys_state_send(mav.MAV_VTOL_STATE_UNDEFINED, landed)
            self.mav.mission_current_send(v.seq if v.mission_running else 0)
            self.mav.ekf_status_report_send(831, 0.05, 0.05, 0.05, 0.05, 0.05)
            d = v.carrot - v.pos
            brg = int(math.degrees(math.atan2(-d[1], d[0])) % 360)
            self.mav.nav_controller_output_send(0, 0, brg, brg, int(min(np.hypot(d[0], d[1]), 65535)), float(d[2]), 0, 0)
        if self.due("home", 5.0):
            self.send_home()
            self.mav.gps_global_origin_send(int(v.home_lat * 1e7), int(v.home_lon * 1e7), int(v.home_alt * 1000))
        for _ in range(min(len(self.param_queue), 10)):
            self.send_param(self.param_queue.pop(0))
        while v.statustexts:
            sev, text = v.statustexts.pop(0)
            self.mav.statustext_send(sev, text.encode())
        while v.reached:
            self.mav.mission_item_reached_send(v.reached.pop(0))
        if self.up is not None and time.time() - self.up["t"] > 1.0:      # re-request a lost mission item
            self.up["t"] = time.time()
            self.mav.mission_request_int_send(self.gcs_sys, self.gcs_comp, len(self.up["items"]), 0)

    def poll(self):
        while True:
            try:
                msg = self.m.recv_match(blocking=False)
            except ConnectionResetError:
                # Windows reports "port unreachable" from an earlier send (ground station not open yet,
                # or closed) as WinError 10054 on the next receive. Harmless for UDP: keep going.
                continue
            if msg is None:
                break
            self.handle(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame", default="quad_x", choices=["quad_x", "hex_x"])
    ap.add_argument("--weights", default="")
    ap.add_argument("--mass", type=float, default=1.0, help="kg")
    ap.add_argument("--gcs", default="127.0.0.1:14550", help="ground station address ip:port")
    ap.add_argument("--home", default="28.5355,77.3910,200", help="lat,lon,altitude above sea level (m)")
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many real seconds (0 = run until Ctrl-C)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, DEFAULT_WEIGHTS[args.frame])
    home = tuple(float(x) for x in args.home.split(","))
    veh = Vehicle(args.frame, weights, home, args.mass)
    veh.P["SIM_SPEEDUP"] = args.speedup
    srv = Server(veh, args.gcs)
    veh.say("NN drone simulator (not ArduPilot)")
    veh.say(f"{args.frame} {args.mass:g} kg ready")
    print(f"Sending MAVLink to {args.gcs}. Open QGroundControl or Mission Planner (UDP 14550). Ctrl-C to stop.")
    t_real0 = time.time(); t_sim0 = veh.t
    try:
        while True:
            srv.poll()
            veh.step()
            srv.send_telemetry()
            sp = max(float(veh.P["SIM_SPEEDUP"]), 0.05)
            lag = (veh.t - t_sim0) / sp - (time.time() - t_real0)
            if lag > 0.002:
                time.sleep(min(lag, 0.05))
            elif lag < -1.0:                       # cannot keep up: do not try to catch up
                t_real0 = time.time(); t_sim0 = veh.t
            if args.duration and time.time() - START > args.duration:
                break
    except KeyboardInterrupt:
        pass


START = time.time()
if __name__ == "__main__":
    main()
