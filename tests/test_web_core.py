"""The browser simulator (web/core.js) must fly the same as the Python ground-link vehicle: one mission in both
with sensor noise off, tracks compared. Needs node."""
import json
import shutil
import subprocess

import numpy as np
import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from nn_drone.export.web_page import browser_network
from nn_drone.ground_link.geodesy import HomePoint
from nn_drone.ground_link.mission_items import MissionItem
from nn_drone.ground_link.vehicle import CopterMode, SimulatedVehicle
from nn_drone.paths import WEB_DIR, default_weights
from nn_drone.simulation.scaling import length_scale

HOME = HomePoint(28.5355, 77.3910, 0.0)
STEP_LIMIT = 400000
SAMPLE_EVERY_STEPS = 50
TRACK_TOLERANCE_LENGTHS = 0.5
RELATIVE = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

JS_RUNNER = r"""
const fs = require('fs'); const {Vehicle} = require(process.argv[2]); const c = JSON.parse(fs.readFileSync(process.argv[3]));
const v = new Vehicle(c.frame, c.w, {SIM_MASS: c.mass, SIM_NOISE: 0, SIM_WIND_SPD: 2, SIM_WIND_DIR: 45, HOME_ALT: 0}); v.idleTime = 0;
v.start([{cmd:'takeoff', alt:20}, {cmd:'waypoint', n:60, e:0, alt:30}, {cmd:'waypoint', n:60, e:60, alt:40, hold:2},
         {cmd:'circle', n:0.001, e:60, alt:30, radius:15, turns:1}, {cmd:'rtl'}]);
let k = 0, i = 0, maxd = 0;
while (v.armed && k < 400000) { v.step(); k++;
  if ((k - 1) % 50 === 0 && k > 1 && i < c.traj.length) { const r = c.traj[i++], p = v.pos;
    maxd = Math.max(maxd, Math.hypot(p[0] - r[0], p[1] - r[1], p[2] - r[2])); } }
console.log(JSON.stringify({steps: k, maxd: maxd, final: v.pos, crashed: v.crashed}));
"""


def python_track(frame, mass_kg):
    vehicle = SimulatedVehicle(frame, default_weights(frame), HOME, mass_kg, log=lambda *_: None)
    for name, value in dict(SIM_NOISE=0.0, SIM_WIND_SPD=2.0, SIM_WIND_DIR=45.0).items():
        vehicle.set_parameter(name, value)

    def item(command, north=0, east=0, height=0, param1=0, param3=0):
        latitude, longitude = vehicle.home.to_global([north, -east, 0]) if (north or east) else (0, 0)
        return MissionItem(seq=0, frame=RELATIVE, command=command, param1=param1, param3=param3, x=latitude, y=longitude, z=height)
    vehicle.mission = [item(mavlink.MAV_CMD_NAV_WAYPOINT), item(mavlink.MAV_CMD_NAV_TAKEOFF, height=20),
                       item(mavlink.MAV_CMD_NAV_WAYPOINT, 60, 0, 30), item(mavlink.MAV_CMD_NAV_WAYPOINT, 60, 60, 40, param1=2),
                       item(mavlink.MAV_CMD_NAV_LOITER_TURNS, 0.001, 60, 30, param1=1, param3=15),
                       item(mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)]
    vehicle.mode = CopterMode.AUTO
    vehicle.arm()
    vehicle.start_mission()
    track, steps = [], 0
    while vehicle.armed and steps < STEP_LIMIT:
        vehicle.step()
        steps += 1
        if steps % SAMPLE_EVERY_STEPS == 0:
            track.append([float(value) for value in vehicle.position])
    return track, steps


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
@pytest.mark.parametrize("frame, mass_kg", [("quad_x", 4.0), ("hex_x", 95.0)])
def test_browser_core_flies_like_python(frame, mass_kg, tmp_path):
    track, steps = python_track(frame, mass_kg)
    case = tmp_path / "case.json"
    case.write_text(json.dumps(dict(frame=frame, mass=mass_kg, w=browser_network(default_weights(frame)), traj=track)))
    runner = tmp_path / "run.js"
    runner.write_text(JS_RUNNER)
    output = subprocess.run(["node", str(runner), str(WEB_DIR / "core.js"), str(case)], check=True, capture_output=True, text=True)
    result = json.loads(output.stdout)
    assert result["steps"] == steps
    assert not result["crashed"]
    assert result["maxd"] < TRACK_TOLERANCE_LENGTHS * length_scale(mass_kg), f"largest track difference {result['maxd']:.3f} m"
