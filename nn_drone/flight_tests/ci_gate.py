"""CI gate: run the test cases on one network and fail if any limit is missed."""
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from nn_drone.charts.flight_test_charts import circle_chart, full_mission_chart
from nn_drone.flight_tests.circle import CirclePlan, run_circle_test
from nn_drone.flight_tests.full_mission import run_full_mission_test
from nn_drone.flight_tests.landing import MotorSequenceTiming
from nn_drone.flight_tests.long_route import RouteSpeeds
from nn_drone.flight_tests.waypoint_mission import fly_waypoint_mission

SQUARE_DRONES = 200
HEAVY_DRONES = 100
HEAVY_MASSES_KG = (10.0, 100.0)
HEAVY_DURATION_S = 90.0
THIN_AIR_ALTITUDE_M = 2000.0
MISSION_MASS_KG = 4.0
MISSION_SPEEDS = RouteSpeeds(horizontal_mps=3.5, climb_mps=3.0, descent_mps=1.8)
MISSION_RANDOMISED_DRONES = 10
SQUARE_SEED, HEAVY_SEED = 2, 3


@dataclass
class Check:
    test: str
    value: object
    limit: str
    passed: bool


class CiGate:
    def __init__(self, pilot, output_dir, log=print):
        self.pilot = pilot
        self.output_dir = Path(output_dir)
        self.log = log
        self.checks = []

    def check(self, name, value, passed, limit):
        self.checks.append(Check(name, value, limit, bool(passed)))
        self.log(f"{'PASS' if passed else 'FAIL'}  {name}: {value}  (limit {limit})")

    def run(self, weights_path):
        started = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._square_mission()
        self._thin_air()
        self._heavy_drones()
        self._circle()
        self._full_mission()
        passed = all(check.passed for check in self.checks)
        report = dict(weights=str(weights_path), frame=self.pilot.frame, checks=[asdict(check) for check in self.checks],
                      passed=passed, seconds=round(time.time() - started))
        (self.output_dir / f"ci_report_{self.pilot.frame}.json").write_text(json.dumps(report, indent=1))
        self.log(f"\n{'ALL TESTS PASSED' if passed else 'TESTS FAILED'} in {report['seconds']} s")
        return passed

    def _square_mission(self):
        flight = fly_waypoint_mission(self.pilot, SQUARE_DRONES, randomize=True, seed=SQUARE_SEED)
        completed, crashed = int(flight.completed.sum()), int(flight.crashed.sum())
        hover_error = float(np.nanmedian(flight.hold_error))
        self.check(f"square: completed of {SQUARE_DRONES}", completed, completed >= 198, ">= 198")
        self.check("square: crashed", crashed, crashed == 0, "== 0")
        self.check("square: hover error median (m)", round(hover_error, 3), hover_error < 0.10, "< 0.10")

    def _thin_air(self):
        flight = fly_waypoint_mission(self.pilot, SQUARE_DRONES, randomize=True, seed=SQUARE_SEED,
                                      start_altitude_m=THIN_AIR_ALTITUDE_M)
        completed, crashed = int(flight.completed.sum()), int(flight.crashed.sum())
        self.check(f"2000 m: completed of {SQUARE_DRONES}", completed, completed >= 180, ">= 180")
        self.check(f"2000 m: crashed of {SQUARE_DRONES}", crashed, crashed <= 10, "<= 10")

    def _heavy_drones(self):
        for mass in HEAVY_MASSES_KG:
            flight = fly_waypoint_mission(self.pilot, HEAVY_DRONES, randomize=True, seed=HEAVY_SEED,
                                          duration_s=HEAVY_DURATION_S, nominal_mass_kg=mass)
            completed, crashed = int(flight.completed.sum()), int(flight.crashed.sum())
            self.check(f"{mass:g} kg: completed of {HEAVY_DRONES}", completed, completed >= 85, ">= 85")
            self.check(f"{mass:g} kg: crashed", crashed, crashed == 0, "== 0")

    def _circle(self):
        plan = CirclePlan()
        nominal, summary = run_circle_test(self.pilot, plan)
        stem = self.output_dir / f"ci_circle_{self.pilot.frame}_results"
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=1))
        circle_chart(stem.with_suffix(".png"), plan, nominal)
        mean_error = summary["nominal_radial_error_mean_m"]
        self.check("circle: mean distance from line (m)", round(mean_error, 3), mean_error < 0.25, "< 0.25")
        self.check("circle: crashed", summary["robust_crashed"], summary["robust_crashed"] == 0, "== 0")

    def _full_mission(self):
        timing = MotorSequenceTiming()
        nominal, summary = run_full_mission_test(self.pilot, MISSION_MASS_KG, MISSION_SPEEDS,
                                                 robust_drones=MISSION_RANDOMISED_DRONES, timing=timing)
        stem = self.output_dir / f"ci_mission_{self.pilot.frame}_results"
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=1))
        full_mission_chart(stem.with_suffix(".png"), nominal, timing)
        standard, randomised = summary["standard"], summary["randomised_100"]
        disarmed_pwm = summary["pwm_sequence_standard_drone"]["13 disarmed"]
        self.check("mission: standard drone crashed", standard["crashed"], not standard["crashed"], "False")
        self.check("mission: touchdown speed (m/s)", round(standard["touchdown_speed_mps"], 2),
                   standard["touchdown_speed_mps"] < 1.0, "< 1.0")
        self.check("mission: touchdown offset (m)", round(standard["touchdown_offset_m"], 2),
                   standard["touchdown_offset_m"] < 0.5, "< 0.5")
        self.check("mission: ends disarmed at 1000 us", round(disarmed_pwm), disarmed_pwm == 1000.0, "== 1000")
        self.check(f"mission: randomised crashed of {MISSION_RANDOMISED_DRONES}", randomised["crashed"],
                   randomised["crashed"] == 0, "== 0")
        self.check(f"mission: randomised landed of {MISSION_RANDOMISED_DRONES}", randomised["landed"],
                   randomised["landed"] == MISSION_RANDOMISED_DRONES, f"== {MISSION_RANDOMISED_DRONES}")
