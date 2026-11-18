"""Command line: python -m nn_drone <command> --help"""
import argparse
import json
import sys
from pathlib import Path

from nn_drone.paths import default_weights
from nn_drone.simulation.config import CURRENT_PROFILE


def _weights_argument(parser, required=False):
    parser.add_argument("--weights", type=Path, required=required,
                        help="network weights (.npz); defaults to the shipped network for --frame")
    parser.add_argument("--frame", default="quad_x", choices=["quad_x", "hex_x"])


def _load_pilot(args):
    from nn_drone.policy import NeuralPilot
    return NeuralPilot.from_checkpoint(args.weights or default_weights(args.frame))


def _write_json(path, data, echo=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    if echo:
        print(json.dumps(data, indent=1))


def train(args):
    from nn_drone.training.ppo import PPOSettings
    from nn_drone.training.trainer import Trainer, TrainingPlan
    ppo = PPOSettings(learning_rate=args.learning_rate, min_log_std=args.min_log_std,
                      temporal_smoothness=args.temporal_smoothness, spatial_smoothness=args.spatial_smoothness)
    plan = TrainingPlan(frame=args.frame, profile_name=args.profile, parallel_drones=args.drones,
                        steps_per_rollout=args.rollout_steps, time_budget_minutes=args.minutes, max_iterations=args.max_iterations,
                        output_stem=args.output, resume_from=args.resume or "", seed=args.seed,
                        critic_warmup_iterations=args.critic_warmup, config_overrides=json.loads(args.config), ppo=ppo,
                        device=args.device)
    Trainer(plan).run()


def square_mission(args):
    from nn_drone.flight_tests.waypoint_report import run_waypoint_test, summarize, write_outputs
    mission, nominal, robust = run_waypoint_test(_load_pilot(args), high=args.high, position_error_clip=args.error_clip)
    summary = summarize(nominal, robust)
    print(json.dumps(summary, indent=1))
    write_outputs(args.output_dir, mission, nominal, summary)


def plot_square_mission(args):
    import numpy as np
    from nn_drone.charts.flight_test_charts import waypoint_mission_chart
    flight_log = np.load(args.flight_log)
    recording = {name: flight_log[name] for name in flight_log.files if name != "mission"}
    waypoint_mission_chart(args.output, recording, flight_log["mission"], json.loads(Path(args.training_log).read_text()))


def circle(args):
    from nn_drone.charts.flight_test_charts import circle_chart
    from nn_drone.flight_tests.circle import CirclePlan, run_circle_test
    common = dict(position_error_clip=args.error_clip)
    if args.altitude is None:
        plan = CirclePlan(radius_m=args.radius, speed_mps=args.speed, mass_kg=args.mass, **common)
    else:
        plan = CirclePlan.at_altitude(args.radius, args.speed, args.mass, args.altitude, args.approach_time, **common)
    nominal, summary = run_circle_test(_load_pilot(args), plan)
    _write_json(f"{args.tag}_results.json", summary)
    circle_chart(f"{args.tag}_results.png", plan, nominal)


def envelope(args):
    from nn_drone.charts.flight_test_charts import envelope_chart
    from nn_drone.flight_tests.envelope import climb_test, mass_sweep
    pilot = _load_pilot(args)
    climb_nominal, climb_summary = climb_test(pilot, args.error_clip)
    print(json.dumps(climb_summary, indent=1), flush=True)
    rows = mass_sweep(pilot, args.error_clip, log=lambda row: print(row, flush=True))
    _write_json(f"{args.tag}_results.json", dict(climb_2000m=climb_summary, mass_sweep=rows), echo=False)
    envelope_chart(f"{args.tag}_results.png", climb_nominal.recording, rows)


def _route_speeds(args):
    from nn_drone.flight_tests.long_route import RouteSpeeds
    return RouteSpeeds(horizontal_mps=args.horizontal_speed, climb_mps=args.climb_speed, descent_mps=args.descent_speed)


def long_mission(args):
    from nn_drone.charts.flight_test_charts import long_mission_chart
    from nn_drone.flight_tests.long_mission import run_long_mission_test
    nominal, summary = run_long_mission_test(_load_pilot(args), args.mass, _route_speeds(args), args.randomised_drones)
    _write_json(f"{args.tag}_results.json", summary)
    long_mission_chart(f"{args.tag}_results.png", nominal)


def full_mission(args):
    from nn_drone.charts.flight_test_charts import full_mission_chart
    from nn_drone.flight_tests.full_mission import run_full_mission_test
    from nn_drone.flight_tests.landing import MotorSequenceTiming
    timing = MotorSequenceTiming()
    nominal, summary = run_full_mission_test(_load_pilot(args), args.mass, _route_speeds(args), args.randomised_drones, timing)
    _write_json(f"{args.tag}_results.json", summary)
    full_mission_chart(f"{args.tag}_results.png", nominal, timing)


def ci_gate(args):
    from nn_drone.flight_tests.ci_gate import CiGate
    weights = args.weights or default_weights(args.frame)
    passed = CiGate(_load_pilot(args), args.output_dir).run(weights)
    sys.exit(0 if passed else 1)


def export_model(args):
    from nn_drone.export.model_package import export_model as package
    card = package(args.weights, args.output_dir, args.test_report)
    print(json.dumps({key: card[key] for key in ("frame", "layers", "parameters", "size_bytes_float32")}))


def build_web(args):
    from nn_drone.export.web_page import build_web_page
    path, size_kb = build_web_page()
    print(path, size_kb, "kB")


def ground_link(args):
    from nn_drone.ground_link.runner import run_ground_link
    home = tuple(float(value) for value in args.home.split(","))
    run_ground_link(args.frame, args.weights or default_weights(args.frame), home, args.mass, args.gcs, args.speedup,
                    args.duration)


def ground_link_test(args):
    from nn_drone.ground_link.scripted_ground_station import ScriptedGroundStation
    sys.exit(0 if ScriptedGroundStation(args.frame, args.mass).run() else 1)


def _add_route_speed_arguments(parser, mass, horizontal, climb, descent):
    parser.add_argument("--mass", type=float, default=mass, help="kg")
    parser.add_argument("--horizontal-speed", type=float, default=horizontal, help="m/s")
    parser.add_argument("--climb-speed", type=float, default=climb, help="m/s")
    parser.add_argument("--descent-speed", type=float, default=descent, help="m/s")
    parser.add_argument("--randomised-drones", type=int, default=100)


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m nn_drone", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("train", help="train a network with PPO")
    command.add_argument("--frame", default="quad_x", choices=["quad_x", "hex_x"])
    command.add_argument("--profile", default=CURRENT_PROFILE.name, help="simulator generation (v1..v5)")
    command.add_argument("--drones", type=int, default=1024, help="drones simulated in parallel")
    command.add_argument("--rollout-steps", type=int, help="default: 128 reference steps (512 at 400 Hz)")
    command.add_argument("--minutes", type=float, default=8.0)
    command.add_argument("--max-iterations", type=int, default=100000)
    command.add_argument("--output", default="policy", help="output stem: <stem>.npz, <stem>_log.json, <stem>_snapNNN.npz")
    command.add_argument("--resume", help="continue from these weights (fewer inputs are zero-padded)")
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--critic-warmup", type=int, default=0, help="iterations with the policy frozen")
    command.add_argument("--config", default="{}", help='simulator overrides as JSON, e.g. {"moving_target_share": 0.0}')
    command.add_argument("--min-log-std", type=float, default=-3.0)
    command.add_argument("--temporal-smoothness", type=float, default=0.0)
    command.add_argument("--spatial-smoothness", type=float, default=0.0)
    command.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    command.set_defaults(handler=train)

    command = commands.add_parser("square-mission", help="square (or --high) waypoint mission, 1 + 500 drones")
    _weights_argument(command)
    command.add_argument("--high", action="store_true", help="20 m / 30 m legs instead of 3-5 m")
    command.add_argument("--error-clip", type=float, default=2.5)
    command.add_argument("--output-dir", type=Path, default=Path("."))
    command.set_defaults(handler=square_mission)

    command = commands.add_parser("plot-square-mission", help="chart of the last square mission and a training log")
    command.add_argument("--flight-log", type=Path, default=Path("flight_log.npz"))
    command.add_argument("--training-log", type=Path, default=Path("models/quad_log.json"))
    command.add_argument("--output", type=Path, default=Path("nn_drone_results.png"))
    command.set_defaults(handler=plot_square_mission)

    command = commands.add_parser("circle", help="circle mission")
    _weights_argument(command)
    command.add_argument("--error-clip", type=float, default=2.5)
    command.add_argument("--tag", default="circle")
    command.add_argument("--radius", type=float, default=5.0, help="m")
    command.add_argument("--speed", type=float, default=1.5, help="m/s")
    command.add_argument("--mass", type=float, default=1.0, help="kg")
    command.add_argument("--altitude", type=float, help="circle altitude above sea level, m")
    command.add_argument("--approach-time", type=float, default=6.0, help="s allowed to reach the circle")
    command.set_defaults(handler=circle)

    command = commands.add_parser("envelope", help="2000 m climb and 1-100 kg mass sweep")
    _weights_argument(command)
    command.add_argument("--error-clip", type=float, default=2.5)
    command.add_argument("--tag", default="envelope")
    command.set_defaults(handler=envelope)

    command = commands.add_parser("long-mission", help="six-waypoint route to 1500 m, in the air")
    _weights_argument(command)
    command.add_argument("--tag", default="mission95")
    _add_route_speed_arguments(command, 95.0, 6.0, 5.0, 3.0)
    command.set_defaults(handler=long_mission)

    command = commands.add_parser("full-mission", help="six-waypoint route from disarmed to disarmed on solid ground")
    _weights_argument(command)
    command.add_argument("--tag", default="mission_full")
    _add_route_speed_arguments(command, 95.0, 6.0, 5.0, 3.0)
    command.set_defaults(handler=full_mission)

    command = commands.add_parser("ci-gate", help="run the CI test cases; exit 1 if any limit is missed")
    _weights_argument(command)
    command.add_argument("--output-dir", type=Path, default=Path("."))
    command.set_defaults(handler=ci_gate)

    command = commands.add_parser("export-model", help="package the flight network: .npz, C header, model card")
    command.add_argument("--weights", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, default=Path("dist"))
    command.add_argument("--test-report", type=Path, help="CI report to include in the model card")
    command.set_defaults(handler=export_model)

    command = commands.add_parser("build-web", help="rebuild web/index.html from its sources and the shipped networks")
    command.set_defaults(handler=build_web)

    command = commands.add_parser("ground-link", help="real-time simulator with a MAVLink link for a ground station")
    _weights_argument(command)
    command.add_argument("--mass", type=float, default=1.0, help="kg")
    command.add_argument("--gcs", default="127.0.0.1:14550", help="ground station address ip:port")
    command.add_argument("--home", default="28.5355,77.3910,200", help="lat,lon,altitude above sea level (m)")
    command.add_argument("--speedup", type=float, default=1.0)
    command.add_argument("--duration", type=float, default=0.0, help="stop after this many real seconds (0 = until Ctrl-C)")
    command.set_defaults(handler=ground_link)

    command = commands.add_parser("ground-link-test", help="scripted ground station flies a mission over MAVLink")
    command.add_argument("--frame", default="quad_x", choices=["quad_x", "hex_x"])
    command.add_argument("--mass", type=float, default=4.0, help="kg")
    command.set_defaults(handler=ground_link_test)

    from nn_drone.ardupilot_comparison.commands import register as register_ardupilot_commands
    register_ardupilot_commands(commands)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)
