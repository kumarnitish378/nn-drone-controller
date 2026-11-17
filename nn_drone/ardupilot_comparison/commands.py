"""Command-line entries for the ArduPilot comparison."""
from pathlib import Path


def _fly_sitl(args):
    from nn_drone.ardupilot_comparison.flights import fly_ardupilot
    fly_ardupilot(args.ardupilot, args.output_dir, args.speedup)


def _fly_network(args):
    from nn_drone.ardupilot_comparison.flights import fly_network
    fly_network(args.output_dir, args.frame, sensor_noise=not args.no_noise)


def _compare(args):
    from nn_drone.ardupilot_comparison.comparison import compare_flights
    compare_flights(args.output_dir)


def _stability(args):
    from nn_drone.ardupilot_comparison.stability import run_stability
    run_stability(args.ardupilot, args.output_dir)


def _replay(args):
    from nn_drone.ardupilot_comparison.real_log_replay import DEFAULT_LABEL, replay_log, shadow_self_test
    if args.self_test:
        shadow_self_test()
        return
    replay_log(args.log, args.output_dir, args.mass, args.tag, args.label or DEFAULT_LABEL)


def register(commands):
    command = commands.add_parser("sitl-fly", help="fly the comparison mission in ArduCopter SITL and keep its log")
    command.add_argument("--ardupilot", type=Path, required=True, help="ArduPilot checkout with a built SITL binary")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--speedup", type=int, default=10)
    command.set_defaults(handler=_fly_sitl)

    command = commands.add_parser("sitl-fly-network", help="fly the comparison mission with a network, save its log")
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--frame", default="quad_x", choices=["quad_x", "hex_x"])
    command.add_argument("--no-noise", action="store_true", help="sensor noise off")
    command.set_defaults(handler=_fly_network)

    command = commands.add_parser("sitl-compare", help="measure both sides' logs: comparison.json, .png and CSV")
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(handler=_compare)

    command = commands.add_parser("sitl-stability", help="attitude steadiness, gust test and delay tolerance")
    command.add_argument("--ardupilot", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(handler=_stability)

    command = commands.add_parser("replay-log", help="replay a real ArduCopter .BIN log against the quad network")
    command.add_argument("--log", type=Path)
    command.add_argument("--output-dir", type=Path, default=Path("."))
    command.add_argument("--mass", type=float, default=1.5, help="kg (not in the log)")
    command.add_argument("--tag", default="real_log_replay")
    command.add_argument("--label")
    command.add_argument("--self-test", action="store_true", help="only check the input builder against the simulator")
    command.set_defaults(handler=_replay)
