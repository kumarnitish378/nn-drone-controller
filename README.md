# nn-drone-controller

An experiment: a small neural network that flies a multirotor by driving the motor PWM directly
(1000–2000 µs), with no PID loops. Everything here is **simulation only** — a simplified rigid-body
simulator written for this experiment, not ArduPilot SITL, and nothing has flown on real hardware.

Two trained networks are included. By default they run at 400 Hz, the rate of ArduCopter's main loop
(simulator profile v5, see [Running at 400 Hz](#running-at-400-hz)):

| Frame | Weights (400 Hz) | Shape | How it was trained |
|---|---|---|---|
| Quad X | `models/quad_x_policy_v4_400hz.npz` | 61 → 64 → 64 → 4 | PPO from scratch (21 inputs), then extended and fine-tuned in stages, at 100 Hz |
| Hexa X | `models/hex_x_policy_v1_400hz.npz` | 67 → 64 → 64 → 6 | PPO from scratch in one go, 98 million samples, at 100 Hz |

The 400 Hz files hold the same weights as `models/quad_x_policy_v4.npz` and `models/hex_x_policy_v1.npz`,
which fly in the 100 Hz simulator they were trained in.

## The network

Quad: `61 → 64 → 64 → 4`, hex: `67 → 64 → 64 → 6`, tanh activations. Output −1..1 per motor, mapped to
`PWM = 1500 + 500 × output`. Motor order follows ArduPilot's QUAD X / HEXA X numbering.

Inputs (quad): gyro (3), accelerometer (3), roll and pitch (2), heading error as sin/cos (2),
position error to the target in the body frame (3), velocity (3), battery voltage (1),
previous motor outputs (4), integrated position error (3), the two previous frames of the
sensor and output values (36), and a mass-class input (1, currently unused: zero weights).

Trained with PPO (PyTorch, `nn_drone/training/`; it trains on an NVIDIA GPU when one is available) on randomised drones: mass, inertia, motor
strength and lag, battery voltage, wind and sensor noise.

Two things around the network are hand-designed, not learned:

- **Size normalisation** – inputs are scaled by `L = (mass / 1 kg)^(1/3)`, so a larger drone looks like
  the 1 kg one to the network. In profiles v1–v4 the control period is scaled too (100 Hz at 1 kg, about
  46 Hz at 100 kg); in v5 every size runs at 400 Hz and the earlier frames the network sees are one
  10 ms × √L step apart (4 loop iterations at 1 kg, 9 at 95 kg).
- **Integral input weights** – initially set by hand to 0.15 × the position-error weights
  (Ki = 0.5 /s), then fine-tuned by PPO.

## Running at 400 Hz

Profile v5 steps the network and the physics at 400 Hz for every drone size. Everything time-based is
counted in size-normalised time (reference steps of 10 ms × √L), so it means the same as at 100 Hz:
the per-step rewards and the motor-change penalty, episode length, the PPO discount and GAE factors, the
position-error integrator, how often training waypoints jump, and the spacing of the history frames.

Both networks, unchanged, pass every CI gate check at 400 Hz. On training-style episodes (512 randomised
1 kg quads, 6 s) the quad scores the same at both rates: 0.78 reward per 10 ms, 0.52 crashes per 1000
reference steps at 100 Hz and 0.31 at 400 Hz, mean distance 1.56 m and 1.51 m.

**Training at 400 Hz has not improved them yet.** Three PPO fine-tuning runs from these weights
(20 minutes each, about 170 million samples) all made the deterministic network worse on the same
episodes; for the quad the crash rate rose from 0.27 to 0.6–0.9 per 1000 reference steps, for the hex the
reward fell from 1.54 to 0.8–1.2. The exploration noise is the likely reason: drawn fresh every 2.5 ms it
costs far more under the motor-change penalty than at 100 Hz (reward per 10 ms with noise 0.05: 0.34
at 400 Hz against 0.71 at 100 Hz), which buries the learning signal. Holding the noise for 10 ms made
training diverge; a critic warm-up and a lower learning rate only slowed the decline.

| Run | Change | Quad crash/1k (start → end) | Hex reward (start → end) |
|---|---|---|---|
| 1 | learning rate 1e-4 | 0.27 → 0.86 | 1.54 → 0.83 |
| 2 | exploration noise held for 10 ms | 0.27 → 1.58 (diverged) | 1.54 → 1.23, crash/1k 0.08 → 1.04 |
| 3 | 30 critic-only iterations, learning rate 3e-5 | 0.25 → 0.62 | 1.54 → 1.17 |

## Results: quad (v4 weights, simulation at 100 Hz)

| Test | Result |
|---|---|
| Square waypoint mission, 500 randomised 1 kg drones | 500 completed, 0 crashed |
| 20 m / 30 m altitude mission, 500 drones | 500 completed, 0 crashed |
| Climb to 2000 m (thrust falls about 18%), 200 drones | 0 crashed, 189 finished within 10 min |
| Mass sweep 1, 2, 5, 10, 25, 50, 100 kg (200 drones each) | all completed, 0 crashed |
| Hover error 6–8 s after arrival | 0.05 m at 1 kg, 0.19 m at 10 kg, 0.70 m at 100 kg |
| 5 m radius circle, 1 kg | 0.14 m from the line on average |
| 50 m radius circle at 1200 m, 95 kg | 0.72 m from the line on average |
| Six-waypoint mission to 1500 m with circle, rectangle and landing | completed at 95 kg and 4 kg |

Charts and raw numbers are in `results/quad_x/`.

## Results: hexacopter (`hex_x_policy_v1.npz`, simulation)

Same test cases, same randomisation. Select the frame with `--frame hex_x`.

| Test | Result |
|---|---|
| Square waypoint mission, 500 randomised 1 kg drones | 500 completed, 0 crashed |
| 20 m / 30 m altitude mission, 500 drones | 500 completed, 0 crashed |
| Climb to 2000 m, 200 drones | 0 crashed, 187 finished within 10 min |
| Mass sweep 1–100 kg (200 drones each) | 0 crashed; completed 200, 200, 199, 199, 197, 194, 189 |
| Hover error 6–8 s after arrival | 0.04 m at 1 kg, 0.10 m at 10 kg, 0.25 m at 100 kg |
| 5 m radius circle, 1 kg | 0.06 m from the line on average |
| 50 m radius circle at 1200 m, 95 kg | 0.47 m from the line on average |
| Six-waypoint mission, 95 kg and 4 kg (100 drones each) | all landed, 0 crashed |
| Same mission from disarmed to disarmed, 4 kg | all took off, landed and disarmed, 0 crashed |
| Same mission from disarmed to disarmed, 95 kg | all took off and landed, 0 crashed; 3 of 100 never disarmed |

Charts and raw numbers are in `results/hex_x/`.

Hex-specific problems:

- **Uneven hover throttle:** in hover the six motors sit at three different levels (about 1800, 1670
  and 1310 µs in pairs) instead of all near the same value. It flies, but wastes power and leaves
  less headroom, which is why heavy or weak drones climb slowly and some do not finish in time.
- **Aggressive:** up to 47° tilt on the square mission (55° for the 95th percentile drone).
- The mass-class input was constant during training, so its weight is folded into the bias (unused).
- No yaw wobble on climb-out, unlike the quad.
- **Landing detection can miss:** in the 95 kg full mission, 3 of 100 randomised drones landed but their
  average throttle never fell below the detector's 1450 µs threshold, so they were not disarmed.

## Against ArduPilot

`ardupilot_compare/` flies one mission with ArduCopter 4.7.2 in ArduPilot's SITL and with both networks, then
compares the logs. In short: ArduPilot's PID controller follows the route more closely (0.10 m average error
against 0.37 m for the quad network and 0.21 m for the hexa), lands closer to the take-off point, tilts less
and moves its motors far more gently. The networks held the hover position slightly tighter in this one flight
and finished the mission sooner. The two sides ran in different simulators; see that folder's README.

## Known problems

- **Yaw wobble on climb-out:** diagonal motor pairs swing about ±100 µs for several seconds after lift-off.
- **It trails moving targets** (it is never told the target's velocity).
- **Altitude overshoot** of about 0.9 m after fast climbs.
- **Training that did not work:** from-scratch runs with the 61-input network never learned to fly;
  several fine-tuning runs made flight worse and were discarded. v4 is the v3 weights plus the
  hand-designed changes above.
- **Landing detector:** the full mission (`full-mission`) now also requires low throttle before declaring a landing;
  the first version could trigger on a slow drone still hovering, which then fell from 10 m.
- **Simulator limits:** simple ground (no bounce, tipping or ground effect), no battery drain,
  no stronger wind at altitude, and the 100 kg results assume a scaled-up airframe with the same
  thrust-to-weight ratio.
- The evaluation scripts of the first three versions had a timing bug (they stopped the clock one waypoint
  early). Those versions now live on as simulator profiles v1–v3 (see below) and are measured with the current
  flight tests.

## Code layout

Everything is one Python package, `nn_drone`, driven from one command line: `python -m nn_drone <command> --help`.

| Package | Purpose |
|---|---|
| `nn_drone/simulation/` | Batched simulator: motor layouts, airframe randomisation and size scaling, rigid-body physics, network inputs, position-error integrator, reward, training targets |
| `nn_drone/policy/` | PyTorch networks, weights files (`.npz`), and `NeuralPilot`, which flies a simulated drone with a trained network |
| `nn_drone/training/` | PPO with optional action-smoothness regularisers, rollout collection, the training loop |
| `nn_drone/flight_tests/` | Square / high mission, circle, envelope, the six-waypoint mission in the air and from disarmed to disarmed, landing detection, the CI gate |
| `nn_drone/ground_link/` | MAVLink link for QGroundControl / Mission Planner: parameters, navigation, flight modes and missions, the server, a scripted ground station |
| `nn_drone/ardupilot_comparison/` | Same mission in ArduCopter SITL and with the networks; log reading, metrics, stability checks, real-log replay ([docs](docs/ardupilot_comparison.md)) |
| `nn_drone/export/` | Model package (`.npz`, C header, model card) and the web page build |
| `nn_drone/charts/` | Charts for all of the above |
| `models/` | Trained weights and training curves; `models/legacy/` holds the v1–v3 quad networks |
| `results/` | Charts and raw numbers: `quad_x/`, `hex_x/`, `ardupilot_comparison/` |
| `web/` | 3D web tool: `index.html` (built), `core.js`, `template.html` |
| `tests/` | The simulator, networks and PPO checked step for step against the original implementation, and the web core against Python |

### Simulator profiles

The experiment went through four simulator generations at 100 Hz, and v5 moves the network to 400 Hz. Each
weights file records the profile it flies in (`simulator_profile`), and the tools fly it there, so old networks
behave exactly as they did:

| Profile | Adds | Network |
|---|---|---|
| v1 | sensors, battery, previous motor outputs (21 inputs for a quad) | `models/legacy/quad_x_v1.npz` |
| v2 | integrated position error (24 inputs) | `models/legacy/quad_x_v2.npz` |
| v3 | two previous frames, moving targets, smoothness rewards (60 inputs) | `models/legacy/quad_x_v3.npz` |
| v4 | size scaling to 100 kg, thinner air with altitude, solid ground, mass-class input (61 inputs) | `models/quad_x_policy_v4.npz`, `models/hex_x_policy_v1.npz` |
| v5 | the network in a fixed 400 Hz loop at every size, physics at 400 Hz (same inputs as v4) | `models/quad_x_policy_v4_400hz.npz`, `models/hex_x_policy_v1_400hz.npz` |

Weights files are shared with the C header export and the web tool: `pi_W0..2`, `pi_b0..2` are the flight
network, stored [inputs][outputs].

## CI: tested model as the output

`.github/workflows/ci.yml` runs on every push and pull request. A third job checks the ground station link
and the web tool. The model jobs run once per frame:

1. `python -m nn_drone ci-gate` runs the test cases (square mission, the same at 2000 m, 10 kg and 100 kg drones,
   the circle, and the six-waypoint mission from disarmed to disarmed) and fails if any limit is missed.
2. Only if all of them pass, `python -m nn_drone export-model` packages the flight network and the workflow uploads it
   as the artifact `model-quad_x` / `model-hex_x`: weights (`.npz`), a C header with the weights as
   `float` arrays, and a model card with the test report.

Model size (flight network only): quad 8,388 parameters (33.6 kB as float32), hex 8,902 parameters (35.6 kB).
## Ground station link (QGroundControl / Mission Planner)

`python -m nn_drone ground-link` runs the simulator in real time and speaks MAVLink, so a ground station can change
parameters, upload missions and fly them. It announces itself as an ArduCopter-type vehicle so the
usual flight modes appear, but it is not ArduPilot: the network in this repository flies it.

```
pip install -r requirements.txt
python -m nn_drone ground-link                           # 1 kg quad, sends to UDP 14550 on this PC
python -m nn_drone ground-link --frame hex_x --mass 95   # 95 kg hexacopter
python -m nn_drone ground-link --gcs 192.168.1.20:14550 --home 28.5355,77.3910,200 --speedup 5
```

Start QGroundControl or Mission Planner: both listen on UDP 14550 and connect on their own when the first
heartbeat arrives (in Mission Planner do not press Connect for UDP 14550; it is already listening).

- **Modes:** GUIDED, AUTO, LOITER, RTL, LAND.
- **Mission commands:** TAKEOFF, WAYPOINT (with hold time), LOITER_TURNS (circle with radius), LOITER_TIME,
  LOITER_UNLIM, RETURN_TO_LAUNCH, LAND, DO_CHANGE_SPEED, DO_JUMP. Anything else is skipped with a message.
- **Drone parameters** (change while disarmed): `SIM_MASS` (kg), `SIM_TW_RATIO`, `SIM_BATT_VOLT`,
  `SIM_WIND_SPD`, `SIM_WIND_DIR`, `SIM_NOISE`, `SIM_SPEEDUP`, `NN_ERR_CLIP`. The frame (quad or hex) is
  chosen with `--frame` at start.
- **Navigation parameters** (ArduCopter names and units): `WPNAV_SPEED`, `WPNAV_SPEED_UP`, `WPNAV_SPEED_DN`,
  `WPNAV_RADIUS`, `LAND_SPEED`, `LAND_ALT_LOW`, `RTL_ALT`, `CIRCLE_RADIUS`, `WP_YAW_BEHAVIOR`, `MOT_SPIN_ARM`,
  `DISARM_DELAY`. Speed defaults scale with mass; if you set them much higher the drone falls behind the path.
- **Telemetry:** position, attitude, GPS, battery voltage, mission progress and the actual motor PWM
  (`SERVO_OUTPUT_RAW`, outputs 1-4 or 1-6).

`python -m nn_drone ground-link-test --frame quad_x --mass 4` is a scripted ground station that downloads and sets parameters,
uploads and reads back a mission, flies it in AUTO, then does a GUIDED take-off, go-to and land. It passes
for a 1 kg quad, a 4 kg quad and a 95 kg hexacopter. **It has not been tried with the real QGroundControl
or Mission Planner yet**; they send more requests than the script does, so expect some rough edges.
No RC or joystick input, no fence or rally points, and battery percentage is not estimated.

## 3D web tool

`web/index.html` is a single page with a 3D view, a flight-plan editor and a drone configurator. The
simulator and both networks run inside the browser, so it needs no Python: open the file in a browser
(it loads three.js from a CDN, so it needs an internet connection the first time).

- Plan: take-off, waypoints, circles, return to launch and land, with a few ready-made plans.
- Drone: quad or hexa, mass 0.5-100 kg, thrust-to-weight, battery, wind, field elevation, sensor noise, speeds.
- Watch: the planned route and flown track in 3D, per-motor PWM bars and chart, and a flight record.
- See the network fly: every neuron's activation and the strongest input × weight paths, live, and each layer's
  activity over the last minute of flight. Pause and hover (or tap) a neuron to inspect one moment.

`web/core.js` is a port of the simulator and of the ground-link flight logic; `tests/test_web_core.py` flies the
same mission in both and checks the tracks agree (they do, to within 5 cm for a 4 kg quad and 12 cm for a
95 kg hexacopter, with identical step counts). Rebuild the page with `python -m nn_drone build-web` after changing
`web/core.js`, `web/template.html` or the weights.

`web/network.html` is the network lab: set every input by hand (attitude, rates, target, velocity, integral,
previous outputs, earlier frames) and see each neuron's activation, the strongest input × weight paths, the motor
commands and the thrust and torques they make, a sweep of any input, and which inputs move an output most.
Its input builder (`web/network_core.js`) is checked against the simulator in `tests/test_network_lab.py`.
To open it on another device: `python -m http.server 8765 --bind 0.0.0.0` in `web/`, then
`http://<this computer's address>:8765/network.html`.

## Running

Needs Python 3.10+ with the packages in `requirements.txt`. For GPU training install the CUDA build of PyTorch
(for example `pip install torch --index-url https://download.pytorch.org/whl/cu128`).

```
python -m nn_drone square-mission                                  # square mission, quad
python -m nn_drone square-mission --high                           # 20 m / 30 m mission
python -m nn_drone circle --tag circle_v4                          # 5 m circle, 1 kg
python -m nn_drone circle --tag big --radius 50 --speed 4 --mass 95 --altitude 1200 --approach-time 30
python -m nn_drone envelope --tag envelope_v4                      # 2000 m climb + mass sweep
python -m nn_drone full-mission --tag m4 --mass 4 --horizontal-speed 3.5 --climb-speed 3 --descent-speed 1.8
python -m nn_drone ci-gate --frame hex_x                           # every CI test case
python -m pytest                                                   # equivalence and unit tests
```

Hexacopter: add `--frame hex_x`; another network: `--weights path.npz`. Training:

```
python -m nn_drone train --frame hex_x --minutes 9 --output hex --config '{"position_error_clip_m": 5.0, "moving_target_share": 0.0, "reward": {"action_change_squared": 0.3, "action_change_absolute": 0.0, "excess_tilt": 0.0, "excess_speed": 0.03, "free_speed": 4.0}}'
```

That is the configuration that trained the hex network (at 100 Hz: add `--profile v4`; repeat with `--resume hex.npz` until about 100 million
samples, on a 1 kg drone at sea level). `--device` picks `cuda` or `cpu` (default: the GPU when PyTorch sees
one); on an RTX 4050 laptop GPU 60 iterations of 1024 drones x 128 steps took about 50 s, against about 250 s
for the original NumPy trainer on the CPU, with the same learning curve. Training defaults to profile v5 (400 Hz, 512 steps per
rollout, about 4x the time per simulated second); `--profile v1`...`v4` trains in an older simulator generation.

`--error-clip 2.5` is the limit (in size-normalised metres) on the position error fed to the network; lower
means slower, gentler flight.
