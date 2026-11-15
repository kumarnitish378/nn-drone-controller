# nn-drone-controller

An experiment: a small neural network that flies a multirotor by driving the motor PWM directly
(1000–2000 µs), with no PID loops. Everything here is **simulation only** — a simplified rigid-body
simulator written for this experiment, not ArduPilot SITL, and nothing has flown on real hardware.

Two trained networks are included:

| Frame | Weights | Shape | How it was trained |
|---|---|---|---|
| Quad X | `quad_x_policy_v4.npz` | 61 → 64 → 64 → 4 | PPO from scratch (21 inputs), then extended and fine-tuned in stages |
| Hexa X | `hex_x_policy_v1.npz` | 67 → 64 → 64 → 6 | PPO from scratch in one go, 98 million samples |

## The network

Quad: `61 → 64 → 64 → 4`, hex: `67 → 64 → 64 → 6`, tanh activations. Output −1..1 per motor, mapped to
`PWM = 1500 + 500 × output`. Motor order follows ArduPilot's QUAD X / HEXA X numbering.

Inputs (quad): gyro (3), accelerometer (3), roll and pitch (2), heading error as sin/cos (2),
position error to the target in the body frame (3), velocity (3), battery voltage (1),
previous motor outputs (4), integrated position error (3), the two previous frames of the
sensor and output values (36), and a mass-class input (1, currently unused: zero weights).

Trained with PPO (written in plain NumPy, `ppo.py`) on randomised drones: mass, inertia, motor
strength and lag, battery voltage, wind and sensor noise.

Two things around the network are hand-designed, not learned:

- **Size normalisation** – inputs and control period are scaled by `L = (mass / 1 kg)^(1/3)`, so a
  larger drone looks like the 1 kg one to the network (100 Hz loop at 1 kg, about 46 Hz at 100 kg).
- **Integral input weights** – initially set by hand to 0.15 × the position-error weights
  (Ki = 0.5 /s), then fine-tuned by PPO.

## Results: quad (v4 weights, simulation)

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

Charts and raw numbers are in `results/`.

## Results: hexacopter (`hex_x_policy_v1.npz`, simulation)

Same test cases, same randomisation. Select the frame with `NN_FRAME=hex_x`.

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

## Known problems

- **Yaw wobble on climb-out:** diagonal motor pairs swing about ±100 µs for several seconds after lift-off.
- **It trails moving targets** (it is never told the target's velocity).
- **Altitude overshoot** of about 0.9 m after fast climbs.
- **Training that did not work:** from-scratch runs with the 61-input network never learned to fly;
  several fine-tuning runs made flight worse and were discarded. v4 is the v3 weights plus the
  hand-designed changes above.
- **Landing detector:** `mission_full.py` now also requires low throttle before declaring a landing;
  the first version could trigger on a slow drone still hovering, which then fell from 10 m.
- **Simulator limits:** simple ground (no bounce, tipping or ground effect), no battery drain,
  no stronger wind at altitude, and the 100 kg results assume a scaled-up airframe with the same
  thrust-to-weight ratio.
- The `evaluate.py` in `v1/`–`v3/` has a timing bug (it stopped the clock one waypoint early);
  use the top-level `evaluate.py` to re-measure old weights.

## Files

| File | Purpose |
|---|---|
| `sim.py` | Simulator: quad X / hex X frames, motors, battery, air density, size scaling, ground, observations, reward |
| `ppo.py` | Network and PPO in NumPy, with optional action-smoothness regularisers |
| `train.py` | Training script |
| `evaluate.py` | Square mission and 20/30 m mission |
| `circle.py` | Circle mission |
| `envelope.py` | 2000 m climb and 1–100 kg sweep |
| `mission.py` | Six-waypoint mission (no ground) |
| `mission_full.py` | Same mission from disarmed to disarmed on solid ground |
| `plot.py` | Chart for the square missions |
| `quad_x_policy_v4.npz`, `hex_x_policy_v1.npz` | Trained weights (`pi_W0..2`, `pi_b0..2` are the flight network) |
| `quad_log.json`, `hex_log.json` | Training curves |
| `v1/`, `v2/`, `v3/` | Earlier versions |

## CI: tested model as the output

`.github/workflows/ci.yml` runs on every push and pull request, once per frame:

1. `ci_test.py` runs the test cases (square mission, the same at 2000 m, 10 kg and 100 kg drones,
   the circle, and the six-waypoint mission from disarmed to disarmed) and fails if any limit is missed.
2. Only if all of them pass, `export_model.py` packages the flight network and the workflow uploads it
   as the artifact `model-quad_x` / `model-hex_x`: weights (`.npz`), a C header with the weights as
   `float` arrays, and a model card with the test report.

Model size (flight network only): quad 8,388 parameters (33.6 kB as float32), hex 8,902 parameters (35.6 kB).

## Running

Needs Python 3 with `numpy` and `matplotlib`.

```
python3 evaluate.py quad_x_policy_v4.npz quad_x 2.5            # square mission
python3 evaluate.py quad_x_policy_v4.npz quad_x 2.5 high       # 20 m / 30 m mission
python3 circle.py quad_x_policy_v4.npz 2.5 circle_v4           # 5 m circle, 1 kg
python3 circle.py quad_x_policy_v4.npz 2.5 big 50 4 95 1200 30 # radius, speed, mass, altitude, approach time
python3 envelope.py quad_x_policy_v4.npz 2.5 envelope_v4       # 2000 m climb + mass sweep
python3 mission_full.py quad_x_policy_v4.npz m4 4 3.5 3 1.8    # mass, horizontal / climb / descent speed
```

Hexacopter: put `NN_FRAME=hex_x` in front and use the hex weights, for example

```
NN_FRAME=hex_x python3 evaluate.py hex_x_policy_v1.npz hex_x 2.5
NN_FRAME=hex_x python3 mission_full.py hex_x_policy_v1.npz m4 4 3.5 3 1.8
python3 train.py --frame hex_x --minutes 9 --out hex --cfg '{"w_da2":0.3,"w_da1":0.0,"w_tilt":0.0,"w_spd":0.03,"spd_free":4.0,"err_clip":5.0,"p_moving":0.0}'
```

The last line is the configuration that trained the hex network (repeat with `--resume hex.npz`
until about 100 million samples, on a 1 kg drone at sea level).

`2.5` is the limit (in size-normalised metres) on the position error fed to the network; lower
means slower, gentler flight.
