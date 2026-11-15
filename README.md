# nn-drone-controller

An experiment: a small neural network that flies a multirotor by driving the motor PWM directly
(1000–2000 µs), with no PID loops. Everything here is **simulation only** — a simplified rigid-body
simulator written for this experiment, not ArduPilot SITL, and nothing has flown on real hardware.

## The network

`61 → 64 → 64 → 4`, tanh activations. Output −1..1 per motor, mapped to `PWM = 1500 + 500 × output`.

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

## Results (v4 weights, simulation)

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

## Known problems

- **Yaw wobble on climb-out:** diagonal motor pairs swing about ±100 µs for several seconds after lift-off.
- **It trails moving targets** (it is never told the target's velocity).
- **Altitude overshoot** of about 0.9 m after fast climbs.
- **Training that did not work:** from-scratch runs with the 61-input network never learned to fly;
  several fine-tuning runs made flight worse and were discarded. v4 is the v3 weights plus the
  hand-designed changes above.
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
| `quad_x_policy_v4.npz` | Trained weights (`pi_W0..2`, `pi_b0..2` are the flight network) |
| `v1/`, `v2/`, `v3/` | Earlier versions |

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

`2.5` is the limit (in size-normalised metres) on the position error fed to the network; lower
means slower, gentler flight.
