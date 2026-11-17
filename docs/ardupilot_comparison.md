# ArduPilot against the neural network

The same mission flown twice: once by ArduCopter 4.7.2 in ArduPilot's own simulator (SITL, standard PID
controllers, default tuning) and once by the neural networks in this repository's simulator. The ArduPilot
DataFlash log (`.BIN`) is read with pymavlink and both flights are measured with the same code.

**Mission:** take off to 20 m, fly 60 m north climbing to 30 m, fly 60 m east climbing to 40 m and hold 30 s,
one 15 m radius circle at 30 m, return to launch and land.
**Same on both sides:** 3 kg quad (ArduPilot's default SITL quad), 3 m/s horizontal and climb, 1.8 m/s descent,
0.5 m/s landing, 2 m/s wind from 45 degrees.

| | ArduCopter PID | Network, quad | Network, hexa |
|---|---|---|---|
| Mission time | 205 s | 184 s | 183 s |
| Distance from planned route, mean | 0.10 m | 0.37 m | 0.21 m |
| Distance from planned route, max | 0.70 m | 0.96 m | 0.63 m |
| Hover position error, mean | 0.043 m | 0.027 m | 0.014 m |
| Hover height wander (std) | 0.002 m | 0.012 m | 0.009 m |
| Overshoot above the 40 m target | none | 0.25 m | 0.09 m |
| Circle radius error, mean | 0.11 m | 0.31 m | 0.03 m |
| Circle height error, max | 0.04 m | 0.49 m | 0.51 m |
| Landing distance from take-off point | 0.03 m | 0.41 m | 0.11 m |
| Steepest tilt | 16 deg | 25 deg | 28 deg |
| PWM in hover, mean | 1581 us | 1618 us | 1585 us |
| PWM in hover, spread between motors | 1 us | 1 us | 477 us |
| PWM in hover, change per 0.1 s | 0.7 us | 8.1 us | 10.9 us |
| PWM range in flight | 1421-1700 us | 1151-1895 us | 1144-1906 us |

All values are in `results/ardupilot_comparison/comparison.json`; the chart is `results/ardupilot_comparison/comparison.png`; the data extracted from
the ArduPilot log (10 Hz) is `results/ardupilot_comparison/ardupilot_extracted.csv`.

## Stability (`sitl-stability`)

| | ArduCopter PID | Network, quad | Network, hexa |
|---|---|---|---|
| Hover roll / pitch wobble (std) | 0.12 / 0.13 deg | 0.19 / 0.26 deg | 0.38 / 0.34 deg |
| Heading swing during climb-out, peak to peak | 0.3 deg | 12.7 deg | 7.2 deg |
| Gust 0 to 8 m/s: pushed off position, peak | 0.50 m | 0.57 m | 0.17 m |
| Gust: time to get back within 0.3 m | 4.4 s | 3.7 s | never left 0.3 m |
| Gust: height change, peak | 0.02 m | 0.19 m | 0.06 m |
| Gust: tilt needed to hold against it | 10.9 deg | 7.9 deg | 7.9 deg |

All three stayed stable through the gust and none crashed. The gust pushed harder on ArduPilot's simulated
airframe (it needed 10.9 deg of tilt to hold, ours 7.9 deg), so its 0.50 m is against a stronger push.

Extra control delay the networks tolerate (network only, not measured for ArduPilot): both fly normally with
24 ms of added delay between network output and motors; the quad crashes at 36 ms and the hexa at 48 ms.
That is a thin margin for real hardware.

The hover heading figure in `results/ardupilot_comparison/stability.json` is not comparable: ArduPilot was still turning to its
commanded heading during the hold.

## Against a real flight log (`replay-log`)

A real ArduCopter flight, not a simulation: an Erle-Copter quad (X frame, ArduCopter 3.3-dev) flown on
5 December 2014, from the public repository https://github.com/erlerobot/logs (file
`pxf/copter/5-12-2014/218.BIN`; 335 s with LOITER, GUIDED, AUTO and LAND). The log is not copied here; clone
that repository to get it. 268 s of position-controlled flight were used.

**1. Shadow replay (open loop).** At every instant the quad network is given what the real autopilot knew
(gyro, accelerometer, attitude, position error to ArduPilot's own target, velocity) and its PWM is compared
with the PWM ArduPilot really sent. The network is not flying the drone.

| | Collective | Roll | Pitch | Yaw |
|---|---|---|---|---|
| Correlation, network PWM against ArduPilot PWM | 0.12 | 0.00 | 0.38 | 0.27 |
| Size of the moving part, ArduPilot / network (us RMS) | 20 / 103 | 9 / 23 | 16 / 29 | 18 / 60 |
| Correlation with ArduPilot's own attitude demand: ArduPilot / network | n/a | 0.42 / 0.40 | 0.54 / 0.66 | 0.33 / 0.68 |

Moment by moment the two PWM streams do not match, and the network's commands are two to five times larger.
But the network pushes toward the attitude ArduPilot was asking for about as consistently as ArduPilot's own
motor commands do. Average PWM was 1698 us for the real drone and 1598 us for the network, which is the two
airframes' different hover throttle, not a control difference.

**2. Same target track, closed loop in our simulator.** The network follows the exact target positions
ArduPilot was following.

| Stretch | Real flight, mean distance from target | Network in simulator |
|---|---|---|
| LOITER, 29 s | 0.30 m | 0.26 m |
| GUIDED, 82 s | 0.83 m | 0.36 m |
| LOITER, 117 s | 0.33 m | 0.11 m |
| AUTO, 40 s (target moving about 1.4 m/s) | 2.70 m | 2.17 m |

This second table flatters the network: the real drone had real wind, GPS error and vibration, and the
simulated one had no wind. It shows the network can follow a real mission's target track in simulation, not
that it would beat ArduPilot outdoors.

Assumptions: drone mass 1.5 kg (not in the log), battery voltage not logged so a neutral value is fed, the
network's integral is held at zero in the open-loop replay, and log data (10 Hz attitude and PWM, 50 Hz IMU)
is interpolated to the network's 93 Hz. The input builder is checked against our simulator first (it
reproduces the simulator's motor outputs exactly), and every sign convention was probed.

```
git clone --depth 1 https://github.com/erlerobot/logs
python -m nn_drone replay-log --log logs/pxf/copter/5-12-2014/218.BIN --output-dir out --mass 1.5
```

## Reading it fairly

- **Two different simulators.** ArduPilot flew in SITL's physics and sensor models; the networks flew in
  this repository's simulator. The airframes are both 3 kg quads but not the same motors or thrust, so this compares behaviour on
  the same mission, not the same aircraft. A like-for-like test needs the network running inside SITL.
- **Sensor noise differs.** With sensor noise switched off in our simulator, the quad network's hover PWM change
  falls from 8.1 to 0.2 us per 0.1 s, so most of its hover jitter is noise passing straight through the network
  (ArduPilot filters its sensors). In flight it still changes 6.1 us per 0.1 s against ArduPilot's 0.9.
- **Positions are true positions on both sides** (SITL's `SIM2` record for ArduPilot). ArduPilot's own EKF
  estimate differed from the truth by at most 0.12 m.
- **ArduPilot's log has PWM at 10 Hz**, so the networks' PWM is resampled to 10 Hz before comparing.
- One flight each, not a statistical sample.

## Reproduce

```
git clone --depth 1 --branch Copter-4.7.2 https://github.com/ArduPilot/ardupilot && cd ardupilot
git submodule update --init --recursive --depth 1 modules/waf modules/mavlink modules/DroneCAN/DSDL \
    modules/DroneCAN/pydronecan modules/DroneCAN/dronecan_dsdlc modules/DroneCAN/libcanard modules/lwip \
    modules/littlefs modules/Micro-CDR modules/Micro-XRCE-DDS-Client
pip install empy==3.3.4 pexpect future pymavlink
./waf configure --board sitl && ./waf copter

cd <this repo> && mkdir out
python -m nn_drone sitl-fly --ardupilot /path/to/ardupilot --output-dir out      # writes out/ardupilot_sitl.BIN
python -m nn_drone sitl-fly-network --output-dir out --frame quad_x
python -m nn_drone sitl-fly-network --output-dir out --frame hex_x
python -m nn_drone sitl-fly-network --output-dir out --frame quad_x --no-noise
python -m nn_drone sitl-compare --output-dir out
python -m nn_drone sitl-stability --ardupilot /path/to/ardupilot --output-dir out
```

ArduCopter 4.7 renamed the navigation parameters and moved them to metres (`WP_SPD`, `WP_SPD_UP`, `WP_SPD_DN`,
`WP_RADIUS_M`, `RTL_ALT_M`, `LAND_SPD_MS`); `nn_drone/ardupilot_comparison/mission_definition.py` holds both sets.
