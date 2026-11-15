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

All values are in `results/comparison.json`; the chart is `results/comparison.png`; the data extracted from
the ArduPilot log (10 Hz) is `results/ardupilot_extracted.csv`.

## Reading it fairly

- **Two different simulators.** ArduPilot flew in SITL's physics and sensor models; the networks flew in
  `sim.py`. The airframes are both 3 kg quads but not the same motors or thrust, so this compares behaviour on
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

cd <this repo>/ardupilot_compare && mkdir out
python3 fly_ardupilot.py /path/to/ardupilot out      # writes out/ardupilot_sitl.BIN
python3 fly_network.py out quad_x && python3 fly_network.py out hex_x && python3 fly_network.py out quad_x nonoise
python3 compare.py out
```

ArduCopter 4.7 renamed the navigation parameters and moved them to metres (`WP_SPD`, `WP_SPD_UP`, `WP_SPD_DN`,
`WP_RADIUS_M`, `RTL_ALT_M`, `LAND_SPD_MS`); `mission_def.py` holds both sets.
