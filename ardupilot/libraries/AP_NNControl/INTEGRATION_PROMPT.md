# Task: integrate AP_NNControl into this ArduPilot tree (ArduCopter)

You are a coding agent working inside an ArduPilot checkout. The folder `libraries/AP_NNControl/` (the folder
this file is in) holds a neural-network flight controller. Wire it into **ArduCopter**: add a new flight mode,
**NN**, in which the network flies the vehicle by setting each motor's PWM. Make it build, and check it in SITL.
Report what you did.

Read this whole file before changing anything.

## What is in this folder

| File | Role | Rule |
|---|---|---|
| `NN_Config.h`, `NN_Model.h`, `NN_Network.*`, `NN_Pilot.*`, `NN_Stm32Cordic.*` | Portable core: builds the network inputs, history and integral, and runs the network. It has been verified number-for-number against the training simulator. | **Do not edit.** If it fails to compile, report the error instead of changing the maths. |
| `models/NN_Model_quad_x.h`, `models/NN_Model_hex_x.h` | Generated weights. Each defines arrays, so include each one from **one** `.cpp` only (already done in `AP_NNControl.cpp`). | **Do not edit.** |
| `AP_NNControl.h/.cpp` | ArduPilot front end: `NN_*` parameters, sensor reading (AHRS, INS, battery), `NNC` log message. | Adapt to this tree's APIs if needed. |
| `AP_MotorsMatrix_NN.h/.cpp` | Subclass of `AP_MotorsMatrix`. It overrides `output_to_motors()` to send the network's PWM while the spool state is `THROTTLE_UNLIMITED`. | Adapt to this tree's APIs if needed. |
| `copter_mode/mode_nn.h.in`, `copter_mode/mode_nn.cpp.in` | The Copter flight mode. They are templates: copy them to `ArduCopter/mode_nn.h` and `ArduCopter/mode_nn.cpp`. | Adapt to this tree's APIs if needed. |
| `INTEGRATION_PROMPT.md` | This file. | |

The ArduPilot-facing files were written against the ArduPilot 4.5/4.6 APIs but have **never been compiled inside
an ArduPilot tree**. Expect to fix some names: for example, how motors are constructed, inertial-nav getters,
`ahrs` helpers, `get_pilot_desired_climb_rate`, logging calls, or the `NEW_NOTHROW` macro. Before you change a
call, look it up in this tree (grep the headers) and use what this tree really has.

## The behaviour that must be kept

These come from how the network was trained; don't "simplify" them away.

1. **Loop and call order.**
   - `AP_NNControl::update()` and then `commit()` run **once per main loop at 400 Hz**, from arming until
     disarming. That includes while the motors spool up, when ArduPilot's PWM goes to the motors rather than
     the network's.
   - `commit()` gets the PWM the motors actually receive that loop:
     - the network's PWM when the spool state is `THROTTLE_UNLIMITED`;
     - otherwise, what ArduPilot sent, read back from the motor outputs.
2. **Reset on arming.** At arming, `reset(1000)` (motors stopped). On entering the mode while already flying,
   `reset(current motor PWM)`.
3. **Conventions.** NED world frame and FRD body frame:
   - position and velocity in metres from the EKF origin;
   - target yaw in radians, clockwise from north.

   The core converts to its internal frames itself. Don't add sign flips.
4. **Motor numbering.** The order is ArduPilot's own QUAD X / HEXA X motor numbering (motor 1 = index 0).
   `AP_MotorsMatrix_NN` writes with `rc_write(motor_index, pwm)`, the same path as the base class.
5. **Who controls the motors.**
   - ArduPilot keeps arming, safety, motor test and spool up/down.
   - The network's PWM replaces the mixer only when the motors are armed, the spool state is
     `THROTTLE_UNLIMITED`, and the network sent a PWM within the last 20 ms.
   - Otherwise the base `AP_MotorsMatrix::output_to_motors()` runs unchanged.
6. **Mode behaviour.**
   - The sticks move a target: pitch/roll at `WPNAV_SPEED`, throttle stick about centre for climb/descent,
     yaw at up to 90°/s.
   - The target is limited to `NN_ERR_CLIP × ∛NN_MASS` metres horizontally from the vehicle.
   - Each loop the mode resets the attitude controller's target and I-terms and calls
     `set_throttle_out(mean thrust)`, so leaving the mode is smooth and the land detector sees real thrust.
   - To land, the pilot switches to LAND.
7. **Refusals.**
   - The mode refuses to start without position (`position_ok()`), when `NN_ENABLE` is 0, when `SCHED_LOOP_RATE`
     is not 400, when the frame has no network (only quad X and hexa X are supported), or when
     `MOT_PWM_MIN`/`MOT_PWM_MAX` are not 1000/2000.
   - Pre-arm fails if `NN_ENABLE` is 1 but the network can't run.

## Steps

1. **Wire the library.** Add `'AP_NNControl',` to `ap_libraries` in `ArduCopter/wscript`. The `copter_mode/*.in`
   files must not be compiled from the library folder, which is why they end in `.in`.
2. **Add the mode.**
   - Copy `copter_mode/mode_nn.h.in` to `ArduCopter/mode_nn.h` and `copter_mode/mode_nn.cpp.in` to
     `ArduCopter/mode_nn.cpp`.
   - In `ArduCopter/config.h`, add a `MODE_NN_ENABLED` define (default 1), following the pattern of the other
     `MODE_*_ENABLED` defines.
   - In `ArduCopter/mode.h`, add `NN = <free number>` to `enum class Number`. Use 40 if it is free; otherwise pick
     a free number and report it. Then `#include "mode_nn.h"` at the end of the file.
   - In `ArduCopter/mode.cpp`, add the `case Mode::Number::NN:` to `Copter::mode_from_mode_num()`, guarded by
     `MODE_NN_ENABLED`.
   - If this tree keeps a list of valid mode numbers elsewhere (parameter `@Values` for `FLTMODE*`, a mode-number
     check in the GCS/MAVLink code, or `mode_reason`), add NN there too.
3. **Copter members.** In `ArduCopter/Copter.h`:
   - include `<AP_NNControl/AP_NNControl.h>` and `<AP_NNControl/AP_MotorsMatrix_NN.h>`;
   - add `AP_MotorsMatrix_NN *motors_nn = nullptr;`;
   - add `ModeNN mode_nn;` next to the other modes (guarded);
   - add `friend class ModeNN;`.
4. **Parameters.**
   - Add `AP_NNControl nn_control;` to `ParametersG2` in `ArduCopter/Parameters.h`.
   - Register it in `ParametersG2::var_info` in `Parameters.cpp` with
     `AP_SUBGROUPINFO(nn_control, "NN_", <next free index>, ParametersG2, AP_NNControl)`.
   - Never reuse an index; add a comment like the neighbouring entries do.
5. **Motors.** In `Copter::allocate_motors()` (`ArduCopter/system.cpp`):
   - For `MOTOR_FRAME_QUAD` and `MOTOR_FRAME_HEXA`, construct `AP_MotorsMatrix_NN` instead of `AP_MotorsMatrix`,
     with **the same constructor arguments this tree passes to `AP_MotorsMatrix`**.
   - Store it in both `motors_nn` and `motors`, and keep `motors_var_info = AP_MotorsMatrix::var_info`.
   - Other frames are unchanged and leave `motors_nn` as `nullptr`.
6. **Pre-arm.** In `AP_Arming_Copter` (`parameter_checks()` or the equivalent in this tree), call
   `copter.g2.nn_control.pre_arm_check(buffer, size)` and fail with that message when it returns false. Do this
   only if `NN_ENABLE` is set; the method already returns true otherwise.
7. **Initialise the network at boot.** After motors and parameters are set up (for example at the end of
   `Copter::init_ardupilot()`):
   - if `g2.nn_control.enabled()`, call
     `g2.nn_control.init(number of motors in motors->get_motor_mask())`;
   - send a GCS warning if it fails.

   `ModeNN::init()` also calls it, but initialising at boot makes the pre-arm check meaningful before the pilot
   selects the mode.
8. **Build.**
   - `./waf configure --board sitl && ./waf copter` must succeed with no new warnings in these files.
   - Then build for the target H7 board the user names (ask if none was given):
     `./waf configure --board <board> && ./waf copter`.
   - Report the flash and RAM use reported by the build before and after, if you can.
9. **SITL check.**
   - Start the vehicle: `sim_vehicle.py -v ArduCopter -f quad --console --map` (and `-f hexa` for the hex network).
   - Set the parameters: `SCHED_LOOP_RATE 400`, `MOT_PWM_MIN 1000`, `MOT_PWM_MAX 2000`, `NN_ENABLE 1`,
     `NN_MASS` = the SITL frame's mass, and `NN_BATT_CELLS` = the SITL battery's cells. Then reboot.
   - Arm in NN mode with throttle centred and raise the target with throttle above centre (or use RC overrides).
   - Check the motors spool up, the vehicle climbs, holds position when the sticks are centred, follows a stick
     input, and that switching to LOITER and to LAND works.
   - Read the `NNC` log message: `Us` is the network update time per loop. Report its mean and maximum, and how
     the flight looked.
   - The network was trained on a different airframe (thrust/weight 2.8, hover about 1600 µs at 16 V), so
     imperfect tracking in SITL is expected and is not something to fix in code. Report it.

## Optional hardware acceleration (STM32H7)

Leave the defaults unless the user asks. They're build defines, not code changes:
- `NN_USE_CMSIS_DSP=1`: dense layers via `arm_mat_vec_mult_f32`. Only when `arm_math.h` is on the include path
  for this board (boards with `HAL_WITH_DSP`).
- `NN_USE_STM32_CORDIC=1`: CORDIC tanh, then parameter `NN_TANH 2`. **Only for STM32H723/725/730/733/735**,
  because other H7s have no CORDIC. The device header must define `CORDIC`, `RCC` and `RCC_AHB2ENR_CORDICEN`.
- `NN_WEIGHTS_SECTION=__attribute__((section("<DTCM section of this board's linker script>")))`: weights in DTCM.
- `NN_WITH_HEX_X=0` or `NN_WITH_QUAD_X=0`: build one frame only, which saves about 34 kB of flash.

## Rules

- Keep changes outside `libraries/AP_NNControl/` minimal, each guarded by `MODE_NN_ENABLED`. A build with
  `MODE_NN_ENABLED 0` must behave exactly like stock ArduPilot.
- Don't change the core `NN_*` files or `models/`. If something in them seems wrong, stop and report it.
- Don't change any default ArduPilot parameter, and don't change behaviour in other modes.
- Follow ArduPilot style: 4-space indent, `@Param` docs on new parameters, `@LoggerMessage` docs on log messages.

## Report at the end

1. Every file you changed, with a one-line reason.
2. Every API you had to adapt in the AP_NNControl files: the old call, the new call, and why.
3. Build results for SITL and the board.
4. SITL results:
   - did it arm, take off, hold and follow the sticks;
   - mode switches;
   - `NNC.Us` mean and maximum;
   - anything that looked wrong.
5. The NN mode number you used.
