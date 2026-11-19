#include "AP_NNControl.h"

#include <new>

#include <AP_AHRS/AP_AHRS.h>
#include <AP_BattMonitor/AP_BattMonitor.h>
#include <AP_HAL/AP_HAL.h>
#include <AP_InertialSensor/AP_InertialSensor.h>
#include <AP_Logger/AP_Logger.h>
#include <AP_Scheduler/AP_Scheduler.h>

// The arrays live here only: each model header defines its weights (about 34 kB of flash per frame).
#ifndef NN_WITH_QUAD_X
#define NN_WITH_QUAD_X 1
#endif
#ifndef NN_WITH_HEX_X
#define NN_WITH_HEX_X 1
#endif
#if NN_WITH_QUAD_X
#include "models/NN_Model_quad_x.h"
#endif
#if NN_WITH_HEX_X
#include "models/NN_Model_hex_x.h"
#endif

extern const AP_HAL::HAL &hal;

namespace {

const uint8_t QUAD_MOTORS = 4;
const uint8_t HEX_MOTORS = 6;
const float MASS_MIN_KG = 0.2f;
const float MASS_MAX_KG = 500.0f;
const float LOOP_RATE_TOLERANCE_HZ = 1.0f;

const NN_Model *model_for(uint8_t motor_count)
{
    switch (motor_count) {
#if NN_WITH_QUAD_X
    case QUAD_MOTORS:
        return &nn_model_quad_x::model;
#endif
#if NN_WITH_HEX_X
    case HEX_MOTORS:
        return &nn_model_hex_x::model;
#endif
    default:
        return nullptr;
    }
}

}  // namespace

const AP_Param::GroupInfo AP_NNControl::var_info[] = {
    // @Param: ENABLE
    // @DisplayName: Neural-network flight control
    // @Description: Lets the NN flight mode fly the vehicle with the trained network
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO_FLAGS("ENABLE", 1, AP_NNControl, _enable, 0, AP_PARAM_FLAG_ENABLE),

    // @Param: MASS
    // @DisplayName: Vehicle mass for the network
    // @Description: Take-off mass. The network scales lengths by the cube root of it and times by the square root of that
    // @Units: kg
    // @Range: 0.2 500
    // @User: Advanced
    AP_GROUPINFO("MASS", 2, AP_NNControl, _mass_kg, 1.0f),

    // @Param: ERR_CLIP
    // @DisplayName: Target distance shown to the network
    // @Description: The target is never shown further away than this, in size-normalised metres (metres / cube root of mass)
    // @Range: 0.5 5
    // @User: Advanced
    AP_GROUPINFO("ERR_CLIP", 3, AP_NNControl, _error_clip, 2.5f),

    // @Param: BATT_CELLS
    // @DisplayName: Battery cells in series
    // @Description: The network was trained on a 4S pack; other packs are rescaled to 4S volts
    // @Range: 1 14
    // @User: Advanced
    AP_GROUPINFO("BATT_CELLS", 4, AP_NNControl, _battery_cells, 4),

    // @Param: TANH
    // @DisplayName: tanh backend
    // @Description: How the network's tanh is computed. CORDIC needs an STM32H72x/H73x and a build with NN_USE_STM32_CORDIC
    // @Values: 0:C library,1:Fast FPU approximation,2:CORDIC co-processor
    // @User: Advanced
    AP_GROUPINFO("TANH", 5, AP_NNControl, _tanh_backend, 0),

    // @Param: LOG
    // @DisplayName: Log the network
    // @Description: Writes the NNC log message (inference time, integral, motor PWM) every loop
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("LOG", 6, AP_NNControl, _log_enable, 1),

    AP_GROUPEND
};

AP_NNControl::AP_NNControl() :
    _pilot(nullptr),
    _last_update_us(0)
{
    AP_Param::setup_object_defaults(this, var_info);
}

bool AP_NNControl::init(uint8_t motor_count)
{
    const NN_Model *model = model_for(motor_count);
    if (model == nullptr || model->period_scales_with_size) {
        return false;
    }
    const float loop_rate_hz = AP::scheduler().get_loop_rate_hz();
    if (fabsf(loop_rate_hz - 1.0f / model->control_period_s) > LOOP_RATE_TOLERANCE_HZ) {
        return false;
    }
    if (_pilot != nullptr && &_pilot->model() == model) {
        return true;
    }
    NN_PilotConfig config;
    config.nominal_mass_kg = constrain_float(_mass_kg, MASS_MIN_KG, MASS_MAX_KG);
    config.position_error_clip = _error_clip;
    config.battery_cells = uint8_t(constrain_int16(_battery_cells, 1, 14));
    if (_pilot != nullptr) {
        _pilot->~NN_Pilot();
    }
    _pilot = new (_pilot_storage) NN_Pilot(*model, config);
    if (!_pilot->valid()) {
        _pilot->~NN_Pilot();
        _pilot = nullptr;
        return false;
    }
    _pilot->network().set_activation(requested_activation());
    return true;
}

NN_Activation AP_NNControl::requested_activation() const
{
    switch (_tanh_backend.get()) {
    case 1:
        return NN_Activation::FastApproximation;
    case 2:
        return NN_Activation::Cordic;
    default:
        return NN_Activation::Exact;
    }
}

bool AP_NNControl::pre_arm_check(char *failure_message, uint8_t length) const
{
    if (!enabled()) {
        return true;
    }
    if (_pilot == nullptr) {
        hal.util->snprintf(failure_message, length, "NN: no network for this frame, or SCHED_LOOP_RATE is not 400");
        return false;
    }
    if (_pilot->network().activation() != requested_activation()) {
        hal.util->snprintf(failure_message, length, "NN: NN_TANH backend not built in");
        return false;
    }
    return true;
}

void AP_NNControl::reset(uint16_t motor_pwm)
{
    if (_pilot != nullptr) {
        _pilot->reset(NN_Pilot::pwm_to_command(motor_pwm));
    }
}

bool AP_NNControl::update(const Vector3f &target_position, float target_yaw, const Vector3f &position,
                          const Vector3f &velocity, uint16_t pwm_out[NN_MAX_MOTORS])
{
    if (_pilot == nullptr) {
        return false;
    }
    const uint32_t start_us = AP_HAL::micros();
    const AP_AHRS &ahrs = AP::ahrs();
    const Matrix3f &body_to_ned = ahrs.get_rotation_body_to_ned();
    const Vector3f &gyro = ahrs.get_gyro();
    const Vector3f &accel = AP::ins().get_accel();

    NN_Sensors sensors;
    const Vector3f *rows[3] = {&body_to_ned.a, &body_to_ned.b, &body_to_ned.c};
    for (uint8_t row = 0; row < 3; row++) {
        sensors.body_to_ned[row][0] = rows[row]->x;
        sensors.body_to_ned[row][1] = rows[row]->y;
        sensors.body_to_ned[row][2] = rows[row]->z;
    }
    sensors.gyro = NN_Vector3{gyro.x, gyro.y, gyro.z};
    sensors.accel = NN_Vector3{accel.x, accel.y, accel.z};
    sensors.position = NN_Vector3{position.x, position.y, position.z};
    sensors.velocity = NN_Vector3{velocity.x, velocity.y, velocity.z};
    sensors.battery_voltage = AP::battery().voltage();
    const NN_Target target{NN_Vector3{target_position.x, target_position.y, target_position.z}, target_yaw};

    float commands[NN_MAX_MOTORS];
    if (!_pilot->update(sensors, target, commands)) {
        return false;
    }
    for (uint8_t motor = 0; motor < _pilot->motor_count(); motor++) {
        pwm_out[motor] = NN_Pilot::command_to_pwm(commands[motor]);
    }
    _last_update_us = AP_HAL::micros() - start_us;
    write_log(pwm_out);
    return true;
}

void AP_NNControl::commit(const uint16_t applied_pwm[NN_MAX_MOTORS])
{
    if (_pilot == nullptr) {
        return;
    }
    float applied[NN_MAX_MOTORS];
    for (uint8_t motor = 0; motor < _pilot->motor_count(); motor++) {
        applied[motor] = NN_Pilot::pwm_to_command(applied_pwm[motor]);
    }
    _pilot->commit(applied);
}

void AP_NNControl::write_log(const uint16_t pwm[NN_MAX_MOTORS]) const
{
#if HAL_LOGGING_ENABLED
    if (_log_enable == 0) {
        return;
    }
    const float *integral = _pilot->integral_ned();
    uint16_t motor_pwm[HEX_MOTORS] = {0, 0, 0, 0, 0, 0};
    for (uint8_t motor = 0; motor < _pilot->motor_count() && motor < HEX_MOTORS; motor++) {
        motor_pwm[motor] = pwm[motor];
    }
    // @LoggerMessage: NNC
    // @Description: Neural-network flight control, every loop
    // @Field: TimeUS: Time since system startup
    // @Field: Us: Network update time (inputs + inference), microseconds
    // @Field: IN,IE,ID: Position-error integral seen by the network (NED, size-normalised m.s)
    // @Field: M1,M2,M3,M4,M5,M6: Network PWM per motor
    AP::logger().WriteStreaming("NNC", "TimeUS,Us,IN,IE,ID,M1,M2,M3,M4,M5,M6", "QIfffHHHHHH",
                                AP_HAL::micros64(), _last_update_us, integral[0], integral[1], integral[2],
                                motor_pwm[0], motor_pwm[1], motor_pwm[2], motor_pwm[3], motor_pwm[4], motor_pwm[5]);
#else
    (void)pwm;
#endif
}
