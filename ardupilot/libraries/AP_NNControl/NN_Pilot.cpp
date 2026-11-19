#include "NN_Pilot.h"

#include <math.h>
#include <string.h>

namespace {

// Input scaling: the constants of nn_drone/simulation/observation.py.
const float GRAVITY_MSS = 9.81f;
const float GYRO_SCALE_RAD_S = 8.0f;
const float ACCELEROMETER_SCALE = 2.0f * GRAVITY_MSS;
const float POSITION_ERROR_SCALE_M = 5.0f;
const float VELOCITY_SCALE_MPS = 5.0f;
const float BATTERY_CENTER_V = 15.6f;
const float BATTERY_SPAN_V = 1.2f;
const uint8_t TRAINING_BATTERY_CELLS = 4;
const float MASS_CLASS_REFERENCE_DECADE = 1.0f;
const float MIN_ERROR_NORM = 1e-6f;
const float REFERENCE_CONTROL_PERIOD_S = 0.01f;
const float PI = 3.14159265358979f;

const float PWM_CENTER_US = 1500.0f;
const float PWM_HALF_RANGE_US = 500.0f;

// Layout of one frame of features (before history and mass class).
const uint8_t GYRO_TO_PITCH = 8;          // gyro(3) accel(3) roll pitch
const uint8_t HEADING_SIN = 8;
const uint8_t POSITION_ERROR = 10;        // position error(3) velocity(3)
const uint8_t POSITION_AND_VELOCITY = 6;
const uint8_t BATTERY = 16;
const uint8_t PREVIOUS_COMMANDS = 17;

float clamp(float value, float low, float high) { return value < low ? low : (value > high ? high : value); }
float length_of(const float vector[3]) { return sqrtf(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]); }

float wrap_pi(float angle)
{
    const float turn = 2.0f * PI;
    angle = fmodf(angle + PI, turn);
    if (angle < 0.0f) {
        angle += turn;
    }
    return angle - PI;
}

// NED world vector -> the simulator's Forward-Left-Up body frame.
void world_to_body_flu(const float body_to_ned[3][3], const float ned[3], float flu[3])
{
    float frd[3];
    for (uint8_t axis = 0; axis < 3; axis++) {
        frd[axis] = body_to_ned[0][axis] * ned[0] + body_to_ned[1][axis] * ned[1] + body_to_ned[2][axis] * ned[2];
    }
    flu[0] = frd[0];
    flu[1] = -frd[1];
    flu[2] = -frd[2];
}

}  // namespace

NN_Pilot::NN_Pilot(const NN_Model &model, const NN_PilotConfig &config) :
    _model(model),
    _network(model)
{
    _position_error_clip = config.position_error_clip;
    _battery_scale = float(TRAINING_BATTERY_CELLS) / float(config.battery_cells > 0 ? config.battery_cells : TRAINING_BATTERY_CELLS);
    _length_scale = cbrtf(config.nominal_mass_kg);
    _time_scale = sqrtf(_length_scale);
    _control_period = model.period_scales_with_size ? model.control_period_s * _time_scale : model.control_period_s;
    _step_length = model.period_scales_with_size ? 1.0f : model.control_period_s / (REFERENCE_CONTROL_PERIOD_S * _time_scale);
    _mass_class = log10f(config.nominal_mass_kg) - MASS_CLASS_REFERENCE_DECADE;
    // Earlier frames are one reference step apart; nearbyintf rounds half to even like numpy.rint.
    const float spacing = nearbyintf(1.0f / _step_length);
    _history_spacing = uint8_t(clamp(spacing, 1.0f, float(MAX_HISTORY_SPACING)));
    _history_capacity = uint8_t(NN_HISTORY_FRAMES * _history_spacing);
    _history_features = uint8_t(GYRO_TO_PITCH + POSITION_AND_VELOCITY + model.motor_count);
    const uint16_t frame_features = PREVIOUS_COMMANDS + model.motor_count + 3;   // + integral(3)
    _valid = model.motor_count <= NN_MAX_MOTORS && model.layers[0].inputs <= NN_MAX_INPUTS &&
             model.layers[0].inputs == frame_features + NN_HISTORY_FRAMES * _history_features + 1;
    reset();
}

void NN_Pilot::reset(float motor_command)
{
    for (uint8_t motor = 0; motor < NN_MAX_MOTORS; motor++) {
        _previous_command[motor] = clamp(motor_command, -1.0f, 1.0f);
    }
    memset(_integral, 0, sizeof(_integral));
    _have_target = false;
    _have_reference = false;
    _history_needs_fill = true;
    _newest_history_slot = 0;
}

uint16_t NN_Pilot::command_to_pwm(float command)
{
    return uint16_t(lrintf(PWM_CENTER_US + PWM_HALF_RANGE_US * clamp(command, -1.0f, 1.0f)));
}

float NN_Pilot::pwm_to_command(uint16_t pwm_us)
{
    return clamp((float(pwm_us) - PWM_CENTER_US) / PWM_HALF_RANGE_US, -1.0f, 1.0f);
}

// The integral of the position error towards the previous loop's target, measured now: the simulator
// integrates after each physics step, which is this loop's sensor reading.
void NN_Pilot::integrate(const NN_Sensors &sensors)
{
    const NN_IntegratorSettings &settings = _model.integrator;
    const float error[3] = {_target_position.x - sensors.position.x, _target_position.y - sensors.position.y,
                            _target_position.z - sensors.position.z};
    const float velocity[3] = {sensors.velocity.x, sensors.velocity.y, sensors.velocity.z};
    float target_velocity[3] = {0.0f, 0.0f, 0.0f};
    if (_have_reference) {
        target_velocity[0] = (_target_position.x - _reference_target.x) / _control_period;
        target_velocity[1] = (_target_position.y - _reference_target.y) / _control_period;
        target_velocity[2] = (_target_position.z - _reference_target.z) / _control_period;
        if (length_of(target_velocity) > settings.target_jump_speed * _time_scale) {
            memset(target_velocity, 0, sizeof(target_velocity));   // a waypoint jump, not a moving target
        }
    }
    _reference_target = _target_position;
    _have_reference = true;

    const bool settled = length_of(error) / _length_scale < settings.capture_radius &&
                         length_of(velocity) / _time_scale < settings.settled_speed &&
                         length_of(target_velocity) / _time_scale < settings.still_target_speed;
    if (!settled) {
        return;
    }
    const float increment = REFERENCE_CONTROL_PERIOD_S * _step_length / _length_scale;
    for (uint8_t axis = 0; axis < 3; axis++) {
        _integral[axis] = clamp(_integral[axis] + error[axis] * increment, -settings.clamp, settings.clamp);
    }
}

uint16_t NN_Pilot::build_frame(const NN_Sensors &sensors, const NN_Target &target, float *features) const
{
    const float (*rotation)[3] = sensors.body_to_ned;
    const float roll = atan2f(rotation[2][1], rotation[2][2]);
    const float pitch_ned = -asinf(clamp(rotation[2][0], -1.0f, 1.0f));
    const float yaw_ned = atan2f(rotation[1][0], rotation[0][0]);
    // The simulator's yaw and pitch turn the other way (Z up, Y left).
    const float heading_error = wrap_pi(yaw_ned - target.yaw);

    const float error_ned[3] = {target.position.x - sensors.position.x, target.position.y - sensors.position.y,
                                target.position.z - sensors.position.z};
    const float velocity_ned[3] = {sensors.velocity.x, sensors.velocity.y, sensors.velocity.z};
    float error[3], velocity[3], integral[3];
    world_to_body_flu(rotation, error_ned, error);
    world_to_body_flu(rotation, velocity_ned, velocity);
    world_to_body_flu(rotation, _integral, integral);

    float error_norm = 0.0f;
    for (uint8_t axis = 0; axis < 3; axis++) {
        error[axis] /= _length_scale;
        velocity[axis] /= _time_scale;
        error_norm += error[axis] * error[axis];
    }
    const float shrink = fminf(1.0f, _position_error_clip / fmaxf(sqrtf(error_norm), MIN_ERROR_NORM));

    const float gyro[3] = {sensors.gyro.x, -sensors.gyro.y, -sensors.gyro.z};
    const float accel[3] = {sensors.accel.x, -sensors.accel.y, -sensors.accel.z};
    uint16_t index = 0;
    for (uint8_t axis = 0; axis < 3; axis++) {
        features[index++] = gyro[axis] * _time_scale / GYRO_SCALE_RAD_S;
    }
    for (uint8_t axis = 0; axis < 3; axis++) {
        features[index++] = accel[axis] / ACCELEROMETER_SCALE;
    }
    features[index++] = roll / PI;
    features[index++] = -pitch_ned / PI;
    features[index++] = sinf(heading_error);
    features[index++] = cosf(heading_error);
    for (uint8_t axis = 0; axis < 3; axis++) {
        features[index++] = error[axis] * shrink / POSITION_ERROR_SCALE_M;
    }
    for (uint8_t axis = 0; axis < 3; axis++) {
        features[index++] = velocity[axis] / VELOCITY_SCALE_MPS;
    }
    features[index++] = (sensors.battery_voltage * _battery_scale - BATTERY_CENTER_V) / BATTERY_SPAN_V;
    for (uint8_t motor = 0; motor < _model.motor_count; motor++) {
        features[index++] = _previous_command[motor];
    }
    for (uint8_t axis = 0; axis < 3; axis++) {
        features[index++] = integral[axis] / _model.integrator.input_scale;
    }
    return index;
}

// Gyro, accelerometer, roll, pitch, position error, velocity and previous commands: no heading, no battery.
void NN_Pilot::remember_history(const float *features)
{
    memcpy(_latest_history, features, GYRO_TO_PITCH * sizeof(float));
    memcpy(_latest_history + GYRO_TO_PITCH, features + POSITION_ERROR, POSITION_AND_VELOCITY * sizeof(float));
    memcpy(_latest_history + GYRO_TO_PITCH + POSITION_AND_VELOCITY, features + PREVIOUS_COMMANDS,
           _model.motor_count * sizeof(float));
    if (_history_needs_fill) {
        for (uint8_t slot = 0; slot < _history_capacity; slot++) {
            memcpy(_history[slot], _latest_history, _history_features * sizeof(float));
        }
        _history_needs_fill = false;
    }
}

// age 1 = one reference step ago, age 2 = two.
const float *NN_Pilot::history_frame(uint8_t age) const
{
    const int slot = (int(_newest_history_slot) - (int(age) * _history_spacing - 1) + 2 * _history_capacity) % _history_capacity;
    return _history[slot];
}

bool NN_Pilot::update(const NN_Sensors &sensors, const NN_Target &target, float *commands)
{
    if (!_valid) {
        return false;
    }
    if (_have_target) {
        integrate(sensors);
    }
    uint16_t count = build_frame(sensors, target, _inputs);
    remember_history(_inputs);
    for (uint8_t age = 1; age <= NN_HISTORY_FRAMES; age++) {
        memcpy(_inputs + count, history_frame(age), _history_features * sizeof(float));
        count += _history_features;
    }
    _inputs[count++] = _mass_class;
    _network.run(_inputs, commands);
    _target_position = target.position;
    _have_target = true;
    return true;
}

void NN_Pilot::commit(const float *applied_commands)
{
    _newest_history_slot = uint8_t((_newest_history_slot + 1) % _history_capacity);
    memcpy(_history[_newest_history_slot], _latest_history, _history_features * sizeof(float));
    for (uint8_t motor = 0; motor < _model.motor_count; motor++) {
        _previous_command[motor] = clamp(applied_commands[motor], -1.0f, 1.0f);
    }
}
