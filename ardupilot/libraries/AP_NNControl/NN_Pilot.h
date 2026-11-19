/*
  The network as a flight controller: builds its inputs from the vehicle state exactly as the simulator it was
  trained in does (nn_drone/simulation/observation.py), keeps the history frames and the position-error
  integral, runs the network and returns one command per motor.

  Conventions are ArduPilot's: NED world frame, FRD body frame, yaw clockwise from north. Internally the
  network works in the simulator's North-West-Up / Forward-Left-Up frames; the conversion is done here.

  Call once per control loop at the rate the model was trained for (model.control_period_s, 400 Hz):
      pilot.update(sensors, target, commands);   // network commands, -1..1 per motor
      ... send commands, or whatever the motors really get while spooling up ...
      pilot.commit(applied);                      // what the motors were given this loop
  No heap allocation; about 3 kB of state.
*/
#pragma once

#include "NN_Model.h"
#include "NN_Network.h"

struct NN_Vector3 {
    float x, y, z;
};

struct NN_Sensors {
    float body_to_ned[3][3];   // attitude as a rotation matrix (AP_AHRS::get_rotation_body_to_ned())
    NN_Vector3 gyro;           // body rates, FRD, rad/s
    NN_Vector3 accel;          // specific force, FRD, m/s^2 (about (0, 0, -9.81) level at rest)
    NN_Vector3 position;       // NED metres from any fixed origin (the same one as the target)
    NN_Vector3 velocity;       // NED m/s
    float battery_voltage;     // volts, whole pack
};

struct NN_Target {
    NN_Vector3 position;   // NED metres, same origin as NN_Sensors::position
    float yaw;             // radians, clockwise from north
};

struct NN_PilotConfig {
    float nominal_mass_kg = 1.0f;       // the network scales lengths by cbrt(mass) and times by sqrt(that)
    float position_error_clip = 2.5f;   // size-normalised metres: the target is never shown further away
    uint8_t battery_cells = 4;          // the network was trained on a 4S pack; other packs are rescaled
};

class NN_Pilot {
public:
    NN_Pilot(const NN_Model &model, const NN_PilotConfig &config);

    // Start of a flight (e.g. arming). motor_command: what the motors get now, -1 = stopped (1000 us).
    void reset(float motor_command = -1.0f);

    // false if the model does not have the input layout this library builds (then commands are untouched).
    bool update(const NN_Sensors &sensors, const NN_Target &target, float *commands);
    void commit(const float *applied_commands);

    static uint16_t command_to_pwm(float command);
    static float pwm_to_command(uint16_t pwm_us);

    NN_Network &network() { return _network; }
    const NN_Model &model() const { return _model; }
    uint8_t motor_count() const { return _model.motor_count; }
    uint16_t input_count() const { return _model.layers[0].inputs; }
    const float *last_inputs() const { return _inputs; }
    const float *integral_ned() const { return _integral; }   // size-normalised metre-seconds
    uint8_t history_spacing() const { return _history_spacing; }
    bool valid() const { return _valid; }

private:
    void integrate(const NN_Sensors &sensors);
    uint16_t build_frame(const NN_Sensors &sensors, const NN_Target &target, float *features) const;
    void remember_history(const float *features);
    const float *history_frame(uint8_t age) const;

    static const uint8_t MAX_HISTORY_SPACING = 16;
    static const uint8_t HISTORY_FEATURES_MAX = 14 + NN_MAX_MOTORS;

    const NN_Model &_model;
    NN_Network _network;
    float _position_error_clip;
    float _battery_scale;
    float _length_scale;
    float _time_scale;
    float _control_period;
    float _step_length;   // control period in reference steps (10 ms x sqrt(length scale))
    float _mass_class;
    uint8_t _history_spacing;
    uint8_t _history_capacity;
    uint8_t _history_features;

    float _previous_command[NN_MAX_MOTORS];
    float _integral[3];
    NN_Vector3 _target_position;
    NN_Vector3 _reference_target;   // the target one loop earlier, to tell motion from a jump
    bool _valid;
    bool _have_target;
    bool _have_reference;
    bool _history_needs_fill;
    uint8_t _newest_history_slot;
    float _history[2 * MAX_HISTORY_SPACING][HISTORY_FEATURES_MAX];
    float _latest_history[HISTORY_FEATURES_MAX];
    float _inputs[NN_MAX_INPUTS];
};
