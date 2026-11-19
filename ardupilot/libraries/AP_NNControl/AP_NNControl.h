/*
  ArduPilot front end of the NN flight library: parameters (NN_*), sensor reading from AHRS, INS and the
  battery monitor, and the network step. The vehicle code supplies the target and the EKF position/velocity
  and sends the PWM to the motors (see AP_MotorsMatrix_NN and ArduCopter/mode_nn.cpp).

  Run update() and commit() once per main loop (SCHED_LOOP_RATE must be the rate the model was trained for,
  400 Hz) from arming to disarming, also while the motors spool up, so the network's history is continuous.
*/
#pragma once

#include <AP_Math/AP_Math.h>
#include <AP_Param/AP_Param.h>

#include "NN_Pilot.h"

class AP_NNControl {
public:
    AP_NNControl();

    AP_NNControl(const AP_NNControl &other) = delete;
    AP_NNControl &operator=(const AP_NNControl &other) = delete;

    static const struct AP_Param::GroupInfo var_info[];

    bool enabled() const { return _enable != 0; }

    // Picks the network for the frame (4 motors: QUAD X, 6: HEXA X). False if there is none or the main
    // loop does not run at the model's rate.
    bool init(uint8_t motor_count);
    bool initialised() const { return _pilot != nullptr; }

    bool pre_arm_check(char *failure_message, uint8_t length) const;

    // Start of a flight; motor_pwm is what the motors get now (1000 = stopped).
    void reset(uint16_t motor_pwm = PWM_STOPPED_US);

    // target / position: NED metres from the EKF origin; velocity: NED m/s; target_yaw: radians from north.
    // Fills pwm_out[motor_count()] with the network's command. False if not initialised.
    bool update(const Vector3f &target_position, float target_yaw, const Vector3f &position, const Vector3f &velocity,
                uint16_t pwm_out[NN_MAX_MOTORS]);
    // What the motors were given this loop (the network's PWM, or ArduPilot's while spooling up).
    void commit(const uint16_t applied_pwm[NN_MAX_MOTORS]);

    uint8_t motor_count() const { return _pilot != nullptr ? _pilot->motor_count() : 0; }
    float nominal_mass_kg() const { return _mass_kg; }
    float error_clip() const { return _error_clip; }
    uint32_t last_update_us() const { return _last_update_us; }

    static const uint16_t PWM_STOPPED_US = 1000;

private:
    void write_log(const uint16_t pwm[NN_MAX_MOTORS]) const;
    NN_Activation requested_activation() const;

    AP_Int8 _enable;
    AP_Float _mass_kg;
    AP_Float _error_clip;
    AP_Int8 _battery_cells;
    AP_Int8 _tanh_backend;
    AP_Int8 _log_enable;

    NN_Pilot *_pilot;
    alignas(NN_Pilot) uint8_t _pilot_storage[sizeof(NN_Pilot)];   // no heap: the pilot is built in place
    uint32_t _last_update_us;
};
