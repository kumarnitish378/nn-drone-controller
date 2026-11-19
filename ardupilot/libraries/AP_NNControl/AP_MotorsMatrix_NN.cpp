#include "AP_MotorsMatrix_NN.h"

#include <AP_HAL/AP_HAL.h>
#include <SRV_Channel/SRV_Channel.h>

void AP_MotorsMatrix_NN::set_network_pwm(const uint16_t *pwm, uint8_t count)
{
    for (uint8_t motor = 0; motor < AP_MOTORS_MAX_NUM_MOTORS; motor++) {
        _network_pwm[motor] = motor < count ? uint16_t(constrain_int16(pwm[motor], NETWORK_PWM_MIN, NETWORK_PWM_MAX))
                                            : uint16_t(NETWORK_PWM_MIN);
    }
    _network_active = true;
    _network_update_ms = AP_HAL::millis();
}

bool AP_MotorsMatrix_NN::network_in_control() const
{
    return _network_active && armed() && _spool_state == SpoolState::THROTTLE_UNLIMITED &&
           AP_HAL::millis() - _network_update_ms < NETWORK_TIMEOUT_MS;
}

bool AP_MotorsMatrix_NN::get_motor_pwm(uint8_t motor, uint16_t &pwm) const
{
    return SRV_Channels::get_output_pwm(SRV_Channels::get_motor_function(motor), pwm);
}

bool AP_MotorsMatrix_NN::network_pwm_range_ok() const
{
    return get_pwm_output_min() == NETWORK_PWM_MIN && get_pwm_output_max() == NETWORK_PWM_MAX;
}

void AP_MotorsMatrix_NN::output_to_motors()
{
    if (!network_in_control()) {
        AP_MotorsMatrix::output_to_motors();
        return;
    }
    for (uint8_t motor = 0; motor < AP_MOTORS_MAX_NUM_MOTORS; motor++) {
        if (motor_enabled[motor]) {
            rc_write(motor, _network_pwm[motor]);
        }
    }
}
