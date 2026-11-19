/*
  AP_MotorsMatrix that can hand the motor PWM to the network. ArduPilot keeps everything else: arming, safety
  switch, motor test, spool-up and spool-down. Only while the motors are at full authority
  (SpoolState::THROTTLE_UNLIMITED) and the network has sent a PWM in the last NETWORK_TIMEOUT_MS does the
  network's PWM replace ArduPilot's mixer output.

  The network was trained with PWM 1000..2000 us mapping linearly to throttle: set MOT_PWM_MIN 1000 and
  MOT_PWM_MAX 2000 (network_pwm_range_ok()).
*/
#pragma once

#include <AP_Motors/AP_MotorsMatrix.h>

class AP_MotorsMatrix_NN : public AP_MotorsMatrix {
public:
    using AP_MotorsMatrix::AP_MotorsMatrix;

    // The network's PWM for motors 1..count (ArduPilot motor numbering); call every loop it flies.
    void set_network_pwm(const uint16_t *pwm, uint8_t count);
    // Back to ArduPilot's mixer at once (e.g. on leaving the NN mode).
    void release_network() { _network_active = false; }
    bool network_in_control() const;

    // The PWM last sent to a motor (0-based), by the network or by ArduPilot.
    bool get_motor_pwm(uint8_t motor, uint16_t &pwm) const;

    bool network_pwm_range_ok() const;

    static const uint32_t NETWORK_TIMEOUT_MS = 20;   // 8 missed 400 Hz loops: ArduPilot takes over again
    static const int16_t NETWORK_PWM_MIN = 1000;
    static const int16_t NETWORK_PWM_MAX = 2000;

protected:
    void output_to_motors() override;

private:
    uint16_t _network_pwm[AP_MOTORS_MAX_NUM_MOTORS] {};
    bool _network_active = false;
    uint32_t _network_update_ms = 0;
};
