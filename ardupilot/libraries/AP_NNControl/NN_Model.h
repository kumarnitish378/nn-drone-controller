/*
  A trained flight network and the simulator settings it was trained with. Instances are generated into
  models/NN_Model_<frame>.h by `python -m nn_drone export-cpp`.
*/
#pragma once

#include <stdint.h>

#include "NN_Config.h"

struct NN_DenseLayer {
    const float *weights;   // [outputs][inputs], row-major: one contiguous row per output neuron
    const float *bias;      // [outputs]
    uint16_t inputs;
    uint16_t outputs;
};

// Integral of the position error that the network sees, with its anti-windup rule ("vehicle and target
// still"): it only integrates within capture_radius of the target while the drone and target are slow.
struct NN_IntegratorSettings {
    float capture_radius;       // size-normalised metres
    float clamp;                // size-normalised metre-seconds
    float input_scale;          // divides the integral before the network sees it
    float settled_speed;        // size-normalised m/s
    float still_target_speed;   // size-normalised m/s
    float target_jump_speed;    // size-normalised m/s; faster target moves count as a jump, not motion
};

struct NN_Model {
    const char *frame;              // "quad_x" or "hex_x": ArduPilot QUAD X / HEXA X motor order
    const char *simulator_profile;
    uint8_t motor_count;
    float control_period_s;         // the loop period the network was trained for
    bool period_scales_with_size;   // true: period * sqrt(cbrt(mass)) (older 100 Hz profiles)
    NN_DenseLayer layers[NN_LAYER_COUNT];
    NN_IntegratorSettings integrator;
};
