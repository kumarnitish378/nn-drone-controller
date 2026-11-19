/*
  Runs the flight network: three dense layers, tanh after each. No allocation, no ArduPilot dependency.
*/
#pragma once

#include "NN_Model.h"

enum class NN_Activation : uint8_t {
    Exact = 0,              // tanhf from the C library: matches training to float precision
    FastApproximation = 1,  // rational approximation on the FPU: 8 multiply-adds + 1 divide, error below 7e-6
    Cordic = 2,             // STM32H72x/H73x CORDIC co-processor (needs NN_USE_STM32_CORDIC)
};

class NN_Network {
public:
    explicit NN_Network(const NN_Model &model);

    // Returns false (and keeps the previous choice) when the backend is not built in.
    bool set_activation(NN_Activation activation);
    NN_Activation activation() const { return _activation; }

    // inputs: model.layers[0].inputs values; outputs: one value per motor in -1..1.
    void run(const float *inputs, float *outputs);

    static float fast_tanh(float value);

private:
    void dense(const NN_DenseLayer &layer, const float *inputs, float *outputs) const;
    void activate(float *values, uint16_t count) const;

    const NN_Model &_model;
    NN_Activation _activation = NN_Activation::Exact;
    float _hidden1[NN_MAX_HIDDEN];
    float _hidden2[NN_MAX_HIDDEN];
};
