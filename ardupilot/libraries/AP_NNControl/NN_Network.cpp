#include "NN_Network.h"

#include <math.h>

#if NN_USE_CMSIS_DSP
#include <arm_math.h>
#endif

#if NN_USE_STM32_CORDIC
#include "NN_Stm32Cordic.h"
#endif

namespace {

// [9/8] Pade approximant of tanh (the continued fraction x / (1 + x^2 / (3 + x^2 / (5 + ...))) cut after 17),
// in powers of x^2. Largest error 6.9e-6 in float, replaced by +-1 beyond FAST_TANH_SATURATION where that is
// closer: 8 multiply-adds and one division.
const float FAST_TANH_SATURATION = 6.3f;
const float PADE_NUMERATOR[] = {34459425.0f, 4729725.0f, 135135.0f, 990.0f, 1.0f};
const float PADE_DENOMINATOR[] = {34459425.0f, 16216200.0f, 945945.0f, 13860.0f, 45.0f};

}  // namespace

NN_Network::NN_Network(const NN_Model &model) : _model(model) {}

bool NN_Network::set_activation(NN_Activation activation)
{
    if (activation == NN_Activation::Cordic) {
#if NN_USE_STM32_CORDIC
        nn_cordic_init();
#else
        return false;
#endif
    }
    _activation = activation;
    return true;
}

float NN_Network::fast_tanh(float value)
{
    if (value >= FAST_TANH_SATURATION) {
        return 1.0f;
    }
    if (value <= -FAST_TANH_SATURATION) {
        return -1.0f;
    }
    const float square = value * value;
    const float numerator = value * (PADE_NUMERATOR[0] + square * (PADE_NUMERATOR[1] + square * (PADE_NUMERATOR[2] +
                                     square * (PADE_NUMERATOR[3] + square * PADE_NUMERATOR[4]))));
    const float denominator = PADE_DENOMINATOR[0] + square * (PADE_DENOMINATOR[1] + square * (PADE_DENOMINATOR[2] +
                              square * (PADE_DENOMINATOR[3] + square * PADE_DENOMINATOR[4])));
    const float approximation = numerator / denominator;
    // The approximant overshoots 1 for large |x|; tanh never does.
    return approximation > 1.0f ? 1.0f : (approximation < -1.0f ? -1.0f : approximation);
}

void NN_Network::dense(const NN_DenseLayer &layer, const float *inputs, float *outputs) const
{
#if NN_USE_CMSIS_DSP
    arm_matrix_instance_f32 matrix;
    arm_mat_init_f32(&matrix, layer.outputs, layer.inputs, const_cast<float *>(layer.weights));
    arm_mat_vec_mult_f32(&matrix, inputs, outputs);
    for (uint16_t output = 0; output < layer.outputs; output++) {
        outputs[output] += layer.bias[output];
    }
#else
    const uint16_t count = layer.inputs;
    for (uint16_t output = 0; output < layer.outputs; output++) {
        const float *row = layer.weights + uint32_t(output) * count;
        // Four independent sums hide the FPU's multiply-add latency.
        float sum0 = layer.bias[output], sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        uint16_t input = 0;
        for (; input + 4 <= count; input += 4) {
            sum0 += row[input] * inputs[input];
            sum1 += row[input + 1] * inputs[input + 1];
            sum2 += row[input + 2] * inputs[input + 2];
            sum3 += row[input + 3] * inputs[input + 3];
        }
        for (; input < count; input++) {
            sum0 += row[input] * inputs[input];
        }
        outputs[output] = (sum0 + sum1) + (sum2 + sum3);
    }
#endif
}

void NN_Network::activate(float *values, uint16_t count) const
{
    switch (_activation) {
    case NN_Activation::FastApproximation:
        for (uint16_t index = 0; index < count; index++) {
            values[index] = fast_tanh(values[index]);
        }
        break;
#if NN_USE_STM32_CORDIC
    case NN_Activation::Cordic:
        nn_cordic_tanh(values, count);
        break;
#endif
    default:
        for (uint16_t index = 0; index < count; index++) {
            values[index] = tanhf(values[index]);
        }
        break;
    }
}

void NN_Network::run(const float *inputs, float *outputs)
{
    const NN_DenseLayer *layers = _model.layers;
    dense(layers[0], inputs, _hidden1);
    activate(_hidden1, layers[0].outputs);
    dense(layers[1], _hidden1, _hidden2);
    activate(_hidden2, layers[1].outputs);
    dense(layers[2], _hidden2, outputs);
    activate(outputs, layers[2].outputs);
}
