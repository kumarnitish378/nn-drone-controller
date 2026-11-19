#include "NN_Config.h"

#if NN_USE_STM32_CORDIC

#include "NN_Stm32Cordic.h"

#include <math.h>

#ifndef NN_CORDIC_DEVICE_HEADER
#define NN_CORDIC_DEVICE_HEADER "hal.h"
#endif
#include NN_CORDIC_DEVICE_HEADER

namespace {

// CORDIC_CSR fields (RM0468).
const uint32_t CSR_FUNCTION_HYPERBOLIC_COSINE = 5u;   // primary result cosh, secondary sinh
const uint32_t CSR_PRECISION_SHIFT = 4;
const uint32_t CSR_PRECISION_24_ITERATIONS = 6u;      // about 2^-20 error, 6 clock cycles
const uint32_t CSR_SCALE_SHIFT = 8;
const uint32_t CSR_SCALE_HYPERBOLIC = 1u;             // hyperbolic functions require scale 1: x is written as x / 2
const uint32_t CSR_TWO_RESULTS = 1u << 19;            // NRES: read cosh and sinh
// NARGS = 0 (one argument), RESSIZE = ARGSIZE = 0 (32-bit q1.31).

const float HYPERBOLIC_INPUT_SCALE = 0.5f;            // 2^-scale
const float LARGEST_DIRECT_ARGUMENT = 1.1f;           // below the 1.118 limit of the hyperbolic functions
const float TANH_SATURATION = 9.0f;                   // tanh(9) = 1 - 3e-8: no need to ask the CORDIC
const float Q31_ONE = 2147483648.0f;

int32_t to_q31(float value) { return int32_t(value * Q31_ONE); }

float cordic_tanh(float value)
{
    float magnitude = fabsf(value);
    if (magnitude >= TANH_SATURATION) {
        return value > 0.0f ? 1.0f : -1.0f;
    }
    uint8_t halvings = 0;
    while (magnitude > LARGEST_DIRECT_ARGUMENT) {
        magnitude *= 0.5f;
        halvings++;
    }
    CORDIC->WDATA = uint32_t(to_q31(magnitude * HYPERBOLIC_INPUT_SCALE));
    // Reading RDATA stalls the bus until the result is ready (zero-overhead mode): no polling needed.
    const int32_t half_cosh = int32_t(CORDIC->RDATA);
    const int32_t half_sinh = int32_t(CORDIC->RDATA);
    float result = float(half_sinh) / float(half_cosh);
    while (halvings-- > 0) {
        result = 2.0f * result / (1.0f + result * result);
    }
    return value < 0.0f ? -result : result;
}

}  // namespace

void nn_cordic_init()
{
    RCC->AHB2ENR |= RCC_AHB2ENR_CORDICEN;
    (void)RCC->AHB2ENR;   // read back so the clock is running before the first register access
    CORDIC->CSR = CSR_FUNCTION_HYPERBOLIC_COSINE | (CSR_PRECISION_24_ITERATIONS << CSR_PRECISION_SHIFT) |
                  (CSR_SCALE_HYPERBOLIC << CSR_SCALE_SHIFT) | CSR_TWO_RESULTS;
}

void nn_cordic_tanh(float *values, uint16_t count)
{
    for (uint16_t index = 0; index < count; index++) {
        values[index] = cordic_tanh(values[index]);
    }
}

#endif  // NN_USE_STM32_CORDIC
