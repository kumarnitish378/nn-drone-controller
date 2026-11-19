/*
  tanh on the CORDIC co-processor of STM32H723/725/730/733/735 (RM0468 section "CORDIC co-processor").

  The CORDIC computes cosh and sinh together for |x| < 1.118; tanh = sinh / cosh. Larger arguments are
  halved until they fit and the result is doubled back with tanh(2y) = 2t / (1 + t^2).
  Built only with NN_USE_STM32_CORDIC=1.
*/
#pragma once

#include <stdint.h>

void nn_cordic_init();
void nn_cordic_tanh(float *values, uint16_t count);
