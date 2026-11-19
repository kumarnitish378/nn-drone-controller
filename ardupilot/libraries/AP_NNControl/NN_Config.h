/*
  Build options for the NN flight library. Every option can be set on the compiler command line or in the
  board's hwdef (e.g. `define NN_USE_CMSIS_DSP 1`).

  NN_USE_CMSIS_DSP     1: dense layers use CMSIS-DSP arm_mat_vec_mult_f32 (Cortex-M SIMD/FPU-tuned kernels).
                       0 (default): portable C++ with four accumulators, which GCC turns into dual-issued
                       VFMA on a Cortex-M7 anyway.
  NN_USE_STM32_CORDIC  1: the CORDIC co-processor of STM32H723/725/730/733/735 is available as a tanh backend
                       (NN_Activation::Cordic). Not every STM32H7 has it: H743/H750/H753 do not.
  NN_CORDIC_DEVICE_HEADER  header that defines the CORDIC and RCC registers (default "hal.h" from ChibiOS).
  NN_WEIGHTS_SECTION   attribute placing the weight arrays, e.g. __attribute__((section(".dtcm"))) to read
                       them from DTCM instead of flash. Empty by default (flash, read through the M7 cache).
*/
#pragma once

#ifndef NN_USE_CMSIS_DSP
#define NN_USE_CMSIS_DSP 0
#endif

#ifndef NN_USE_STM32_CORDIC
#define NN_USE_STM32_CORDIC 0
#endif

#ifndef NN_WEIGHTS_SECTION
#define NN_WEIGHTS_SECTION
#endif

// 32-byte alignment matches the Cortex-M7 cache line, so each weight row starts on a fresh line.
#define NN_WEIGHTS_ALIGNMENT __attribute__((aligned(32)))

#define NN_MAX_INPUTS 96
#define NN_MAX_HIDDEN 64
#define NN_MAX_MOTORS 8
#define NN_LAYER_COUNT 3
#define NN_HISTORY_FRAMES 2
