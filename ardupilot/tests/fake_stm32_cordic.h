/*
  Host stand-in for the STM32H7 CORDIC and RCC registers, so NN_Stm32Cordic.cpp can be compiled and checked
  on a PC. Writing WDATA computes cosh and sinh (function 5, scale 1) in double precision and queues them as
  q1.31, the way the peripheral returns them; reading RDATA pops the queue.
  This checks the library's argument scaling, range reduction and q1.31 handling, not the silicon.
*/
#pragma once

#include <cmath>
#include <cstdint>

namespace fake_stm32 {

const double Q31_ONE = 2147483648.0;
const double HYPERBOLIC_SCALE = 2.0;   // scale 1: the argument and both results are divided by 2
const double HYPERBOLIC_LIMIT = 1.118;

inline uint32_t to_q31(double value)
{
    const double clipped = value >= 1.0 ? (Q31_ONE - 1.0) / Q31_ONE : (value < -1.0 ? -1.0 : value);
    return uint32_t(int32_t(std::lround(clipped * Q31_ONE)));
}

struct CordicPeripheral {
    struct WriteRegister {
        CordicPeripheral *owner;
        WriteRegister &operator=(uint32_t value)
        {
            const double argument = double(int32_t(value)) / Q31_ONE * HYPERBOLIC_SCALE;
            owner->argument_in_range = owner->argument_in_range && std::fabs(argument) <= HYPERBOLIC_LIMIT;
            owner->results[0] = to_q31(std::cosh(argument) / HYPERBOLIC_SCALE);
            owner->results[1] = to_q31(std::sinh(argument) / HYPERBOLIC_SCALE);
            owner->next_result = 0;
            return *this;
        }
    };
    struct ReadRegister {
        CordicPeripheral *owner;
        operator uint32_t() const { return owner->results[owner->next_result++ & 1]; }
    };

    uint32_t CSR = 0;
    WriteRegister WDATA{this};
    ReadRegister RDATA{this};
    uint32_t results[2] = {0, 0};
    unsigned next_result = 0;
    bool argument_in_range = true;
};

struct ResetAndClockControl {
    uint32_t AHB2ENR = 0;
};

inline CordicPeripheral &cordic()
{
    static CordicPeripheral peripheral;
    return peripheral;
}

inline ResetAndClockControl &rcc()
{
    static ResetAndClockControl registers;
    return registers;
}

}  // namespace fake_stm32

#define CORDIC (&fake_stm32::cordic())
#define RCC (&fake_stm32::rcc())
#define RCC_AHB2ENR_CORDICEN (1u << 4)
