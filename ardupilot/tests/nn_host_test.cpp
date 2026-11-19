/*
  Host check of the C++ flight library against the Python simulator (driven by tests/test_cpp_library.py).

    nn_host_test replay <recording.bin> <exact|fast|cordic>
        Feeds a recorded Python flight through NN_Pilot step by step and prints, as JSON, the largest
        difference from Python in the network inputs and motor commands, and the host time per update.
    nn_host_test tanh
        Largest error of the fast and CORDIC tanh against tanhf over -12..12.

  Recording: little-endian float64 values. Header: motor_count, steps, input_count, nominal_mass_kg,
  position_error_clip, initial_motor_command. Then per step: body_to_ned (9, row-major), gyro (3), accel (3),
  position (3), velocity (3), battery voltage, target position (3), target yaw, Python commands (motor_count),
  applied commands (motor_count), Python inputs (input_count).
*/
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "NN_Pilot.h"
#if NN_USE_STM32_CORDIC
#include "NN_Stm32Cordic.h"
#endif
#include "models/NN_Model_hex_x.h"
#include "models/NN_Model_quad_x.h"

namespace {

const int HEADER_VALUES = 6;
const int ATTITUDE_VALUES = 9;
const int STATE_VALUES = ATTITUDE_VALUES + 3 + 3 + 3 + 3 + 1;   // + gyro, accel, position, velocity, battery
const int TARGET_VALUES = 4;
const int MOTOR_COUNT_QUAD = 4;
const float TANH_SCAN_LIMIT = 12.0f;
const int TANH_SCAN_POINTS = 200001;

bool read_recording(const char *path, std::vector<double> &values)
{
    FILE *file = std::fopen(path, "rb");
    if (file == nullptr) {
        return false;
    }
    double value;
    while (std::fread(&value, sizeof(value), 1, file) == 1) {
        values.push_back(value);
    }
    std::fclose(file);
    return values.size() >= size_t(HEADER_VALUES);
}

NN_Vector3 vector_at(const double *values) { return NN_Vector3{float(values[0]), float(values[1]), float(values[2])}; }

bool parse_activation(const char *name, NN_Activation &activation)
{
    if (std::strcmp(name, "exact") == 0) {
        activation = NN_Activation::Exact;
    } else if (std::strcmp(name, "fast") == 0) {
        activation = NN_Activation::FastApproximation;
    } else if (std::strcmp(name, "cordic") == 0) {
        activation = NN_Activation::Cordic;
    } else {
        return false;
    }
    return true;
}

int replay(const char *path, const char *activation_name)
{
    std::vector<double> values;
    NN_Activation activation;
    if (!read_recording(path, values) || !parse_activation(activation_name, activation)) {
        std::fprintf(stderr, "cannot read %s or unknown activation %s\n", path, activation_name);
        return 2;
    }
    const int motor_count = int(values[0]), steps = int(values[1]), input_count = int(values[2]);
    NN_PilotConfig config;
    config.nominal_mass_kg = float(values[3]);
    config.position_error_clip = float(values[4]);
    const NN_Model &model = motor_count == MOTOR_COUNT_QUAD ? nn_model_quad_x::model : nn_model_hex_x::model;
    NN_Pilot pilot(model, config);
    if (!pilot.valid() || pilot.input_count() != input_count || !pilot.network().set_activation(activation)) {
        std::fprintf(stderr, "model / recording mismatch or activation not built in\n");
        return 2;
    }
    pilot.reset(float(values[5]));

    const int step_values = STATE_VALUES + TARGET_VALUES + 2 * motor_count + input_count;
    if (int(values.size()) != HEADER_VALUES + steps * step_values) {
        std::fprintf(stderr, "recording has %d values, expected %d\n", int(values.size()), HEADER_VALUES + steps * step_values);
        return 2;
    }
    double largest_input_error = 0.0, largest_command_error = 0.0, update_seconds = 0.0;
    int worst_input = -1;
    float commands[NN_MAX_MOTORS], applied[NN_MAX_MOTORS];
    for (int step = 0; step < steps; step++) {
        const double *record = values.data() + HEADER_VALUES + step * step_values;
        NN_Sensors sensors;
        for (int row = 0; row < 3; row++) {
            for (int column = 0; column < 3; column++) {
                sensors.body_to_ned[row][column] = float(record[row * 3 + column]);
            }
        }
        const double *state = record + ATTITUDE_VALUES;
        sensors.gyro = vector_at(state);
        sensors.accel = vector_at(state + 3);
        sensors.position = vector_at(state + 6);
        sensors.velocity = vector_at(state + 9);
        sensors.battery_voltage = float(state[12]);
        const double *target_values = record + STATE_VALUES;
        NN_Target target{vector_at(target_values), float(target_values[3])};
        const double *python_commands = target_values + TARGET_VALUES;
        const double *python_applied = python_commands + motor_count;
        const double *python_inputs = python_applied + motor_count;

        const auto start = std::chrono::steady_clock::now();
        pilot.update(sensors, target, commands);
        update_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        for (int input = 0; input < input_count; input++) {
            const double error = std::fabs(pilot.last_inputs()[input] - python_inputs[input]);
            if (error > largest_input_error) {
                largest_input_error = error;
                worst_input = input;
            }
        }
        for (int motor = 0; motor < motor_count; motor++) {
            largest_command_error = std::fmax(largest_command_error, std::fabs(commands[motor] - python_commands[motor]));
            applied[motor] = float(python_applied[motor]);   // the motors got what Python gave them (spool-up included)
        }
        pilot.commit(applied);
    }
    std::printf("{\"steps\": %d, \"history_spacing\": %d, \"largest_input_error\": %.3g, \"worst_input\": %d, "
                "\"largest_command_error\": %.3g, \"host_update_us\": %.3f}\n",
                steps, pilot.history_spacing(), largest_input_error, worst_input, largest_command_error,
                1e6 * update_seconds / steps);
    return 0;
}

int tanh_errors()
{
    double fast_error = 0.0, cordic_error = 0.0;
#if NN_USE_STM32_CORDIC
    nn_cordic_init();
#endif
    for (int point = 0; point < TANH_SCAN_POINTS; point++) {
        const float value = -TANH_SCAN_LIMIT + 2.0f * TANH_SCAN_LIMIT * float(point) / float(TANH_SCAN_POINTS - 1);
        const double exact = std::tanh(double(value));
        fast_error = std::fmax(fast_error, std::fabs(NN_Network::fast_tanh(value) - exact));
#if NN_USE_STM32_CORDIC
        float cordic = value;
        nn_cordic_tanh(&cordic, 1);
        cordic_error = std::fmax(cordic_error, std::fabs(cordic - exact));
#endif
    }
    std::printf("{\"fast_tanh_error\": %.3g, \"cordic_tanh_error\": %.3g}\n", fast_error, cordic_error);
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc == 4 && std::strcmp(argv[1], "replay") == 0) {
        return replay(argv[2], argv[3]);
    }
    if (argc == 2 && std::strcmp(argv[1], "tanh") == 0) {
        return tanh_errors();
    }
    std::fprintf(stderr, "usage: nn_host_test replay <recording.bin> <exact|fast|cordic> | nn_host_test tanh\n");
    return 2;
}
