/* Network lab core (browser + node): builds the flight network's inputs from physical values exactly as
   nn_drone.simulation.observation does (sensor noise off), runs the network keeping every layer's
   activations, and finds the steady state in which the network's previous outputs equal its outputs.
   Axes: body x forward, y left, z up. */
(function (root) {
  "use strict";
  const GRAVITY = 9.81;
  const GYRO_SCALE = 8.0, ACCELEROMETER_SCALE = 2 * GRAVITY, POSITION_ERROR_SCALE = 5.0, VELOCITY_SCALE = 5.0;
  const BATTERY_CENTER = 15.6, BATTERY_SPAN = 1.2, INTEGRAL_SCALE = 1.5, MASS_CLASS_REFERENCE_DECADE = 1.0;
  const MIN_ERROR_NORM = 1e-6, HISTORY_FRAMES = 2;
  const PWM_CENTER = 1500, PWM_HALF_RANGE = 500, PWM_MIN = 1000, PWM_SPAN = 1000;
  const STEADY_ITERATIONS = 600, STEADY_BLEND = 0.5, STEADY_TOLERANCE = 1e-5;

  const AXES = ["fwd", "left", "up"];
  const GYRO_AXES = ["roll", "pitch", "yaw"];

  function inputLayout(motorCount) {
    const motors = Array.from({ length: motorCount }, (_, index) => `motor ${index + 1}`);
    const frame = (prefix) => [
      ...GYRO_AXES.map((axis) => ({ group: `${prefix}gyro`, name: `${prefix}gyro ${axis}` })),
      ...AXES.map((axis) => ({ group: `${prefix}accel`, name: `${prefix}accel ${axis}` })),
      { group: `${prefix}attitude`, name: `${prefix}roll` }, { group: `${prefix}attitude`, name: `${prefix}pitch` }];
    const history = (index) => {
      const prefix = `h${index} `;
      return [...frame(prefix),
              ...AXES.map((axis) => ({ group: `${prefix}target`, name: `${prefix}target ${axis}` })),
              ...AXES.map((axis) => ({ group: `${prefix}velocity`, name: `${prefix}velocity ${axis}` })),
              ...motors.map((motor) => ({ group: `${prefix}previous`, name: `${prefix}prev ${motor}` }))];
    };
    return [...frame(""),
            { group: "heading", name: "heading err sin" }, { group: "heading", name: "heading err cos" },
            ...AXES.map((axis) => ({ group: "target", name: `target ${axis}` })),
            ...AXES.map((axis) => ({ group: "velocity", name: `velocity ${axis}` })),
            { group: "battery", name: "battery" },
            ...motors.map((motor) => ({ group: "previous", name: `prev ${motor}` })),
            ...AXES.map((axis) => ({ group: "integral", name: `integral ${axis}` })),
            ...history(1), ...history(2),
            { group: "mass", name: "mass class" }];
  }

  const degrees = (value) => value * Math.PI / 180;
  const wrap = (angle) => { const turn = 2 * Math.PI; return ((((angle + Math.PI) % turn) + turn) % turn) - Math.PI; };

  /* state: massKg, batteryV, errorClip, rollDeg, pitchDeg, headingErrorDeg, gyroDegS[3], accel[3],
     target[3] (body, m), velocity[3] (body, m/s), integral[3] (body, size-normalised m·s),
     previous[M] (-1..1). Returns {current, history}: the current features and the history slice. */
  function currentFeatures(state) {
    const length = Math.cbrt(state.massKg), timeScale = Math.sqrt(length);
    const gyro = state.gyroDegS.map((rate) => degrees(rate) * timeScale);
    let target = state.target.map((value) => value / length);
    const velocity = state.velocity.map((value) => value / timeScale);
    const norm = Math.hypot(...target), shrink = Math.min(1, state.errorClip / Math.max(norm, MIN_ERROR_NORM));
    target = target.map((value) => value * shrink);
    const headingError = wrap(degrees(state.headingErrorDeg));
    const features = [...gyro.map((value) => value / GYRO_SCALE), ...state.accel.map((value) => value / ACCELEROMETER_SCALE),
                      degrees(state.rollDeg) / Math.PI, degrees(state.pitchDeg) / Math.PI,
                      Math.sin(headingError), Math.cos(headingError),
                      ...target.map((value) => value / POSITION_ERROR_SCALE), ...velocity.map((value) => value / VELOCITY_SCALE),
                      (state.batteryV - BATTERY_CENTER) / BATTERY_SPAN, ...state.previous,
                      ...state.integral.map((value) => value / INTEGRAL_SCALE)];
    return features;
  }

  function historySlice(features, motorCount) {
    return [...features.slice(0, 8), ...features.slice(10, 16), ...features.slice(17, 17 + motorCount)];
  }

  /* history: null = steady (both earlier frames equal the current one), else [frame1, frame2]. */
  function buildInputs(state, history) {
    const motorCount = state.previous.length;
    const features = currentFeatures(state);
    const now = historySlice(features, motorCount);
    const frames = history || Array.from({ length: HISTORY_FRAMES }, () => now);
    return { inputs: [...features, ...frames[0], ...frames[1], Math.log10(state.massKg) - MASS_CLASS_REFERENCE_DECADE], historyNow: now };
  }

  function prepare(network) {
    const layers = [[network.W0, network.b0], [network.W1, network.b1], [network.W2, network.b2]].map(([weights, bias]) => ({
      weights, bias, inputs: weights.length, outputs: bias.length }));
    return Object.assign({}, network, { layers });
  }

  /* Returns the activations of every layer: [inputs, hidden 1, hidden 2, outputs]. */
  function forward(prepared, inputs) {
    const activations = [inputs];
    let current = inputs;
    for (const layer of prepared.layers) {
      const next = new Array(layer.outputs);
      for (let output = 0; output < layer.outputs; output++) {
        let sum = layer.bias[output];
        for (let input = 0; input < layer.inputs; input++) sum += current[input] * layer.weights[input][output];
        next[output] = Math.tanh(sum);
      }
      activations.push(next);
      current = next;
    }
    return activations;
  }

  /* Previous motor outputs "steady": the network keeps seeing its own output until it stops changing.
     options.start: previous outputs to start that search from (a nearby solution makes it fast). */
  function evaluate(prepared, state, options) {
    let previous = (options.steadyPrevious && options.start ? options.start : state.previous).slice(), activations = null, built = null, converged = !options.steadyPrevious;
    const iterations = options.steadyPrevious ? STEADY_ITERATIONS : 1;
    for (let iteration = 0; iteration < iterations; iteration++) {
      built = buildInputs(Object.assign({}, state, { previous }), options.history);
      activations = forward(prepared, built.inputs);
      if (!options.steadyPrevious) break;
      const outputs = activations[activations.length - 1];
      const change = Math.max(...outputs.map((value, index) => Math.abs(value - previous[index])));
      previous = previous.map((value, index) => (1 - STEADY_BLEND) * value + STEADY_BLEND * outputs[index]);
      if (change < STEADY_TOLERANCE) { converged = true; break; }
    }
    const outputs = activations[activations.length - 1];
    return { activations, inputs: built.inputs, historyNow: built.historyNow, previous, converged,
             pwm: outputs.map(actionToPwm) };
  }

  const actionToPwm = (action) => PWM_CENTER + PWM_HALF_RANGE * Math.max(-1, Math.min(1, action));
  const pwmToAction = (pwm) => (pwm - PWM_CENTER) / PWM_HALF_RANGE;
  const pwmToThrottle = (pwm) => (pwm - PWM_MIN) / PWM_SPAN;

  /* What the motor commands do together: thrust over weight, and roll / pitch / yaw torque as a share of
     what full differential thrust could give. Signs: roll + = right side down, pitch + = nose down,
     yaw + = nose turning left (seen from above). */
  function motorEffect(pwm, layout, thrustToWeight, batteryV, nominalBatteryV) {
    const throttleSquared = pwm.map((value) => pwmToThrottle(value) ** 2);
    const count = pwm.length, voltageFactor = (batteryV / nominalBatteryV) ** 2;
    const forward = layout.angles.map((angle) => Math.cos(degrees(angle)));
    const left = layout.angles.map((angle) => -Math.sin(degrees(angle)));
    const sum = (values) => values.reduce((total, value) => total + value, 0);
    const roll = sum(throttleSquared.map((value, index) => value * left[index])) / sum(left.map(Math.abs)) * 2;
    const pitch = -sum(throttleSquared.map((value, index) => value * forward[index])) / sum(forward.map(Math.abs)) * 2;
    const yaw = -sum(throttleSquared.map((value, index) => value * layout.spin[index])) / count * 2;
    return { thrustToWeight: thrustToWeight * voltageFactor * sum(throttleSquared) / count, roll, pitch, yaw };
  }

  const api = { inputLayout, buildInputs, prepare, forward, evaluate, actionToPwm, pwmToAction, motorEffect, GRAVITY };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NetworkLab = api;
})(typeof window !== "undefined" ? window : globalThis);
