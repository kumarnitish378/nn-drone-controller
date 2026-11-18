/* Neural-network drone simulator core (browser + node). A line-for-line port of nn_drone.simulation (single
   drone) and of the flight logic in nn_drone.ground_link.vehicle. Axes: x north, y west, z up; body x forward,
   y left, z up. Network timing comes with the weights: {period, scalesWithSize, substeps}. */
(function (root) {
  "use strict";
  const G = 9.81, I_MAX = 1.5, I_CLAMP = 4.0, I_ZONE = 1.0, I_SPEED = 0.5, REFERENCE_PERIOD = 0.01, HISTORY_FRAMES = 2;
  const FROUDE_SCALED_100_HZ = { period: REFERENCE_PERIOD, scalesWithSize: true, substeps: 2 };
  const PWM_OFF = 1000, PWM_SPOOL = 1450, T_SPOOL = 2.0, T_CONFIRM = 1.0, T_RAMP = 1.0, T_IDLE_END = 2.0;
  const FRAMES = {
    quad_x: { angles: [45, -135, -45, 135], spin: [1, 1, -1, -1] },
    hex_x: { angles: [90, -90, -30, 150, 30, -150], spin: [-1, 1, -1, 1, 1, -1] },
  };

  // ---------- small vector / quaternion helpers ----------
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  function qRot(q, v) {            // body -> world
    const w = q[0], qv = [q[1], q[2], q[3]];
    const t = cross(qv, v).map((x) => 2 * x), c = cross(qv, t);
    return [v[0] + w * t[0] + c[0], v[1] + w * t[1] + c[1], v[2] + w * t[2] + c[2]];
  }
  const qRotInv = (q, v) => qRot([q[0], -q[1], -q[2], -q[3]], v);
  function euler(q) {
    const [w, x, y, z] = q;
    return [Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x)))),
            Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))];
  }
  const wrap = (a) => { const t = 2 * Math.PI; return ((((a + Math.PI) % t) + t) % t) - Math.PI; };
  function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }

  // ---------- network ----------
  class Policy {
    constructor(w) {               // w: {W0:[in][h], b0, W1, b1, W2, b2} as nested arrays
      this.L = [[w.W0, w.b0], [w.W1, w.b1], [w.W2, w.b2]].map(([W, b]) => {
        const nin = W.length, nout = b.length, flat = new Float64Array(nin * nout);
        for (let i = 0; i < nin; i++) for (let j = 0; j < nout; j++) flat[i * nout + j] = W[i][j];
        return { W: flat, b: Float64Array.from(b), nin, nout };
      });
    }
    forward(x) {
      let h = x;
      for (const { W, b, nin, nout } of this.L) {
        const o = new Float64Array(nout);
        for (let j = 0; j < nout; j++) o[j] = b[j];
        const n = Math.min(nin, h.length);       // missing trailing inputs have zero weight
        for (let i = 0; i < n; i++) { const hi = h[i]; if (hi !== 0) { const r = i * nout; for (let j = 0; j < nout; j++) o[j] += hi * W[r + j]; } }
        for (let j = 0; j < nout; j++) o[j] = Math.tanh(o[j]);
        h = o;
      }
      return h;
    }
  }

  // ---------- physics (sim.py) ----------
  class Sim {
    constructor(c) {
      this.c = Object.assign({ frame: "quad_x", mass: 1, tw: 2.8, v0: 16, noise: true, errClip: 2.5, alt0: 0, wind: [0, 0, 0],
                               timing: FROUDE_SCALED_100_HZ }, c);
      const f = FRAMES[this.c.frame]; this.M = f.angles.length;
      this.mx = f.angles.map((a) => 0.17 * Math.cos(a * Math.PI / 180));
      this.my = f.angles.map((a) => -0.17 * Math.sin(a * Math.PI / 180));
      this.spin = f.spin;
      const mn = this.c.mass; this.mn = mn; this.Ls = Math.cbrt(mn); this.rL = Math.sqrt(this.Ls);
      this.mass = mn; const s53 = Math.pow(mn, 5 / 3);
      this.inertia = [0.010 * s53, 0.010 * s53, 0.018 * s53];
      const timing = this.c.timing; this.substeps = timing.substeps;
      this.dt = timing.scalesWithSize ? timing.period * this.rL : timing.period;
      this.stepLength = timing.scalesWithSize ? 1 : timing.period / (REFERENCE_PERIOD * this.rL);
      this.histSpacing = Math.max(1, Math.round(1 / this.stepLength));
      this.tmax = this.c.tw * mn * G / this.M; this.tau = 0.05 * this.rL;
      this.kyaw = 0.016 * this.Ls; this.drag = 0.25 * Math.pow(mn, 2 / 3); this.rdamp = 0.002 * s53;
      this.histDim = 14 + this.M; this.obsDim = 21 + this.M + 2 * this.histDim;
      this.p = [0, 0, 0]; this.v = [0, 0, 0]; this.q = [1, 0, 0, 0]; this.w = [0, 0, 0];
      this.mot = new Array(this.M).fill(0); this.prevA = new Array(this.M).fill(-1);
      this.ierr = [0, 0, 0]; this.volt = this.c.v0; this.accB = [0, 0, G];
      this.onGround = true; this.impact = 0; this.past = []; this.lastBase = null;
      this.prevTarget = [0, 0, 0]; this.target = [0, 0, 0]; this.targetYaw = 0; this.wind = this.c.wind.slice();
      this.pwm = new Array(this.M).fill(1000);
    }
    base() {
      const c = this.c, L = this.Ls, rL = this.rL, nz = c.noise ? 1 : 0;
      let [roll, pitch, yaw] = euler(this.q);
      let eb = qRotInv(this.q, [this.target[0] - this.p[0], this.target[1] - this.p[1], this.target[2] - this.p[2]]);
      let vb = qRotInv(this.q, this.v);
      let gy = this.w.slice(), ac = this.accB.slice(), herr = wrap(this.targetYaw - yaw), volt = this.volt;
      if (nz) {
        gy = gy.map((x) => x + 0.02 * gauss()); ac = ac.map((x) => x + 0.3 * gauss());
        roll += 0.005 * gauss(); pitch += 0.005 * gauss(); herr += 0.01 * gauss();
        eb = eb.map((x) => x + 0.03 * gauss()); vb = vb.map((x) => x + 0.05 * gauss()); volt += 0.03 * gauss();
      }
      gy = gy.map((x) => x * rL); eb = eb.map((x) => x / L); vb = vb.map((x) => x / rL);
      const d = norm(eb), k = Math.min(1, c.errClip / Math.max(d, 1e-6)); eb = eb.map((x) => x * k);
      const ib = qRotInv(this.q, this.ierr);
      return [gy[0] / 8, gy[1] / 8, gy[2] / 8, ac[0] / (2 * G), ac[1] / (2 * G), ac[2] / (2 * G),
              roll / Math.PI, pitch / Math.PI, Math.sin(herr), Math.cos(herr),
              eb[0] / 5, eb[1] / 5, eb[2] / 5, vb[0] / 5, vb[1] / 5, vb[2] / 5, (volt - 15.6) / 1.2,
              ...this.prevA, ib[0] / I_MAX, ib[1] / I_MAX, ib[2] / I_MAX];
    }
    core(b) { return [...b.slice(0, 8), ...b.slice(10, 16), ...b.slice(17, 17 + this.M)]; }
    observe() {
      const b = this.base();
      const capacity = HISTORY_FRAMES * this.histSpacing, n = this.past.length;
      if (!n) { const now = this.core(b); for (let i = 0; i < capacity; i++) this.past.push(now); }
      this.lastBase = b;
      const len = this.past.length, h1 = this.past[len - this.histSpacing], h2 = this.past[len - 2 * this.histSpacing];
      return Float64Array.from([...b, ...h1, ...h2, Math.log10(this.mn) - 1]);
    }
    physics(u, dt) {
      const M = this.M, k = Math.min(1, dt / this.tau);
      for (let i = 0; i < M; i++) this.mot[i] += (u[i] - this.mot[i]) * k;
      const vs = Math.pow(this.volt / 16, 2);
      const rho = Math.pow(Math.max(0.3, Math.min(1.1, 1 - 2.2558e-5 * (this.c.alt0 + this.p[2]))), 4.2559);
      let Tt = 0, tx = 0, ty = 0, tz = 0, m2 = 0;
      for (let i = 0; i < M; i++) {
        const T = this.tmax * vs * rho * this.mot[i] * this.mot[i];
        Tt += T; tx += T * this.my[i]; ty -= T * this.mx[i]; tz -= T * this.spin[i]; m2 += this.mot[i] * this.mot[i];
      }
      this.volt = this.c.v0 - 0.9 * m2 / M;
      const w = this.w, I = this.inertia;
      const tq = [tx * this.Ls - this.rdamp * w[0], ty * this.Ls - this.rdamp * w[1], tz * this.kyaw - this.rdamp * w[2]];
      const g = cross(w, [I[0] * w[0], I[1] * w[1], I[2] * w[2]]);
      for (let i = 0; i < 3; i++) w[i] += (tq[i] - g[i]) / I[i] * dt;
      const q = this.q;
      const dq = [0.5 * (-q[1] * w[0] - q[2] * w[1] - q[3] * w[2]), 0.5 * (q[0] * w[0] + q[2] * w[2] - q[3] * w[1]),
                  0.5 * (q[0] * w[1] - q[1] * w[2] + q[3] * w[0]), 0.5 * (q[0] * w[2] + q[1] * w[1] - q[2] * w[0])];
      let n = 0; for (let i = 0; i < 4; i++) { q[i] += dq[i] * dt; n += q[i] * q[i]; }
      n = Math.sqrt(n); for (let i = 0; i < 4; i++) q[i] /= n;
      const th = qRot(q, [0, 0, Tt]), a = [0, 0, 0];
      for (let i = 0; i < 3; i++) a[i] = (th[i] - this.drag * rho * (this.v[i] - this.wind[i])) / this.mass;
      this.accB = qRotInv(q, a);
      a[2] -= G;
      for (let i = 0; i < 3; i++) { this.v[i] += a[i] * dt; this.p[i] += this.v[i] * dt; }
      const gnd = this.p[2] <= 0 && this.v[2] <= 0;
      if (gnd) {
        if (!this.onGround) this.impact = Math.max(this.impact, -this.v[2]);
        const yaw = euler(q)[2]; this.q = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
        this.p[2] = 0; this.v = [0, 0, 0]; this.w = [0, 0, 0]; this.accB = [0, 0, G];
      }
      this.onGround = gnd;
    }
    step(action) {
      const M = this.M, a = Array.from(action, (x) => Math.max(-1, Math.min(1, x)));
      if (this.lastBase) { this.past.push(this.core(this.lastBase)); if (this.past.length > HISTORY_FRAMES * this.histSpacing) this.past.shift(); }
      const u = a.map((x) => (1500 + 500 * x - 1000) / 1000);
      this.pwm = a.map((x) => 1500 + 500 * x);
      for (let s = 0; s < this.substeps; s++) this.physics(u, this.dt / this.substeps);
      const L = this.Ls, rL = this.rL;
      const err = [this.target[0] - this.p[0], this.target[1] - this.p[1], this.target[2] - this.p[2]];
      const dist = norm(err) / L;
      let tv = [0, 1, 2].map((i) => (this.target[i] - this.prevTarget[i]) / this.dt);
      if (norm(tv) > 5 * rL) tv = [0, 0, 0];
      this.prevTarget = this.target.slice();
      if (dist < I_ZONE && norm(this.v) / rL < I_SPEED && norm(tv) / rL < 0.05)
        for (let i = 0; i < 3; i++) this.ierr[i] = Math.max(-I_CLAMP, Math.min(I_CLAMP, this.ierr[i] + err[i] / L * REFERENCE_PERIOD * this.stepLength));
      this.prevA = a;
    }
  }

  // ---------- flight logic (mavlink_sim.py Vehicle) ----------
  function defaultParams(frame, mass) {
    const rl = Math.pow(mass, 1 / 6);
    return { SIM_MASS: mass, SIM_TW_RATIO: 2.8, SIM_WIND_SPD: 0, SIM_WIND_DIR: 0, SIM_BATT_VOLT: 16, SIM_NOISE: 1, NN_ERR_CLIP: 2.5,
             WPNAV_SPEED: Math.round(250 * rl), WPNAV_SPEED_UP: Math.round(250 * rl), WPNAV_SPEED_DN: Math.round(150 * rl),
             WPNAV_RADIUS: Math.round(100 * Math.cbrt(mass)), LAND_SPEED: 50, LAND_ALT_LOW: 1000, RTL_ALT: 1500,
             WP_YAW_BEHAVIOR: 1, MOT_SPIN_ARM: 0.15, HOME_ALT: 0 };
  }

  class Vehicle {
    /* mission: [{cmd:'takeoff'|'waypoint'|'circle'|'rtl'|'land', n, e, alt, hold, radius, turns}], n/e/alt in metres from home */
    constructor(frame, weights, params, log) {
      this.frame = frame; this.weights = weights; this.P = Object.assign(defaultParams(frame, params.SIM_MASS || 1), params);
      this.log = log || (() => {}); this.mission = []; this.t = 0; this.rebuild();
    }
    rebuild() {
      const P = this.P, d = P.SIM_WIND_DIR * Math.PI / 180, s = P.SIM_WIND_SPD;
      this.sim = new Sim({ frame: this.frame, mass: Math.max(0.2, Math.min(500, P.SIM_MASS)), tw: P.SIM_TW_RATIO, v0: P.SIM_BATT_VOLT,
                           noise: P.SIM_NOISE >= 0.5, errClip: P.NN_ERR_CLIP, alt0: P.HOME_ALT,
                           wind: [-s * Math.cos(d), s * Math.sin(d), 0], timing: this.weights.timing || FROUDE_SCALED_100_HZ });
      this.policy = new Policy(this.weights);
      this.dt = this.sim.dt; this.L = this.sim.Ls;
      this.armed = false; this.phase = "disarmed"; this.tPhase = this.t; this.carrot = [0, 0, 0]; this.yawT = 0;
      this.goals = []; this.seq = -1; this.running = false; this.confirm = 0; this.crashed = false;
      this.pwm = new Array(this.sim.M).fill(PWM_OFF); this.ramp0 = this.pwm.slice(); this.reached = [];
      this.status = "Disarmed"; this.done = false; this.touchdown = null; if (this.idleTime === undefined) this.idleTime = 2.0;
    }
    get pos() { return this.sim.p.slice(); }
    say(s) { this.status = s; this.log(this.t, s); }
    rtlGoals() {
      const p = this.pos, alt = Math.max(p[2], this.P.RTL_ALT / 100), low = this.P.LAND_ALT_LOW / 100;
      return [{ kind: "goto", pos: [p[0], p[1], alt] }, { kind: "goto", pos: [0, 0, alt] }, { kind: "goto", pos: [0, 0, Math.min(alt, low)] },
              { kind: "land", pos: [0, 0, 0] }];
    }
    start(mission) {                     // arm and fly the mission
      if (this.armed || this.crashed) return false;
      this.mission = mission.slice(); if (!this.mission.length) return false;
      this.armed = true; this.phase = "idle"; this.tPhase = this.t; this.running = true; this.seq = -1; this.goals = [];
      this.reached = []; this.done = false; this.touchdown = null; this.sim.impact = 0;
      this.say("Armed, idling");
      return true;
    }
    command(c) {                         // 'rtl' | 'land' | 'hold' while flying
      if (this.phase !== "fly") return false;
      this.running = false;
      if (c === "rtl") { this.goals = this.rtlGoals(); this.say("Return to launch"); }
      else if (c === "land") { this.goals = [{ kind: "land", pos: this.pos }]; this.carrot = this.pos; this.say("Landing here"); }
      else { this.goals = [{ kind: "hold", pos: this.pos }]; this.carrot = this.pos; this.say("Holding position"); }
      return true;
    }
    nextItem() {
      for (;;) {
        this.seq += 1;
        if (this.seq >= this.mission.length) { this.running = false; this.goals = [{ kind: "hold", pos: null }]; this.say("Mission complete, holding"); return; }
        const it = this.mission[this.seq];
        const here = this.goals.length && this.goals[this.goals.length - 1].pos ? this.goals[this.goals.length - 1].pos : this.carrot;
        const x = it.n, y = -it.e, alt = it.alt;
        if (it.cmd === "takeoff") { this.goals = [{ kind: "goto", pos: [this.pos[0], this.pos[1], alt], hold: 0 }]; this.say("Take-off to " + alt + " m"); }
        else if (it.cmd === "waypoint") { this.goals = [{ kind: "goto", pos: [x, y, alt], hold: it.hold || 0 }]; this.say("Flying to item " + (this.seq + 1)); }
        else if (it.cmd === "circle") {
          const r = it.radius || 10, cen = [x, y, alt]; let d = [here[0] - x, here[1] - y]; const n = Math.hypot(d[0], d[1]);
          d = n > 1e-3 ? [d[0] / n, d[1] / n] : [1, 0];
          const edge = [x + Math.abs(r) * d[0], y + Math.abs(r) * d[1], alt];
          this.goals = [{ kind: "goto", pos: edge, hold: 0 }, { kind: "circle", pos: edge, center: cen, radius: r, turns: Math.max(it.turns || 1, 1) }];
          this.say("Flying to circle, item " + (this.seq + 1));
        }
        else if (it.cmd === "rtl") { this.goals = this.rtlGoals(); this.say("Return to launch"); }
        else if (it.cmd === "land") {
          const low = this.P.LAND_ALT_LOW / 100, hx = it.n === null || it.n === undefined ? here[0] : x, hy = it.e === null || it.e === undefined ? here[1] : y;
          this.goals = [{ kind: "goto", pos: [hx, hy, here[2]], hold: 0 }, { kind: "goto", pos: [hx, hy, Math.min(here[2], low)], hold: 0 }, { kind: "land", pos: [hx, hy, 0] }];
          this.say("Landing");
        } else continue;
        return;
      }
    }
    goalDone() {
      this.goals.shift();
      if (!this.goals.length) { if (this.running) this.nextItem(); else this.goals = [{ kind: "hold", pos: this.carrot.slice() }]; }
    }
    advance() {
      const P = this.P, dt = this.dt; if (!this.goals.length) return;
      const g = this.goals[0]; if (!g.pos) g.pos = this.carrot.slice();
      const vh = P.WPNAV_SPEED / 100, vu = P.WPNAV_SPEED_UP / 100, vd = P.WPNAV_SPEED_DN / 100;
      if (g.kind === "goto" || g.kind === "hold") {
        const d = [g.pos[0] - this.carrot[0], g.pos[1] - this.carrot[1], g.pos[2] - this.carrot[2]], dh = Math.hypot(d[0], d[1]);
        const T = Math.max(dh / Math.max(vh, 0.1), d[2] > 0 ? d[2] / Math.max(vu, 0.1) : -d[2] / Math.max(vd, 0.1));
        if (T > dt) {
          for (let i = 0; i < 3; i++) this.carrot[i] += d[i] * (dt / T);
          if (P.WP_YAW_BEHAVIOR >= 0.5 && dh > 3 * this.L) this.yawT = Math.atan2(d[1], d[0]);
        } else {
          this.carrot = g.pos.slice();
          const p = this.pos, near = Math.hypot(p[0] - g.pos[0], p[1] - g.pos[1], p[2] - g.pos[2]) < P.WPNAV_RADIUS / 100;
          if (g.kind === "goto" && near) {
            if (g.tArr === undefined) { g.tArr = this.t; if (this.running && this.goals.length === 1) this.reached.push(this.seq); }
            if (this.t - g.tArr >= (g.hold || 0)) this.goalDone();
          }
        }
      } else if (g.kind === "circle") {
        if (g.a0 === undefined) { g.a0 = Math.atan2(this.carrot[1] - g.center[1], this.carrot[0] - g.center[0]); g.ang = 0; }
        const sgn = g.radius > 0 ? -1 : 1; g.ang += vh * dt / Math.abs(g.radius);
        const a = g.a0 + sgn * g.ang, R = Math.abs(g.radius);
        this.carrot = [g.center[0] + R * Math.cos(a), g.center[1] + R * Math.sin(a), g.center[2]];
        if (P.WP_YAW_BEHAVIOR >= 0.5) this.yawT = a + sgn * Math.PI / 2;
        if (g.ang >= 2 * Math.PI * g.turns) { if (this.running) this.reached.push(this.seq); this.goalDone(); }
      } else if (g.kind === "land") {
        this.carrot[0] = g.pos[0]; this.carrot[1] = g.pos[1];
        const rate = this.pos[2] < P.LAND_ALT_LOW / 100 ? P.LAND_SPEED / 100 : vd;
        if (this.carrot[2] > -3 * this.L) this.carrot[2] -= rate * dt;
      }
    }
    step() {
      const s = this.sim, P = this.P, idle = 1000 + 1000 * Math.max(0, Math.min(0.3, P.MOT_SPIN_ARM));
      const landing = this.goals.length && this.goals[0].kind === "land";
      if (this.phase === "fly") this.advance(); else this.carrot = this.pos;
      s.target = this.carrot.slice(); s.targetYaw = this.yawT;
      let a = Array.from(this.policy.forward(s.observe()));
      let pwm = null; const ph = this.phase, M = s.M;
      if (ph === "disarmed") pwm = new Array(M).fill(PWM_OFF);
      else if (ph === "idle") {
        pwm = new Array(M).fill(idle);
        if (this.t - this.tPhase >= this.idleTime) { this.phase = "spool"; this.tPhase = this.t; this.say("Spooling up"); }
      } else if (ph === "spool") {
        const f = Math.min(1, (this.t - this.tPhase) / T_SPOOL); pwm = new Array(M).fill(idle + f * (PWM_SPOOL - idle));
        if (f >= 1) { this.phase = "fly"; this.carrot = this.pos; this.yawT = euler(s.q)[2]; this.nextItem(); }
      } else if (ph === "ramp") {
        const f = Math.min(1, (this.t - this.tPhase) / T_RAMP); pwm = this.ramp0.map((x) => (1 - f) * x + f * idle);
        if (f >= 1) { this.phase = "idle_end"; this.tPhase = this.t; }
      } else if (ph === "idle_end") {
        pwm = new Array(M).fill(idle);
        if (this.t - this.tPhase >= T_IDLE_END) { this.armed = false; this.phase = "disarmed"; this.goals = []; this.done = true; this.say("Landed and disarmed"); }
      }
      if (pwm) a = pwm.map((x) => (x - 1500) / 500);
      const wasGround = s.onGround;
      s.step(a); this.pwm = s.pwm.slice(); this.t += this.dt;
      if (this.phase === "fly") {
        const up = 1 - 2 * (s.q[1] * s.q[1] + s.q[2] * s.q[2]);
        if (landing && s.onGround && !wasGround && !this.touchdown) this.touchdown = { speed: s.impact, offset: Math.hypot(s.p[0] - this.goals[0].pos[0], s.p[1] - this.goals[0].pos[1]), t: this.t };
        if (up < 0 || s.impact > 3) { this.crashed = true; this.armed = false; this.phase = "disarmed"; this.goals = []; this.running = false; this.done = true; this.say("CRASH: motors stopped"); }
        else if (landing) {
          const mean = this.pwm.reduce((x, y) => x + y, 0) / M;
          const cond = Math.abs(s.v[2]) < 0.15 && this.carrot[2] < s.p[2] - 0.5 && mean < PWM_SPOOL;
          this.confirm = cond ? this.confirm + this.dt : 0;
          if (this.confirm >= T_CONFIRM) { this.phase = "ramp"; this.tPhase = this.t; this.ramp0 = this.pwm.slice(); this.confirm = 0; this.running = false; this.say("Landing detected"); }
        }
      }
    }
    telemetry() {
      const s = this.sim, e = euler(s.q);
      return { n: s.p[0], e: -s.p[1], alt: s.p[2], vn: s.v[0], ve: -s.v[1], vup: s.v[2], roll: e[0], pitch: -e[1], heading: ((-e[2] * 180 / Math.PI) % 360 + 360) % 360,
               volt: s.volt, pwm: this.pwm.slice(), phase: this.phase, armed: this.armed, t: this.t, q: s.q.slice(),
               carrot: [this.carrot[0], -this.carrot[1], this.carrot[2]], seq: this.seq, speed: Math.hypot(s.v[0], s.v[1], s.v[2]) };
    }
  }

  const api = { Sim, Policy, Vehicle, defaultParams, FRAMES, euler };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.NNDrone = api;
})(typeof window !== "undefined" ? window : globalThis);
