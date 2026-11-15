"""Minimal PPO in pure NumPy (manual backprop) - no deep-learning framework needed."""
import numpy as np


class MLP:
    """tanh hidden layers; optional tanh on the output."""

    def __init__(self, sizes, out_tanh, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            s = np.sqrt(1.0 / sizes[i])
            if i == len(sizes) - 2:
                s *= out_scale
            self.W.append((rng.standard_normal((sizes[i], sizes[i + 1])) * s).astype(np.float32))
            self.b.append(np.zeros(sizes[i + 1], dtype=np.float32))
        self.out_tanh = out_tanh

    def params(self):
        return self.W + self.b

    def forward(self, x, cache=False):
        acts = [x]
        n = len(self.W)
        for i in range(n):
            x = x @ self.W[i] + self.b[i]
            if i < n - 1 or self.out_tanh:
                x = np.tanh(x)
            acts.append(x)
        if cache:
            self._acts = acts
        return x

    def backward(self, dout):
        """dout = dLoss/doutput. Returns grads in the same order as params()."""
        acts = self._acts
        n = len(self.W)
        gW, gb = [None] * n, [None] * n
        d = dout
        for i in reversed(range(n)):
            if i < n - 1 or self.out_tanh:
                d = d * (1 - acts[i + 1] ** 2)
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = d @ self.W[i].T
        return gW + gb


class Adam:
    def __init__(self, params, lr):
        self.p = params; self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, max_norm=0.5):
        gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        sc = min(1.0, max_norm / (gn + 1e-8))
        self.t += 1
        b1, b2 = 0.9, 0.999
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * sc
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            mh = m / (1 - b1 ** self.t); vh = v / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), seed=0, lr=3e-4,
                 gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.0, init_logstd=-0.7):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.pi = MLP([obs_dim, *hidden, act_dim], out_tanh=True, rng=rng, out_scale=0.1)
        self.vf = MLP([obs_dim, 128, 128, 1], out_tanh=False, rng=rng)
        self.logstd = np.full(act_dim, init_logstd, dtype=np.float32)
        self.opt_pi = Adam(self.pi.params() + [self.logstd], lr)
        self.opt_vf = Adam(self.vf.params(), lr)
        self.gamma, self.lam, self.clip, self.ent_coef = gamma, lam, clip, ent_coef

    def act(self, obs, deterministic=False):
        mu = self.pi.forward(obs)
        if deterministic:
            return mu, None
        a = mu + np.exp(self.logstd) * self.rng.standard_normal(mu.shape).astype(np.float32)
        return a, self._logp(a, mu)

    def _logp(self, a, mu):
        z = (a - mu) / np.exp(self.logstd)
        return (-0.5 * z * z - self.logstd - 0.9189385).sum(1)

    def value(self, obs):
        return self.vf.forward(obs)[:, 0]

    def update(self, obs, act, logp_old, adv, ret, epochs=5, batch=8192):
        n = len(obs)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, batch):
                i = perm[s:s + batch]
                o, a, lo, ad, rt = obs[i], act[i], logp_old[i], adv[i], ret[i]
                m = len(i)
                # ---- policy
                mu = self.pi.forward(o, cache=True)
                std = np.exp(self.logstd)
                z = (a - mu) / std
                logp = (-0.5 * z * z - self.logstd - 0.9189385).sum(1)
                ratio = np.exp(np.clip(logp - lo, -20, 20))
                unclipped = (ratio > 1 - self.clip) & (ratio < 1 + self.clip)
                s1 = ratio * ad; s2 = np.clip(ratio, 1 - self.clip, 1 + self.clip) * ad
                active = (s1 <= s2) | unclipped
                dlogp = (-(ad * ratio) * active / m)[:, None]
                dmu = dlogp * z / std
                dlogstd = (dlogp * (z * z - 1)).sum(0) - self.ent_coef
                g = self.pi.backward(dmu.astype(np.float32))
                self.opt_pi.step(g + [dlogstd.astype(np.float32)])
                np.clip(self.logstd, -3.0, 0.0, out=self.logstd)
                # ---- value
                v = self.vf.forward(o, cache=True)[:, 0]
                dv = ((v - rt) / m)[:, None].astype(np.float32)
                self.opt_vf.step(self.vf.backward(dv))

    def save(self, path):
        d = {f"pi_W{i}": w for i, w in enumerate(self.pi.W)}
        d.update({f"pi_b{i}": b for i, b in enumerate(self.pi.b)})
        d.update({f"vf_W{i}": w for i, w in enumerate(self.vf.W)})
        d.update({f"vf_b{i}": b for i, b in enumerate(self.vf.b)})
        d["logstd"] = self.logstd
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for net, k in ((self.pi, "pi"), (self.vf, "vf")):
            for i in range(len(net.W)):
                w = d[f"{k}_W{i}"]
                if w.shape != net.W[i].shape:      # warm start: new inputs begin at zero weight
                    net.W[i][...] = 0; net.W[i][:w.shape[0]] = w
                else:
                    net.W[i][...] = w
                net.b[i][...] = d[f"{k}_b{i}"]
        self.logstd[...] = d["logstd"]
