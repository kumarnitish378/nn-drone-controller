import argparse, time, json, os
import numpy as np
from sim import MultirotorEnv
from ppo import PPO

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="quad_x")
ap.add_argument("--envs", type=int, default=1024)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--minutes", type=float, default=8.0)
ap.add_argument("--max_iters", type=int, default=100000)
ap.add_argument("--out", default="policy")
ap.add_argument("--resume", default="")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--critic_warmup", type=int, default=0)
args = ap.parse_args()

env = MultirotorEnv(args.envs, frame=args.frame, seed=args.seed)
agent = PPO(env.obs_dim, env.act_dim, seed=args.seed, lr=args.lr)
if args.resume:
    agent.load(args.resume)
N, T = args.envs, args.steps
obs = env.observe()
log = []
logpath = args.out + "_log.json"
if args.resume and os.path.exists(logpath):
    log = json.load(open(logpath))
samples0 = log[-1]["samples"] if log else 0
t0 = time.time()
for it in range(args.max_iters):
    agent.opt_pi.lr = 0.0 if it < args.critic_warmup else args.lr   # let the critic learn a new reward first
    O = np.zeros((T, N, env.obs_dim), np.float32); A = np.zeros((T, N, env.act_dim), np.float32)
    LP = np.zeros((T, N), np.float32); R = np.zeros((T, N), np.float32)
    V = np.zeros((T + 1, N), np.float32); D = np.zeros((T, N), bool)
    crashes = 0; dist_sum = 0.0
    for t in range(T):
        a, lp = agent.act(obs)
        V[t] = agent.value(obs)
        O[t], A[t], LP[t] = obs, a, lp
        obs, r, crashed, timeout, final_obs, info = env.step(a)
        if final_obs is not None:      # time-limit: bootstrap from the pre-reset state
            r = r + agent.gamma * agent.value(final_obs) * timeout
        R[t] = r; D[t] = crashed | timeout
        crashes += int(crashed.sum()); dist_sum += float(info["dist"].mean())
    V[T] = agent.value(obs)
    adv = np.zeros((T, N), np.float32); last = 0
    for t in reversed(range(T)):
        nd = 1.0 - D[t]
        delta = R[t] + agent.gamma * V[t + 1] * nd - V[t]
        last = delta + agent.gamma * agent.lam * nd * last
        adv[t] = last
    ret = adv + V[:-1]
    agent.update(O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1),
                 adv.reshape(-1), ret.reshape(-1))
    el = time.time() - t0
    rec = dict(samples=samples0 + (it + 1) * T * N, reward=float(R.mean() * 10),
               crash_per_1k=crashes / (T * N) * 1000, dist=dist_sum / T,
               std=float(np.exp(agent.logstd).mean()))
    log.append(rec)
    if it % 10 == 0:
        print(f"it {it:4d} {el:5.0f}s samples {rec['samples']/1e6:6.2f}M rew {rec['reward']:6.3f} "
              f"crash/1k {rec['crash_per_1k']:6.3f} dist {rec['dist']:5.2f} std {rec['std']:.3f}", flush=True)
    if it % 20 == 0 or el > args.minutes * 60:
        agent.save(args.out + ".npz"); json.dump(log, open(logpath, "w"))
    if el > args.minutes * 60:
        break
print("done")
