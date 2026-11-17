import argparse, time, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                help="cuda = NVIDIA GPU through CuPy, auto = GPU if available else CPU")
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
ap.add_argument("--cfg", default="{}")
ap.add_argument("--min_logstd", type=float, default=-3.0)
ap.add_argument("--lam_t", type=float, default=0.0)
ap.add_argument("--lam_s", type=float, default=0.0)
args = ap.parse_args()

os.environ["NN_DEVICE"] = args.device          # must be set before sim/ppo are imported (see backend.py)
import backend
from backend import xp as np
from sim import MultirotorEnv
from ppo import PPO
print("device:", backend.device_name(), flush=True)

env = MultirotorEnv(args.envs, frame=args.frame, seed=args.seed, **json.loads(args.cfg))
agent = PPO(env.obs_dim, env.act_dim, seed=args.seed, lr=args.lr)
if args.resume:
    agent.load(args.resume)
agent.min_logstd = args.min_logstd
agent.logstd[...] = np.maximum(agent.logstd, args.min_logstd)
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
    crashes = np.zeros((), np.int64); dist_sum = np.zeros(())   # stay on the device: no sync per step
    for t in range(T):
        a, lp = agent.act(obs)
        V[t] = agent.value(obs)
        O[t], A[t], LP[t] = obs, a, lp
        obs, r, crashed, timeout, final_obs, info = env.step(a)
        if final_obs is not None:      # time-limit: bootstrap from the pre-reset state
            r = r + agent.gamma * agent.value(final_obs) * timeout
        R[t] = r; D[t] = crashed | timeout
        crashes += crashed.sum(); dist_sum += info["dist"].mean()
    V[T] = agent.value(obs)
    adv = np.zeros((T, N), np.float32); last = 0
    for t in reversed(range(T)):
        nd = 1.0 - D[t]
        delta = R[t] + agent.gamma * V[t + 1] * nd - V[t]
        last = delta + agent.gamma * agent.lam * nd * last
        adv[t] = last
    ret = adv + V[:-1]
    On = np.concatenate([O[1:], obs[None]], 0)
    agent.update(O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1),
                 adv.reshape(-1), ret.reshape(-1), obs_next=On.reshape(T * N, -1),
                 cont=(~D).reshape(-1).astype(np.float32), lam_t=args.lam_t, lam_s=args.lam_s)
    el = time.time() - t0
    rec = dict(samples=samples0 + (it + 1) * T * N, reward=float(R.mean() * 10),
               crash_per_1k=float(crashes) / (T * N) * 1000, dist=float(dist_sum) / T,
               std=float(np.exp(agent.logstd).mean()))
    log.append(rec)
    if it % 10 == 0:
        print(f"it {it:4d} {el:5.0f}s samples {rec['samples']/1e6:6.2f}M rew {rec['reward']:6.3f} "
              f"crash/1k {rec['crash_per_1k']:6.3f} dist {rec['dist']:5.2f} std {rec['std']:.3f}", flush=True)
    if it % 20 == 0 or el > args.minutes * 60:
        agent.save(args.out + ".npz"); json.dump(log, open(logpath, "w"))
    if it % 30 == 0 and it >= args.critic_warmup:
        agent.save(f"{args.out}_snap{it:03d}.npz")
    if el > args.minutes * 60:
        break
print("done")
