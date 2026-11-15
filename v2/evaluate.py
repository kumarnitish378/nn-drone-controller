import sys, json
import numpy as np
from sim import MultirotorEnv, euler_from_quat, wrap, G
from ppo import PPO

weights = sys.argv[1] if len(sys.argv) > 1 else "quad.npz"
frame = sys.argv[2] if len(sys.argv) > 2 else "quad_x"
ERR_CLIP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

# x, y, z (metres, relative to start), heading (deg)
MISSION = [(0, 0, 3, 0), (6, 0, 3, 0), (6, 6, 3, 90), (0, 6, 5, 180), (0, 0, 5, -90), (0, 0, 2, 0)]
ACCEPT = 0.4      # m, waypoint reached radius
HOLD_LAST = 8.0   # s to hold the final waypoint


def fly(n, randomize, noise, seed, max_t=70.0, record=False):
    env = MultirotorEnv(n, frame=frame, randomize=randomize, noise=noise, seed=seed, err_clip=ERR_CLIP)
    agent = PPO(env.obs_dim, env.act_dim); agent.load(weights)
    env.reset_idx(np.arange(n), hard=False)
    wp = np.zeros(n, dtype=int)
    done_t = np.full(n, np.nan); crashed = np.zeros(n, bool)
    hold_err = [[] for _ in range(n)]
    rec = dict(t=[], p=[], rpy=[], pwm=[], tgt=[], volt=[])
    max_tilt = np.zeros(n)
    steps = int(max_t / env.cfg["ctrl_dt"])
    for k in range(steps):
        m = np.array(MISSION)[wp]
        env.target = m[:, :3].astype(float); env.target_yaw = np.deg2rad(m[:, 3])
        a, _ = agent.act(env.observe(), deterministic=True)
        _, _, c, _, _, info = env.step(a, auto_target=False)
        crashed |= c | (env.p[:, 2] < -3.0)
        t = (k + 1) * env.cfg["ctrl_dt"]
        max_tilt = np.maximum(max_tilt, np.rad2deg(np.arccos(np.clip(1 - 2 * (env.q[:, 1] ** 2 + env.q[:, 2] ** 2), -1, 1))))
        reach = (info["dist"] < ACCEPT) & (wp < len(MISSION) - 1)
        wp[reach] += 1
        last = (wp == len(MISSION) - 1)
        newly = last & (info["dist"] < ACCEPT) & np.isnan(done_t)
        done_t[newly] = t
        for i in np.nonzero(last & ~np.isnan(done_t))[0]:
            hold_err[i].append(info["dist"][i])
        if record:
            r, p, y = euler_from_quat(env.q)
            rec["t"].append(t); rec["p"].append(env.p[0].copy())
            rec["rpy"].append([r[0], p[0], y[0]]); rec["pwm"].append(info["pwm"][0].copy())
            rec["tgt"].append(env.target[0].copy()); rec["volt"].append(env.volt[0])
        if np.all(~np.isnan(done_t) & (t - done_t > HOLD_LAST) | crashed):
            break
    ok = ~np.isnan(done_t) & ~crashed
    hold = np.array([np.mean(h[-200:]) if len(h) > 200 else np.nan for h in hold_err])
    return dict(max_tilt=max_tilt, ok=ok, crashed=crashed, done_t=done_t, hold=hold,
                rec={k: np.array(v) for k, v in rec.items()})


if __name__ == "__main__":
    nom = fly(1, randomize=False, noise=True, seed=1, record=True)
    rob = fly(500, randomize=True, noise=True, seed=2)
    out = dict(
        nominal_completed=bool(nom["ok"][0]),
        nominal_mission_time_s=float(nom["done_t"][0]),
        nominal_final_hold_error_m=float(nom["hold"][0]),
        robust_n=500,
        robust_completed=int(rob["ok"].sum()),
        robust_crashed=int(rob["crashed"].sum()),
        robust_timeout=int((~rob["ok"] & ~rob["crashed"]).sum()),
        robust_mission_time_median_s=float(np.nanmedian(rob["done_t"])),
        robust_hold_error_median_m=float(np.nanmedian(rob["hold"])),
        robust_hold_error_p95_m=float(np.nanpercentile(rob["hold"], 95)),
        robust_hold_error_max_m=float(np.nanmax(rob["hold"])),
        robust_max_tilt_p95_deg=float(np.percentile(rob["max_tilt"], 95)),
    )
    rec = nom["rec"]
    pw = rec["pwm"]
    out["pwm_min"] = float(pw.min()); out["pwm_max"] = float(pw.max())
    out["pwm_hover_mean"] = float(pw[-200:].mean())
    out["pwm_jitter_hover_us"] = float(np.abs(np.diff(pw[-200:], axis=0)).mean())
    out["max_tilt_deg"] = float(np.rad2deg(np.abs(rec["rpy"][:, :2]).max()))
    print(json.dumps(out, indent=1))
    json.dump(out, open("results.json", "w"), indent=1)
    np.savez("flight_log.npz", **rec, mission=np.array(MISSION))
