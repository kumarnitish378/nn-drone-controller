import sys, json
import numpy as np
from sim import MultirotorEnv, euler_from_quat, wrap, G
from ppo import PPO

weights = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".npz") else "quad.npz"
import os
frame = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NN_FRAME", "quad_x")
ERR_CLIP = float(sys.argv[3]) if len(sys.argv) > 3 else 2.5

# x, y, z (metres, relative to start), heading (deg)
MISSION = [(0, 0, 3, 0), (6, 0, 3, 0), (6, 6, 3, 90), (0, 6, 5, 180), (0, 0, 5, -90), (0, 0, 2, 0)]
if len(sys.argv) > 4 and sys.argv[4] == "high":     # high-altitude mission: 20 m and 30 m legs, then descend
    MISSION = [(0, 0, 20, 0), (8, 0, 20, 0), (8, 8, 20, 90), (0, 8, 30, 180), (0, 0, 30, -90), (0, 0, 5, 0)]
ACCEPT = 0.4      # m, waypoint reached radius
HOLD_LAST = 8.0   # s to hold the final waypoint


def fly(n, randomize, noise, seed, max_t=70.0, record=False, mission=None, hold=None, **envkw):
    MISSION = mission if mission is not None else globals()['MISSION']
    HOLD_LAST = hold if hold is not None else globals()['HOLD_LAST']
    envkw.setdefault('err_clip', ERR_CLIP)
    env = MultirotorEnv(n, frame=frame, randomize=randomize, noise=noise, seed=seed, **envkw)
    agent = PPO(env.obs_dim, env.act_dim); agent.load(weights)
    env.reset_idx(np.arange(n), hard=False)
    wp = np.zeros(n, dtype=int)
    done_t = np.full(n, np.nan); crashed = np.zeros(n, bool)
    hold_sum = np.zeros(n); hold_cnt = np.zeros(n)
    rec = dict(t=[], p=[], rpy=[], pwm=[], tgt=[], volt=[])
    rec_every = max(1, int(max_t / 70))
    max_tilt = np.zeros(n)
    dt = float(env.dt_env[0])           # control period (longer for bigger drones)
    steps = int(max_t / dt)
    for k in range(steps):
        m = np.array(MISSION)[wp]
        env.target = m[:, :3].astype(float); env.target_yaw = np.deg2rad(m[:, 3])
        a, _ = agent.act(env.observe(), deterministic=True)
        _, _, c, _, _, info = env.step(a, auto_target=False)
        # crash = flipped over, runaway speed, or fell below start (distance to target is not a crash here)
        crashed |= ((1 - 2 * (env.q[:, 1] ** 2 + env.q[:, 2] ** 2)) < 0) | (np.linalg.norm(env.v, axis=1) > 20) | (env.p[:, 2] < -3.0)
        t = (k + 1) * dt
        max_tilt = np.maximum(max_tilt, np.rad2deg(np.arccos(np.clip(1 - 2 * (env.q[:, 1] ** 2 + env.q[:, 2] ** 2), -1, 1))))
        last = (wp == len(MISSION) - 1)                 # already heading for the final waypoint this step
        newly = last & (info["dist"] < ACCEPT) & np.isnan(done_t)
        reach = (info["dist"] < ACCEPT) & (wp < len(MISSION) - 1)
        wp[reach] += 1
        done_t[newly] = t
        # hold error = mean distance from the final waypoint, measured HOLD_LAST-2 .. HOLD_LAST s after arrival
        w = last & ~np.isnan(done_t) & (t - done_t > HOLD_LAST - 2.0) & (t - done_t <= HOLD_LAST)
        hold_sum[w] += info["dist"][w]; hold_cnt[w] += 1
        if record and k % rec_every == 0:
            r, p, y = euler_from_quat(env.q)
            rec["t"].append(t); rec["p"].append(env.p[0].copy())
            rec["rpy"].append([r[0], p[0], y[0]]); rec["pwm"].append(info["pwm"][0].copy())
            rec["tgt"].append(env.target[0].copy()); rec["volt"].append(env.volt[0])
        if np.all(~np.isnan(done_t) & (t - done_t > HOLD_LAST) | crashed):
            break
    ok = ~np.isnan(done_t) & ~crashed
    hold = np.where(hold_cnt > 50, hold_sum / np.maximum(hold_cnt, 1), np.nan)
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
    p, tg, tt = rec["p"], rec["tgt"], rec["t"]
    vz = np.gradient(p[:, 2], tt)
    out["max_climb_rate_mps"] = float(vz.max()); out["max_descent_rate_mps"] = float(-vz.min())
    out["max_altitude_overshoot_m"] = float(max(0.0, (p[:, 2] - np.maximum.accumulate(tg[:, 2])).max()))
    lvl = (np.abs(np.gradient(tg[:, 2])) == 0) & (np.abs(tg[:, 2] - p[:, 2]) < 1.0) & (np.abs(vz) < 0.5)
    out["level_flight_altitude_error_mean_m"] = float(np.abs(tg[lvl, 2] - p[lvl, 2]).mean())
    print(json.dumps({k: out[k] for k in list(out)[-4:]}, indent=1))
    json.dump(out, open("results.json", "w"), indent=1)
    np.savez("flight_log.npz", **rec, mission=np.array(MISSION))
