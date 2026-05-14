"""
exp_speed.py — PPO / RecurrentPPO / A* 추론 속도 비교
======================================================
각 모델의 step당 action 결정 시간 측정.

실행:
    cd stage2
    python experiments/exp_speed.py
    python experiments/exp_speed.py --steps 500 --scenarios 1 4
"""

import sys, os, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines'))

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _find_model(model_dir, n_agents):
    single = os.path.join(model_dir, "fire_evac_model.zip")
    if os.path.exists(single):
        vecnorm = single.replace(".zip", "_vecnorm.pkl")
        return single, vecnorm
    candidates = []
    for f in (os.listdir(model_dir) if os.path.isdir(model_dir) else []):
        if f.endswith(".zip") and "ppl" in f and "best" not in f:
            try:
                n = int(f.replace("fire_evac_model_", "").replace("ppl.zip", ""))
                candidates.append(n)
            except ValueError:
                pass
    if not candidates:
        return None, None
    best = min(candidates, key=lambda x: abs(x - n_agents))
    path = os.path.join(model_dir, f"fire_evac_model_{best}ppl.zip")
    return path, path.replace(".zip", "_vecnorm.pkl")


def bench_astar(scenario, n_agents, n_steps):
    from astar_real import astar_action

    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    env.reset()
    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        _ = astar_action(env)
        times.append(time.perf_counter() - t0)
        obs, _, term, trunc, _ = env.step(np.array([10.0, 10.0, 2.0]))
        if term or trunc:
            env.reset()
    env.close()
    return np.array(times) * 1000  # ms


def bench_ppo(model_dir, model_cls_name, scenario, n_agents, n_steps):
    if model_cls_name == "recurrent":
        from sb3_contrib import RecurrentPPO as ModelCls
    else:
        from stable_baselines3 import PPO as ModelCls

    mpath, vpath = _find_model(model_dir, n_agents)
    if mpath is None or not os.path.exists(mpath):
        return None, f"모델 없음: {model_dir}"

    model = ModelCls.load(mpath)
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    vec = DummyVecEnv([lambda: env])
    if vpath and os.path.exists(vpath):
        vec = VecNormalize.load(vpath, vec)
        vec.training = False
        vec.norm_reward = False

    obs = vec.reset()
    lstm_states = None
    ep_starts = np.ones((1,), dtype=bool)
    times = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        if model_cls_name == "recurrent":
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
            ep_starts = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=True)
        times.append(time.perf_counter() - t0)

        obs, _, done, _ = vec.step(action)
        if done[0]:
            obs = vec.reset()
            lstm_states = None
            ep_starts = np.ones((1,), dtype=bool)

    vec.close()
    return np.array(times) * 1000, None


def _fmt(t):
    return f"{t:.3f}ms"


def print_result(label, times):
    if times is None:
        print(f"  {label:<22} 스킵")
        return
    print(f"  {label:<22} mean={_fmt(times.mean())}  "
          f"p50={_fmt(np.percentile(times,50))}  "
          f"p95={_fmt(np.percentile(times,95))}  "
          f"p99={_fmt(np.percentile(times,99))}  "
          f"max={_fmt(times.max())}")


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",     type=int, default=300)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--no-ppo",    action="store_true")
    parser.add_argument("--no-rppo",   action="store_true")
    parser.add_argument("--no-astar",  action="store_true")
    args = parser.parse_args()

    ppo_dir  = os.path.join(BASE, "model", "ppo")
    rppo_dir = os.path.join(BASE, "model", "recurrent_ppo")

    print(f"\n{'═'*78}")
    print(f"  추론 속도 비교  ({args.steps} steps per scenario)")
    print(f"{'─'*78}")
    print(f"  {'모델':<22} {'mean':>10}  {'p50':>10}  {'p95':>10}  {'p99':>10}  {'max':>10}")
    print(f"{'─'*78}")

    for sc in args.scenarios:
        n = SCENARIO_CONFIGS[sc]["n_agents"]
        print(f"\n  [S{sc} {SCENARIO_CONFIGS[sc]['name']} — {n}명]")

        if not args.no_astar:
            print("  A* 측정 중...", end="\r")
            t = bench_astar(sc, n, args.steps)
            print_result("A* (astar_real)", t)

        if not args.no_ppo:
            print("  PPO 측정 중...", end="\r")
            t, err = bench_ppo(ppo_dir, "ppo", sc, n, args.steps)
            if err:
                print(f"  {'PPO':<22} {err}")
            else:
                print_result("PPO", t)

        if not args.no_rppo:
            print("  RecurrentPPO 측정 중...", end="\r")
            t, err = bench_ppo(rppo_dir, "recurrent", sc, n, args.steps)
            if err:
                print(f"  {'RecurrentPPO':<22} {err}")
            else:
                print_result("RecurrentPPO", t)

    print(f"\n{'═'*78}")
    print("  * A*: 매 스텝 전체 인원에 대해 BFS+경로 계산")
    print("  * PPO/RecurrentPPO: 신경망 forward pass 1회")
    print(f"{'═'*78}\n")