"""
RL (PPO) 평가
===============
train.py로 학습된 PPO 모델을 동일 시드/조건으로 평가.
Dijkstra/A*와 공정 비교를 위해 동일 시드 사용.

실행: python eval_rl.py [--episodes 50] [--people 10] [--scenarios 1 2 3 4]
"""

import sys, os, json, time
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from train import FireEvacEnv, SCENARIO_CONFIGS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

STRATEGY_NAME = "RL (PPO)"


def eval_rl(n_agents=10, scenarios=(1,2,3,4), n_episodes=50):
    model_path = f"fire_evac_model_{n_agents}ppl"
    vecnorm_path = f"{model_path}_vecnorm.pkl"

    if not os.path.exists(model_path + ".zip"):
        print(f"[!] 모델 없음: {model_path}.zip")
        print(f"    먼저 python train.py --mode train --people {n_agents} 실행 필요")
        return None

    print(f"[RL] 모델 로드: {model_path}.zip")
    model = PPO.load(model_path)

    all_results = {}

    for sc in scenarios:
        cfg = SCENARIO_CONFIGS[sc]
        print(f"\n[{STRATEGY_NAME}] 시나리오 {sc} ({cfg['name']}) | {n_agents}명 | {n_episodes}회")
        t0 = time.time()
        records = []

        for ep in range(n_episodes):
            seed = ep * 100 + sc  # ★ 통일 시드

            env = FireEvacEnv(scenario=sc, n_agents=n_agents)
            vec_env = DummyVecEnv([lambda: env])
            if os.path.exists(vecnorm_path):
                vec_env = VecNormalize.load(vecnorm_path, vec_env)
                vec_env.training = False
                vec_env.norm_reward = False

            # 시드 적용: DummyVecEnv reset 후 내부 env를 동일 시드로 재설정
            obs = vec_env.reset()
            env.reset(seed=seed)
            raw_obs = np.array([env._get_obs()])
            if isinstance(vec_env, VecNormalize):
                obs = vec_env.normalize_obs(raw_obs)
            else:
                obs = raw_obs

            total_r = 0.0
            for _ in range(cfg["max_steps"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, infos = vec_env.step(action)
                total_r += float(r[0])
                if done[0]:
                    break

            info = infos[0]
            records.append({
                "episode": ep, "seed": seed,
                "survival_rate": info["survival_rate"],
                "escaped": info["escaped"],
                "dead": info["dead"],
                "steps": info["step"],
                "reward": total_r,
            })
            vec_env.close()

        elapsed = time.time() - t0
        avg_sr = np.mean([r["survival_rate"] for r in records])
        avg_rew = np.mean([r["reward"] for r in records])
        print(f"  생존율 {avg_sr:.1%} | 보상 {avg_rew:+.1f} | {elapsed:.1f}s")
        all_results[str(sc)] = records

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=f"{STRATEGY_NAME} 평가")
    parser.add_argument("--episodes",  type=int, default=50)
    parser.add_argument("--people",    type=int, default=10)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    results = eval_rl(n_agents=args.people, scenarios=args.scenarios, n_episodes=args.episodes)

    if results is not None:
        out = {"strategy": STRATEGY_NAME, "n_agents": args.people,
               "n_episodes": args.episodes,
               "model_path": f"fire_evac_model_{args.people}ppl",
               "results": results}
        out_path = f"result_rl_{args.people}ppl.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {out_path}")
