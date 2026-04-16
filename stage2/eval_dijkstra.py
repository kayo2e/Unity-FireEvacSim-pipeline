"""
Static Dijkstra 평가
=====================
train.py의 _compute_bfs_with_risk() 로직을 그대로 사용.
RL 대신 고정 fire_cost_weight 값으로 운영.

동일 시드로 실행하여 RL/A*와 공정 비교.

실행: python eval_dijkstra.py [--episodes 50] [--people 10] [--scenarios 1 2 3 4]
"""

import sys, json, time
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from train import FireEvacEnv, SCENARIO_CONFIGS, DummyVecEnv

STRATEGY_NAME = "Static Dijkstra"
FIRE_COST = 20.0  # 고정값


def eval_dijkstra(n_agents=10, scenarios=(1,2,3,4), n_episodes=50):
    all_results = {}

    for sc in scenarios:
        cfg = SCENARIO_CONFIGS[sc]
        print(f"\n[{STRATEGY_NAME} fc={FIRE_COST}] 시나리오 {sc} ({cfg['name']}) | {n_agents}명 | {n_episodes}회")
        t0 = time.time()
        records = []

        for ep in range(n_episodes):
            seed = ep * 100 + sc  # ★ 통일 시드
            env = FireEvacEnv(scenario=sc, n_agents=n_agents)
            vec_env = DummyVecEnv([lambda: env])
            obs = vec_env.reset()
            env.reset(seed=seed)
            obs = np.array([env._get_obs()])

            total_r = 0.0
            for _ in range(cfg["max_steps"]):
                action = np.array([[FIRE_COST]])
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

    results = eval_dijkstra(n_agents=args.people, scenarios=args.scenarios, n_episodes=args.episodes)

    out = {"strategy": STRATEGY_NAME, "n_agents": args.people,
           "n_episodes": args.episodes, "fire_cost": FIRE_COST, "results": results}
    out_path = f"result_dijkstra_{args.people}ppl.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
