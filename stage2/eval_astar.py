"""
A* 평가
========
FireEvacEnv를 상속하여 _compute_dirs_for_strategy를 A*로 교체.
맨해튼 거리 휴리스틱 + 화재/연기 비용.

Dijkstra와의 차이: 휴리스틱으로 출구 방향을 우선 탐색하여
탐색 효율이 높음. 하지만 파라미터는 여전히 고정.

동일 시드로 실행하여 Dijkstra/RL과 공정 비교.

실행: python eval_astar.py [--episodes 50] [--people 10] [--scenarios 1 2 3 4]
"""

import sys, json, time
import heapq
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from train import (
    FireEvacEnv, SCENARIO_CONFIGS, DummyVecEnv,
    EXIT_POSITIONS, WALKABLE, EXIT,
    N, S, E, W, DELTA,
)

STRATEGY_NAME = "A* (Manhattan)"
FIRE_COST = 20.0
SMOKE_COST = 5.0
HEURISTIC_WEIGHT = 0.5


class AStarEvacEnv(FireEvacEnv):
    """_compute_dirs_for_strategy를 A*로 완전 교체."""

    def _compute_dirs_for_strategy(self, fire_cost):
        ROWS, COLS = self.ROWS, self.COLS
        g_cost = np.full((ROWS, COLS), 9999.0)
        parent_dir = np.full((ROWS, COLS), -1, dtype=np.int32)

        open_exits = [
            (r, c) for r, c in EXIT_POSITIONS
            if self.grid[r, c] == EXIT
        ]
        if not open_exits:
            return np.zeros(self.n_lights, dtype=np.int32)

        # 출구에서 역방향 확장
        q = []
        for (r, c) in open_exits:
            g_cost[r, c] = 0
            h = 0  # 출구 자체는 휴리스틱 0
            heapq.heappush(q, (0.0, r, c))

        opposite = {N: S, S: N, E: W, W: E}

        while q:
            f, r, c = heapq.heappop(q)
            if f > g_cost[r, c] + 50:
                continue

            for d, (dr, dc) in DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                if self.grid[nr, nc] not in WALKABLE:
                    continue

                # 이동 비용 (Dijkstra와 동일한 비용 체계)
                if self.fire_map[nr, nc] > 0:
                    step_cost = fire_cost
                elif self.smoke_map[nr, nc] > 0:
                    step_cost = SMOKE_COST
                else:
                    step_cost = 1.0

                # 밀도 비용 (train.py와 동일)
                density = sum(1 for p in self.people_data if p["pos"] == (nr, nc))
                step_cost += density * 2.0

                new_g = g_cost[r, c] + step_cost
                if new_g < g_cost[nr, nc]:
                    g_cost[nr, nc] = new_g
                    parent_dir[nr, nc] = opposite[d]
                    # ★ A* 핵심: 맨해튼 휴리스틱 추가
                    h = min(abs(nr - er) + abs(nc - ec) for er, ec in open_exits)
                    heapq.heappush(q, (new_g + h * HEURISTIC_WEIGHT, nr, nc))

        dirs = np.zeros(self.n_lights, dtype=np.int32)
        for i, (r, c) in enumerate(self.light_cells):
            if parent_dir[r, c] >= 0:
                dirs[i] = parent_dir[r, c]
            else:
                dirs[i] = self._bfs_best(r, c)
        return dirs


def eval_astar(n_agents=10, scenarios=(1,2,3,4), n_episodes=50):
    all_results = {}

    for sc in scenarios:
        cfg = SCENARIO_CONFIGS[sc]
        print(f"\n[{STRATEGY_NAME}] 시나리오 {sc} ({cfg['name']}) | {n_agents}명 | {n_episodes}회")
        t0 = time.time()
        records = []

        for ep in range(n_episodes):
            seed = ep * 100 + sc  # ★ 통일 시드
            env = AStarEvacEnv(scenario=sc, n_agents=n_agents)
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

    results = eval_astar(n_agents=args.people, scenarios=args.scenarios, n_episodes=args.episodes)

    out = {"strategy": STRATEGY_NAME, "n_agents": args.people,
           "n_episodes": args.episodes, "fire_cost": FIRE_COST,
           "heuristic_weight": HEURISTIC_WEIGHT, "results": results}
    out_path = f"result_astar_{args.people}ppl.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
