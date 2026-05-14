"""
A* 최단경로 베이스라인
====================

- Manhattan heuristic 기반 A*
- 화재/연기/혼잡 무시
- 벽만 통과 불가
- Pure shortest-path baseline

Simple BFS 와 차이:
  BFS : uninformed search
  A*  : heuristic-informed shortest path

사용:
    action = astar_action(env)
"""

import heapq
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from env_core import (
    FireEvacEnv,
    EXIT_A_POS,
    EXIT_B_POS,
    WALKABLE,
    QUEUE_RADIUS,
)


# ══════════════════════════════════════════════
# Manhattan heuristic
# ══════════════════════════════════════════════
def _heuristic(r: int, c: int, goals: list) -> int:
    return min(abs(r - gr) + abs(c - gc) for gr, gc in goals)


# ══════════════════════════════════════════════
# Pure A* shortest path
# ══════════════════════════════════════════════
def _astar_to_exit(
    env: FireEvacEnv,
    start_r: int,
    start_c: int,
    exit_positions: list,
) -> float:
    """
    Pure shortest-path A*.

    - 벽만 차단
    - 화재/연기/혼잡 무시
    - 모든 이동 비용 = 1
    """

    goals = {(r, c) for r, c in exit_positions}

    if not goals:
        return np.inf

    if (start_r, start_c) in goals:
        return 0.0

    # (f_score, g_score, r, c)
    pq = []

    start_h = _heuristic(start_r, start_c, exit_positions)

    heapq.heappush(
        pq,
        (start_h, 0, start_r, start_c)
    )

    visited = set()

    while pq:
        f, g, r, c = heapq.heappop(pq)

        if (r, c) in visited:
            continue

        visited.add((r, c))

        if (r, c) in goals:
            return float(g)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < env.ROWS and 0 <= nc < env.COLS):
                continue

            # 벽만 차단
            if env.grid[nr, nc] not in WALKABLE:
                continue

            if (nr, nc) in visited:
                continue

            new_g = g + 1
            new_h = _heuristic(nr, nc, exit_positions)
            new_f = new_g + new_h

            heapq.heappush(
                pq,
                (new_f, new_g, nr, nc)
            )

    return np.inf


# ══════════════════════════════════════════════
# A* 행동 결정
# ══════════════════════════════════════════════
def astar_action(env: FireEvacEnv) -> np.ndarray:
    """
    Pure shortest-path A* evacuation policy.
    """

    n_alive = len(env.people_data)

    if n_alive == 0:
        return np.array([10.0, 10.0, 2.0], dtype=np.float32)

    valid_a = [
        p for p in EXIT_A_POS
        if p not in env.blocked_exits
    ]

    valid_b = [
        p for p in EXIT_B_POS
        if p not in env.blocked_exits
    ]

    prefer_a = 0

    for p in env.people_data:
        r, c = p["pos"]

        dist_a = _astar_to_exit(env, r, c, valid_a)
        dist_b = _astar_to_exit(env, r, c, valid_b)

        if dist_a <= dist_b:
            prefer_a += 1

    prefer_b = n_alive - prefer_a

    ratio_a = prefer_a / n_alive
    ratio_b = prefer_b / n_alive

    # 출구 비용 변환
    exit_a_cost = float(
        np.clip(
            5.0 + (1.0 - ratio_a) * 45.0,
            5.0,
            50.0,
        )
    )

    exit_b_cost = float(
        np.clip(
            5.0 + (1.0 - ratio_b) * 45.0,
            5.0,
            50.0,
        )
    )

    # crowd weight
    bfs_a = env._dist_to_exit_A
    bfs_b = env._dist_to_exit_B

    near_a = (
        sum(
            1 for p in env.people_data
            if bfs_a[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS
        )
        if bfs_a is not None else 0
    )

    near_b = (
        sum(
            1 for p in env.people_data
            if bfs_b[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS
        )
        if bfs_b is not None else 0
    )

    queue_ratio = abs(near_a - near_b) / n_alive

    crowd_weight = float(
        np.clip(
            0.5 + queue_ratio * 4.5,
            0.5,
            5.0,
        )
    )

    return np.array(
        [
            exit_a_cost,
            exit_b_cost,
            crowd_weight,
        ],
        dtype=np.float32,
    )


# ══════════════════════════════════════════════
# 배치 테스트
# ══════════════════════════════════════════════
def run_test(scenario: int, n_agents: int = 10, n_episodes: int = 30,
             save_results: bool = True, render: bool = False):
    import os, csv, json
    from datetime import datetime

    from env_core import SCENARIO_CONFIGS

    cfg = SCENARIO_CONFIGS[scenario]
    render_mode = "human" if render else None

    print(f"\n{'═'*62}")
    print(f"  Pure A* 베이스라인 (Manhattan 휴리스틱, 화재/연기/혼잡 무시)")
    print(f"  시나리오 {scenario}: {cfg['name']} | {n_agents}명 × {n_episodes}회")
    print(f"{'═'*62}")

    records = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                          render_mode=render_mode)
        obs, info = env.reset()

        total_r  = 0.0
        max_fire = int(env.fire_map.sum())

        for _ in range(cfg["max_steps"]):
            action = astar_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r  += reward
            max_fire  = max(max_fire, info["fire_cells"])
            if render:
                env.render()
            if terminated or truncated:
                break

        rec = {
            "episode":        ep + 1,
            "scenario":       scenario,
            "scenario_name":  cfg["name"],
            "n_agents":       n_agents,
            "escaped":        info["escaped"],
            "escaped_A":      info.get("escaped_A", 0),
            "escaped_B":      info.get("escaped_B", 0),
            "dead":           info["dead"],
            "remaining":      info["remaining"],
            "survival_rate":  round(info["survival_rate"], 4),
            "total_reward":   round(total_r, 2),
            "steps_taken":    info["step"],
            "max_fire_cells": max_fire,
            "blocked_exits":  str(info["blocked_exits"]),
        }
        records.append(rec)
        print(f"  [ep {ep+1:>3}/{n_episodes}] 탈출 {rec['escaped']}/{n_agents} | "
              f"생존율 {rec['survival_rate']:.0%} | "
              f"{rec['steps_taken']}스텝 | "
              f"화재셀 {rec['max_fire_cells']}")
        env.close()

    def stats(vals):
        a = np.array(vals, dtype=float)
        return {
            "mean":   round(float(a.mean()), 4),
            "std":    round(float(a.std()),  4),
            "min":    round(float(a.min()),  4),
            "max":    round(float(a.max()),  4),
            "median": round(float(np.median(a)), 4),
        }

    summary = {
        "model":          "astar_real_baseline",
        "scenario":       scenario,
        "scenario_name":  cfg["name"],
        "n_agents":       n_agents,
        "n_episodes":     n_episodes,
        "survival_rate":  stats([r["survival_rate"]   for r in records]),
        "total_reward":   stats([r["total_reward"]    for r in records]),
        "steps_taken":    stats([r["steps_taken"]     for r in records]),
        "escaped":        stats([r["escaped"]          for r in records]),
        "escaped_A":      stats([r["escaped_A"]        for r in records]),
        "escaped_B":      stats([r["escaped_B"]        for r in records]),
        "dead":           stats([r["dead"]             for r in records]),
        "max_fire_cells": stats([r["max_fire_cells"]   for r in records]),
    }

    print(f"\n{'─'*62}")
    for key in ("survival_rate", "steps_taken", "escaped", "dead"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}  median={s['median']:>8}")
    print(f"{'═'*62}")

    if save_results:
        _res_dir = os.path.join(_ROOT, "result", "astar_real")
        os.makedirs(_res_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"s{scenario}_{n_agents}ppl_{ts}"

        csv_path  = os.path.join(_res_dir, f"test_results_{tag}.csv")
        json_path = os.path.join(_res_dir, f"test_summary_{tag}.json")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  에피소드 기록: {csv_path}")
        print(f"  통계 요약    : {json_path}")

    return records, summary


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    from env_core import SCENARIO_CONFIGS

    parser = argparse.ArgumentParser(
        description="Pure A* 베이스라인 (Manhattan 휴리스틱, 화재 무시)")
    parser.add_argument("--scenario",      type=int,  default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--all-scenarios", action="store_true", help="시나리오 1~4 전부 실행")
    parser.add_argument("--n",             type=int,  default=None,
                        help="인원수 (미지정 시 시나리오 권장값 자동 적용)")
    parser.add_argument("--episodes",      type=int,  default=30)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    args = parser.parse_args()

    scenarios = [1, 2, 3, 4] if args.all_scenarios else [args.scenario]

    all_summaries = {}
    for sc in scenarios:
        n_agents = args.n if args.n is not None else SCENARIO_CONFIGS[sc]["n_agents"]
        _, summary = run_test(
            scenario=sc,
            n_agents=n_agents,
            n_episodes=args.episodes,
            save_results=not args.no_save,
            render=args.render,
        )
        all_summaries[sc] = summary

    if args.all_scenarios:
        print(f"\n{'═'*62}")
        print("  전체 시나리오 생존율 (Pure A* — Manhattan 휴리스틱)")
        print(f"{'─'*62}")
        for sc, s in all_summaries.items():
            sr = s["survival_rate"]
            n  = s["n_agents"]
            print(f"  S{sc} {s['scenario_name']:<8} ({n:>2}명) | "
                  f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                  f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
        print(f"{'═'*62}")