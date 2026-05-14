"""
Simple A* Baseline | Fire Evacuation Stage 2
============================================

화재/연기/혼잡 비용을 전혀 반영하지 않는
순수 거리 기반 A* evacuation baseline.

[비교 구조]
  Simple A* (이 파일)
      : 사람 위치 → 출구까지 Pure A* shortest path
      → 거리 기준 출구 배정
      → [exit_A_cost, exit_B_cost, crowd_weight]
      → 유도등

  Hazard-aware A*
      : 화재 셀 차단 + 혼잡 고려 + 출구 분산

  PPO
      : 관측 → 신경망 → 유도등

[구현 원리]
  simple_astar_action(env):
    1. 각 생존자 위치에서 출구 A/B까지 A* 탐색
       - edge cost = 1
       - heuristic = Manhattan distance
       - 벽만 차단
       - 화재/연기/혼잡 무시
    2. 더 가까운 출구 선택
    3. 출구 선호 비율 → exit cost 변환
    4. 출구 근처 인원 차이 → crowd_weight

실행:
    python astar_simple_baseline.py --scenario 1 --n 10 --episodes 30
"""

import heapq
import sys
import numpy as np
import os
import csv
import json

from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from env_core import (
    FireEvacEnv,
    SCENARIO_CONFIGS,
    EXIT_A_POS,
    EXIT_B_POS,
    WALKABLE,
    QUEUE_RADIUS,
)


# ══════════════════════════════════════════════
# Manhattan heuristic
# ══════════════════════════════════════════════
def _heuristic(r: int, c: int, goals: set) -> int:
    return min(
        abs(r - gr) + abs(c - gc)
        for gr, gc in goals
    )


# ══════════════════════════════════════════════
# Pure shortest-path A*
# ══════════════════════════════════════════════
def _astar_to_exit(
    env: FireEvacEnv,
    start_r: int,
    start_c: int,
    exit_positions: list,
) -> float:
    """
    Pure A* shortest path.

    - 모든 edge cost = 1
    - heuristic = Manhattan distance
    - 벽만 차단
    - 화재/연기/혼잡 무시
    """

    goals = {(r, c) for r, c in exit_positions}

    if not goals:
        return np.inf

    if (start_r, start_c) in goals:
        return 0.0

    # (f_score, g_score, r, c)
    open_heap = []

    start_h = _heuristic(start_r, start_c, goals)

    heapq.heappush(
        open_heap,
        (start_h, 0, start_r, start_c)
    )

    closed = set()

    while open_heap:

        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) in closed:
            continue

        closed.add((r, c))

        if (r, c) in goals:
            return float(g)

        for dr, dc in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]:

            nr = r + dr
            nc = c + dc

            if not (0 <= nr < env.ROWS and 0 <= nc < env.COLS):
                continue

            # 벽만 차단
            if env.grid[nr, nc] not in WALKABLE:
                continue

            if (nr, nc) in closed:
                continue

            new_g = g + 1
            new_h = _heuristic(nr, nc, goals)
            new_f = new_g + new_h

            heapq.heappush(
                open_heap,
                (new_f, new_g, nr, nc)
            )

    return np.inf


# ══════════════════════════════════════════════
# A* 행동 결정
# ══════════════════════════════════════════════
def simple_astar_action(env: FireEvacEnv) -> np.ndarray:
    """
    Pure shortest-path A* evacuation policy.
    """

    n_alive = len(env.people_data)

    if n_alive == 0:
        return np.array(
            [10.0, 10.0, 2.0],
            dtype=np.float32
        )

    valid_a = [
        p for p in EXIT_A_POS
        if p not in env.blocked_exits
    ]

    valid_b = [
        p for p in EXIT_B_POS
        if p not in env.blocked_exits
    ]

    prefer_a = 0

    for person in env.people_data:

        r, c = person["pos"]

        dist_a = _astar_to_exit(
            env,
            r,
            c,
            valid_a,
        )

        dist_b = _astar_to_exit(
            env,
            r,
            c,
            valid_b,
        )

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

    # 출구 주변 혼잡 불균형
    dist_a_map = env._dist_to_exit_A
    dist_b_map = env._dist_to_exit_B

    near_a = (
        sum(
            1
            for p in env.people_data
            if dist_a_map[p["pos"][0], p["pos"][1]]
            <= QUEUE_RADIUS
        )
        if dist_a_map is not None
        else 0
    )

    near_b = (
        sum(
            1
            for p in env.people_data
            if dist_b_map[p["pos"][0], p["pos"][1]]
            <= QUEUE_RADIUS
        )
        if dist_b_map is not None
        else 0
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
def run_test(
    scenario: int,
    n_agents: int = 10,
    n_episodes: int = 30,
    save_results: bool = True,
    render: bool = False,
):

    cfg = SCENARIO_CONFIGS[scenario]

    render_mode = "human" if render else None

    print(f"\n{'═'*62}")
    print("  Simple A* Baseline")
    print(
        f"  Scenario {scenario}: "
        f"{cfg['name']} | "
        f"{n_agents} agents × {n_episodes} episodes"
    )
    print(f"{'═'*62}")

    records = []

    for ep in range(n_episodes):

        env = FireEvacEnv(
            scenario=scenario,
            n_agents=n_agents,
            render_mode=render_mode,
        )

        obs, info = env.reset()

        total_reward = 0.0
        max_fire = int(env.fire_map.sum())

        for _ in range(cfg["max_steps"]):

            action = simple_astar_action(env)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            max_fire = max(
                max_fire,
                info["fire_cells"]
            )

            if render:
                env.render()

            if terminated or truncated:
                break

        rec = {
            "episode": ep + 1,
            "scenario": scenario,
            "scenario_name": cfg["name"],
            "n_agents": n_agents,
            "escaped": info["escaped"],
            "escaped_A": info.get("escaped_A", 0),
            "escaped_B": info.get("escaped_B", 0),
            "dead": info["dead"],
            "remaining": info["remaining"],
            "survival_rate": round(
                info["survival_rate"],
                4,
            ),
            "total_reward": round(
                total_reward,
                2,
            ),
            "steps_taken": info["step"],
            "max_fire_cells": max_fire,
            "blocked_exits": str(
                info["blocked_exits"]
            ),
        }

        records.append(rec)

        print(
            f"  [ep {ep+1:>3}/{n_episodes}] "
            f"escape {rec['escaped']}/{n_agents} | "
            f"survival {rec['survival_rate']:.0%} | "
            f"{rec['steps_taken']} steps | "
            f"fire {rec['max_fire_cells']}"
        )

        env.close()

    # 통계
    def stats(vals):

        arr = np.array(vals, dtype=float)

        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "median": round(float(np.median(arr)), 4),
        }

    summary = {
        "model": "simple_astar_baseline",
        "scenario": scenario,
        "scenario_name": cfg["name"],
        "n_agents": n_agents,
        "n_episodes": n_episodes,

        "survival_rate": stats(
            [r["survival_rate"] for r in records]
        ),

        "total_reward": stats(
            [r["total_reward"] for r in records]
        ),

        "steps_taken": stats(
            [r["steps_taken"] for r in records]
        ),

        "escaped": stats(
            [r["escaped"] for r in records]
        ),

        "dead": stats(
            [r["dead"] for r in records]
        ),

        "max_fire_cells": stats(
            [r["max_fire_cells"] for r in records]
        ),
    }

    print(f"\n{'─'*62}")

    for key in [
        "survival_rate",
        "steps_taken",
        "escaped",
        "dead",
    ]:

        s = summary[key]

        print(
            f"  {key:<16} "
            f"mean={s['mean']:>8}  "
            f"std={s['std']:>7}  "
            f"min={s['min']:>7}  "
            f"max={s['max']:>7}"
        )

    print(f"{'═'*62}")

    # 저장
    if save_results:

        _res_dir = os.path.join(_ROOT, "result", "astar_simple")
        os.makedirs(_res_dir, exist_ok=True)

        ts = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        tag = f"s{scenario}_{n_agents}ppl_{ts}"

        csv_path = os.path.join(_res_dir, f"test_results_{tag}.csv")

        json_path = os.path.join(_res_dir, f"test_summary_{tag}.json"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=records[0].keys(),
            )

            writer.writeheader()
            writer.writerows(records)

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n  episode log : {csv_path}")
        print(f"  summary     : {json_path}")

    return records, summary


# ══════════════════════════════════════════════
# Entry
# ══════════════════════════════════════════════
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Simple A* Baseline"
    )

    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
    )

    parser.add_argument(
        "--all-scenarios",
        action="store_true",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
    )

    parser.add_argument(
        "--render",
        action="store_true",
    )

    args = parser.parse_args()

    scenarios = (
        [1, 2, 3, 4]
        if args.all_scenarios
        else [args.scenario]
    )

    all_summaries = {}

    for sc in scenarios:

        n_agents = (
            args.n
            if args.n is not None
            else SCENARIO_CONFIGS[sc]["n_agents"]
        )

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
        print("  Overall Survival Rate (Simple A*)")
        print(f"{'─'*62}")

        for sc, s in all_summaries.items():

            sr = s["survival_rate"]
            n = s["n_agents"]

            print(
                f"  S{sc} "
                f"{s['scenario_name']:<8} "
                f"({n:>2} agents) | "
                f"{sr['mean']:.1%} ± {sr['std']:.1%}"
            )

        print(f"{'═'*62}")