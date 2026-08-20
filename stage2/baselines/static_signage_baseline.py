"""
정적 유도등(Static Signage) 베이스라인
======================================

동적 유도등 RL 문헌의 표준 비교군 — 실제 건물의 기본 상태를 모사한다.
모든 표지판이 화재·군중 상태와 무관하게 **최초 1회 계산한 방향으로 고정**되고,
이후 절대 갱신되지 않는다(EC directive 92/58/EEC 준수 표준 표지판과 동일한
설계 원칙 — "가장 가까운 출구"만 가리키고 실시간 위험은 반영하지 않음).

astar_real.py(Pure A*)와의 차이: astar_real은 화재/연기를 무시하되 매 스텝
재탐색해서 사람 이동에는 반응한다. 이 베이스라인은 그마저도 하지 않는다 —
최초 인원 배치 기준으로 딱 한 번만 계산한다.

주의: 이 시뮬레이션 엔진(`env_core._compute_bfs_with_risk`)은 화재 셀 통과
비용이 항상 켜져 있어(구조적 안전장치), 정적 유도등이라도 그리드 차원의 최소
위험 회피는 공유한다 — 이는 실제 건물에서도 사람이 눈앞의 불길을 피하는 것과
같은 최소한의 현실성이며, "출구 배정 비율·혼잡 가중치를 절대 갱신하지 않는다"는
정적 신호의 핵심 정의는 그대로 유지된다.

사용:
    action = static_signage_action(env)
"""

import heapq
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from env_core import EXIT_A_POS, EXIT_B_POS, WALKABLE, QUEUE_RADIUS


def _heuristic(r: int, c: int, goals: list) -> int:
    return min(abs(r - gr) + abs(c - gc) for gr, gc in goals)


def _manhattan_to_exit(env, r: int, c: int, goals: list) -> float:
    """벽만 피하는 순수 Manhattan 거리 — astar_real._astar_to_exit과 동일한
    화재/연기 무시 원칙, 초기 배치 1회 계산용이라 간이 BFS로 충분."""
    if not goals:
        return float("inf")
    from collections import deque
    dist = {(r, c): 0}
    q = deque([(r, c)])
    goal_set = set(goals)
    while q:
        cr, cc = q.popleft()
        if (cr, cc) in goal_set:
            return dist[(cr, cc)]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < env.ROWS and 0 <= nc < env.COLS
                    and env.grid[nr, nc] in WALKABLE
                    and (nr, nc) not in dist):
                dist[(nr, nc)] = dist[(cr, cc)] + 1
                q.append((nr, nc))
    return float("inf")


def static_signage_action(env) -> np.ndarray:
    """
    최초 호출 시 인원 초기 배치만으로 출구 배정 비율을 계산해 env 인스턴스에
    캐시하고, 이후 매 호출마다 캐시된 값을 그대로 반환한다 — 화재·군중 상태가
    바뀌어도 절대 재계산하지 않는다. exp1_compare.py의 run_astar 루프처럼
    에피소드마다 새 env 인스턴스를 생성하는 패턴이면 캐시가 자동으로
    에피소드 경계에서 초기화된다.
    """
    cached = getattr(env, "_static_signage_cache", None)
    if cached is not None:
        return cached

    n_alive = len(env.people_data)
    if n_alive == 0:
        action = np.array([10.0, 10.0, 2.0], dtype=np.float32)
        env._static_signage_cache = action
        return action

    valid_a = [p for p in EXIT_A_POS if p not in env.blocked_exits]
    valid_b = [p for p in EXIT_B_POS if p not in env.blocked_exits]

    prefer_a = 0
    for p in env.people_data:
        r, c = p["pos"]
        dist_a = _manhattan_to_exit(env, r, c, valid_a)
        dist_b = _manhattan_to_exit(env, r, c, valid_b)
        if dist_a <= dist_b:
            prefer_a += 1
    prefer_b = n_alive - prefer_a

    ratio_a = prefer_a / n_alive
    ratio_b = prefer_b / n_alive

    # 출구 비용 변환 — astar_real과 동일한 스케일(고정 신호 강도로 해석)
    exit_a_cost = float(np.clip(5.0 + (1.0 - ratio_a) * 45.0, 5.0, 50.0))
    exit_b_cost = float(np.clip(5.0 + (1.0 - ratio_b) * 45.0, 5.0, 50.0))
    # 정적 신호는 혼잡도를 아예 관측하지 않음 — 중립값 고정
    crowd_weight = 2.0

    action = np.array([exit_a_cost, exit_b_cost, crowd_weight], dtype=np.float32)
    env._static_signage_cache = action
    return action


# ══════════════════════════════════════════════
# 배치 테스트 (astar_real.py의 run_test와 동일한 CSV/JSON 스키마)
# ══════════════════════════════════════════════
def run_test(scenario: int, n_agents: int = 10, n_episodes: int = 30,
             save_results: bool = True, render: bool = False):
    import csv, json
    from datetime import datetime
    from env_core import FireEvacEnv, SCENARIO_CONFIGS

    cfg = SCENARIO_CONFIGS[scenario]
    render_mode = "human" if render else None

    print(f"\n{'═'*62}")
    print(f"  정적 유도등 베이스라인 (최초 1회 계산 후 고정, 화재/군중 무관)")
    print(f"  시나리오 {scenario}: {cfg['name']} | {n_agents}명 × {n_episodes}회")
    print(f"{'═'*62}")

    records = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents, render_mode=render_mode)
        obs, info = env.reset()

        total_r = 0.0
        max_fire = int(env.fire_map.sum())

        for _ in range(cfg["max_steps"]):
            action = static_signage_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            max_fire = max(max_fire, info["fire_cells"])
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
              f"생존율 {rec['survival_rate']:.0%} | {rec['steps_taken']}스텝")
        env.close()

    def stats(vals):
        a = np.array(vals, dtype=float)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
                "median": round(float(np.median(a)), 4)}

    summary = {
        "model": "static_signage_baseline",
        "scenario": scenario, "scenario_name": cfg["name"],
        "n_agents": n_agents, "n_episodes": n_episodes,
        "survival_rate": stats([r["survival_rate"] for r in records]),
        "total_reward":  stats([r["total_reward"]  for r in records]),
        "steps_taken":   stats([r["steps_taken"]   for r in records]),
        "escaped":       stats([r["escaped"]       for r in records]),
        "escaped_A":     stats([r["escaped_A"]     for r in records]),
        "escaped_B":     stats([r["escaped_B"]     for r in records]),
        "dead":          stats([r["dead"]          for r in records]),
    }

    print(f"\n{'─'*62}")
    for key in ("survival_rate", "steps_taken", "escaped", "dead"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}  median={s['median']:>8}")
    print(f"{'═'*62}")

    if save_results:
        res_dir = os.path.join(_ROOT, "result", "static_signage")
        os.makedirs(res_dir, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"s{scenario}_{n_agents}ppl_{ts}"

        csv_path  = os.path.join(res_dir, f"test_results_{tag}.csv")
        json_path = os.path.join(res_dir, f"test_summary_{tag}.json")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  에피소드 기록: {csv_path}")
        print(f"  통계 요약    : {json_path}")

    return records, summary


if __name__ == "__main__":
    import argparse
    from env_core import SCENARIO_CONFIGS

    parser = argparse.ArgumentParser(description="정적 유도등 베이스라인")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    scenarios = list(SCENARIO_CONFIGS.keys()) if args.all_scenarios else [args.scenario]
    for sc in scenarios:
        n = SCENARIO_CONFIGS[sc]["n_agents"] if args.all_scenarios else args.n
        run_test(sc, n_agents=n, n_episodes=args.episodes, save_results=not args.no_save)
