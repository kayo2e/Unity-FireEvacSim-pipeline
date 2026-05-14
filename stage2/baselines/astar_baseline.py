"""
화재 인식 BFS 베이스라인 | 화재대피유도시스템 Stage 2
=====================================================
PPO 모델과의 공정한 성능 비교를 위한 베이스라인.
rule_based_action()은 train.py의 BC 사전학습에서도 데모 생성용으로 재사용된다.

[비교 구조]
  Hazard BFS (이 파일) : env 상태(fire/smoke/occupancy) → 화재 셀 차단 BFS 탐색
                         → 출구별 최단 보행 거리 비교 → [exit_A_cost, exit_B_cost, crowd_weight] → 유도등
  Simple BFS           : 화재 포함 통과 가능한 순수 최단거리 BFS → 유도등
  PPO (단독)           : 환경 관측(15개 스칼라) → 신경망 → [exit_A_cost, exit_B_cost, crowd_weight] → 유도등
  PPO + BC 사전학습    : rule_based 데모로 신경망 초기화 → PPO RL 파인튜닝 → 유도등

[BFS 구현 원리]
  bfs_action(env):
    1. 각 생존자 위치에서 출구 A, B 각각으로 BFS 탐색
       - 화재 셀: 통과 불가 (blocked) — env_core._bfs_dist 와 동일 방식
       - 연기/혼잡 셀: 통과 가능, 비용=1 (균등)
       - 벽: 통과 불가
    2. dist_a vs dist_b 비교 → 각 생존자의 최적 출구 배정
    3. 선호 비율(ratio_a, ratio_b)을 exit_a_cost / exit_b_cost로 변환
    4. 실제 출구 근방 혼잡 불균형 → crowd_weight 결정

  rule_based_action(obs):
    BC 사전학습용 데모 생성에만 사용. obs(15 스칼라)만 입력받는 단순 휴리스틱.
    env 내부에 직접 접근하지 않으므로 PPO와 동일한 정보 수준 유지.

실행:
    python astar_baseline.py --scenario 1 --n 10 --episodes 30
    python astar_baseline.py --all-scenarios --n 10 --episodes 30
    python astar_baseline.py --all-scenarios --n 10 --episodes 30 --render
"""

from collections import deque
import sys
import numpy as np
import os, csv, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from env_core import FireEvacEnv, SCENARIO_CONFIGS, EXIT_A_POS, EXIT_B_POS, WALKABLE, QUEUE_RADIUS


# ══════════════════════════════════════════════
# 화재 인식 BFS (화재 셀 차단)
# ══════════════════════════════════════════════
def _bfs_to_exit(env: FireEvacEnv, start_r: int, start_c: int,
                 exit_positions: list) -> float:
    """
    (start_r, start_c)에서 exit_positions 중 하나까지 BFS 탐색.
    화재 셀은 통과 불가 (env_core._bfs_dist 방식과 동일).
    반환: 최단 스텝 수 (도달 불가능이면 inf)
    """
    goals = {(r, c) for r, c in exit_positions}
    if not goals:
        return np.inf
    if (start_r, start_c) in goals:
        return 0.0

    visited = {(start_r, start_c)}
    queue = deque([(start_r, start_c, 0)])

    while queue:
        r, c, dist = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.ROWS and 0 <= nc < env.COLS):
                continue
            if env.grid[nr, nc] not in WALKABLE:
                continue
            if env.fire_map[nr, nc] > 0:
                continue
            if (nr, nc) in visited:
                continue
            if (nr, nc) in goals:
                return float(dist + 1)
            visited.add((nr, nc))
            queue.append((nr, nc, dist + 1))

    return np.inf


# ══════════════════════════════════════════════
# BFS 행동 결정
# ══════════════════════════════════════════════
def bfs_action(env: FireEvacEnv) -> np.ndarray:
    """
    화재 인식 BFS 기반 행동 결정. env 내부 상태에 직접 접근.
    각 생존자 위치에서 출구 A/B까지 BFS를 각각 실행해 거리를 비교하고,
    선호 비율을 exit_a_cost / exit_b_cost로 변환한다.
    """
    n_alive = len(env.people_data)
    if n_alive == 0:
        return np.array([10.0, 10.0, 2.0], dtype=np.float32)

    valid_a = [p for p in EXIT_A_POS if p not in env.blocked_exits]
    valid_b = [p for p in EXIT_B_POS if p not in env.blocked_exits]

    prefer_a = 0
    for p in env.people_data:
        r, c = p["pos"]
        dist_a = _bfs_to_exit(env, r, c, valid_a)
        dist_b = _bfs_to_exit(env, r, c, valid_b)
        if dist_a <= dist_b:
            prefer_a += 1
    prefer_b = n_alive - prefer_a

    ratio_a = prefer_a / n_alive
    ratio_b = prefer_b / n_alive

    exit_a_cost = float(np.clip(5.0 + (1.0 - ratio_a) * 45.0, 5.0, 50.0))
    exit_b_cost = float(np.clip(5.0 + (1.0 - ratio_b) * 45.0, 5.0, 50.0))

    bfs_a = env._dist_to_exit_A
    bfs_b = env._dist_to_exit_B
    near_a = (sum(1 for p in env.people_data
                  if bfs_a[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
              if bfs_a is not None else 0)
    near_b = (sum(1 for p in env.people_data
                  if bfs_b[p["pos"][0], p["pos"][1]] <= QUEUE_RADIUS)
              if bfs_b is not None else 0)
    queue_ratio = abs(near_a - near_b) / n_alive
    crowd_weight = float(np.clip(0.5 + queue_ratio * 4.5, 0.5, 5.0))

    return np.array([exit_a_cost, exit_b_cost, crowd_weight], dtype=np.float32)


# ══════════════════════════════════════════════════════
# 하위 호환: astar_action → bfs_action 으로 전달
# ══════════════════════════════════════════════════════
def astar_action(env: FireEvacEnv) -> np.ndarray:
    return bfs_action(env)


# ══════════════════════════════════════════════
# BC 사전학습용 규칙 기반 정책 (obs만 입력)
# ══════════════════════════════════════════════
def rule_based_action(obs: np.ndarray) -> np.ndarray:
    """
    BC 사전학습 데모 생성 전용. obs(15개 스칼라)만 입력으로 받는 단순 휴리스틱.
    env 내부에 직접 접근하지 않으므로 PPO와 동일한 정보 수준.

    사용 피처:
      F1 (obs[0]): 출구A 위협 (1=안전, 0=위험)
      F2 (obs[1]): 출구B 위협 (1=안전, 0=위험)
      F4 (obs[3]): 탈출 완료 비율
      F5 (obs[4]): 사망 비율
    """
    f1 = float(obs[0])
    f2 = float(obs[1])
    f4 = float(obs[3])
    f5 = float(obs[4])

    def threat_to_cost(threat: float) -> float:
        return float(np.clip(50.0 - threat * 45.0, 5.0, 50.0))

    exit_a_cost  = threat_to_cost(f1)
    exit_b_cost  = threat_to_cost(f2)
    remaining    = max(0.0, 1.0 - f4 - f5)
    crowd_weight = float(np.clip(0.5 + remaining * 4.5, 0.5, 5.0))

    return np.array([exit_a_cost, exit_b_cost, crowd_weight], dtype=np.float32)


# ══════════════════════════════════════════════
# 배치 테스트
# ══════════════════════════════════════════════
def run_test(scenario: int, n_agents: int = 10, n_episodes: int = 30,
             save_results: bool = True, render: bool = False):

    cfg = SCENARIO_CONFIGS[scenario]
    render_mode = "human" if render else None

    print(f"\n{'═'*62}")
    print(f"  Hazard BFS 베이스라인 (화재 셀 차단)")
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
            action = bfs_action(env)
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
        "model":          "bfs_hazard_baseline",
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
        _res_dir = os.path.join(_ROOT, "result", "astar_baseline")
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

    parser = argparse.ArgumentParser(description="Hazard BFS 베이스라인 (화재 셀 차단)")
    parser.add_argument("--scenario",      type=int,  default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--all-scenarios", action="store_true", help="시나리오 1~4 전부 실행")
    parser.add_argument("--n",             type=int,  default=None,
                        help="인원수 (미지정 시 시나리오 권장값 자동 적용)")
    parser.add_argument("--episodes",      type=int,  default=30, help="에피소드 수")
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
        print("  전체 시나리오 생존율 (Hazard BFS — 화재 셀 차단)")
        print(f"{'─'*62}")
        for sc, s in all_summaries.items():
            sr = s["survival_rate"]
            n  = s["n_agents"]
            print(f"  S{sc} {s['scenario_name']:<8} ({n:>2}명) | "
                  f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                  f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
        print(f"{'═'*62}")
        print("\n  ※ PPO 모델 결과와 비교하려면:")
        for sc in scenarios:
            n = SCENARIO_CONFIGS[sc]["n_agents"]
            print(f"    python train.py --mode test --test-n {n} "
                  f"--test-scenario {sc} --test-episodes 30")
