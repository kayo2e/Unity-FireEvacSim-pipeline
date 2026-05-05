"""
A* 규칙 기반 베이스라인 | 화재대피유도시스템 Stage 2
=====================================================
PPO 모델과의 공정한 성능 비교를 위한 베이스라인.
rule_based_action()은 train.py의 BC 사전학습에서도 데모 생성용으로 재사용된다.

[비교 구조]
  A* 베이스라인        : 환경 관측 → 규칙 기반 → [exit_A_cost, exit_B_cost, crowd_weight] → 유도등
  PPO (단독)           : 환경 관측 → 신경망   → [exit_A_cost, exit_B_cost, crowd_weight] → 유도등
  PPO + BC 사전학습    : A* 데모로 신경망 초기화 → PPO RL 파인튜닝 → 유도등

제어 메커니즘(유도등 → 사람 이동)은 세 방식 모두 완전히 동일.
차이는 비용 결정 방식: 규칙 / 학습 / A*로 초기화된 학습.

[BC 사전학습 연동 — train.py]
  train.py는 학습 시작 전 이 파일의 rule_based_action()을 호출해
  A* 행동 시연(demonstration)을 수집한 뒤 PPO 신경망을 지도학습으로 초기화한다.
  이후 PPO RL로 파인튜닝해 A* 수준에서 출발해 분산 유도까지 추가 학습한다.

  collect_astar_demos()  → 12,000 샘플 수집 (3000스텝 × 4환경)
  pretrain_bc()          → MSE 지도학습으로 PPO 신경망 초기화 (기본 5에폭)
  model.learn()          → PPO RL 파인튜닝

[병목(Bottleneck) 설계]
  - EXIT_CAPACITY: 출구 셀당 스텝당 최대 탈출 인원 제한 (train.py와 동일 환경)
  - CELL_CAPACITY: 복도 셀당 최대 점유 인원 제한
  → 두 모델 모두 동일한 병목 환경에서 평가됨 (공정 비교)

[A* 규칙 기반 정책의 한계 — 분산유도 불가]
  - 각 출구까지 화재 BFS 거리만 고려 (출구별 큐 길이 무시)
  - crowd_weight는 전체 잔여 인원 비율로 결정 (출구별 혼잡도 미분화)
  - 병목 발생 시 모든 인원이 가까운 출구에 집중 → 대기열 형성 → 화재 도달 시 집단 사망
  - F7/F8(출구 근접 혼잡도) 피처를 사용하지 않아 부하 분산 불가

[PPO + BC 사전학습의 강점]
  - A* 수준의 기본 경로 선택에서 학습 시작 → 초반 랜덤 탐색 낭비 없음
  - F7/F8 피처로 출구별 혼잡도를 실시간 관측해 분산 유도 추가 학습
  - exit_A_cost / exit_B_cost를 차별화해 혼잡한 출구를 우회
  - 두 출구에 분산 탈출 시 +15 보상 → 분산 정책 강화

실행:
    python astar_baseline.py --scenario 1 --n 10 --episodes 30
    python astar_baseline.py --all-scenarios --n 10 --episodes 30
    python astar_baseline.py --all-scenarios --n 10 --episodes 30 --render
"""

import sys
import numpy as np
import os, csv, json
from datetime import datetime
from collections import deque

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 환경 및 상수 공유
from env_core import FireEvacEnv, SCENARIO_CONFIGS
from env_core import (
    EXIT_A_POS, EXIT_B_POS,
    WALKABLE, EXIT, DELTA,
    BASE_GRID,
)

ROWS, COLS = BASE_GRID.shape


# ══════════════════════════════════════════════
# 규칙 기반 정책: 출구별 화재 위협 → 비용 결정
# ══════════════════════════════════════════════
def compute_fire_dist_to_exit(grid, fire_map, exit_positions: list, blocked_exits: set) -> float:
    """
    BFS로 화재 셀에서 출구까지의 최단 거리를 반환.
    (값이 작을수록 화재가 출구에 가깝다 → 위험)
    출구가 막혔거나 화재에 완전히 둘러싸인 경우 0 반환.
    """
    valid_exits = [(r, c) for (r, c) in exit_positions
                   if grid[r, c] == EXIT and (r, c) not in blocked_exits]
    if not valid_exits:
        return 0.0  # 출구 자체가 없음 → 최대 위험

    fire_cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                  if fire_map[r, c] > 0]
    if not fire_cells:
        return 999.0  # 화재 없음 → 위험 없음

    # 화재 셀 → 출구까지 BFS
    dist = np.full((ROWS, COLS), 9999.0)
    q = deque()
    for (r, c) in fire_cells:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DELTA.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < ROWS and 0 <= nc < COLS
                    and dist[nr, nc] == 9999
                    and grid[nr, nc] in WALKABLE):
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    return float(min(dist[r, c] for (r, c) in valid_exits))


def rule_based_action(env: FireEvacEnv) -> np.ndarray:
    """
    현재 환경 상태를 보고 규칙으로 [exit_A_cost, exit_B_cost, crowd_weight] 결정.

    [A* 규칙 — 분산유도 불가 (베이스라인)]
      1. 화재 BFS 거리 기반으로만 출구 비용 결정
      2. 출구별 큐 길이(F7/F8) 미사용 → 혼잡도 인식 불가
      3. crowd_weight는 전체 잔여 인원 비율로만 결정 (출구별 미분화)

    ※ 병목 환경(EXIT_CAPACITY)에서 A*는 가장 가까운 출구에 인원이 집중되어
      대기열이 형성되어도 이를 감지·분산하지 못한다. (PPO와의 핵심 차이)
    """
    dist_a = compute_fire_dist_to_exit(
        env.grid, env.fire_map, EXIT_A_POS, env.blocked_exits)
    dist_b = compute_fire_dist_to_exit(
        env.grid, env.fire_map, EXIT_B_POS, env.blocked_exits)

    # 출구 폐쇄 여부 확인
    a_blocked = all(cell in env.blocked_exits for cell in EXIT_A_POS)
    b_blocked = all(cell in env.blocked_exits for cell in EXIT_B_POS)

    # 화재 거리 → 비용 변환 (출구 큐 길이는 고려 안 함)
    def dist_to_cost(d, blocked):
        if blocked:
            return 50.0
        d = min(d, 20.0)
        cost = 50.0 - (d / 20.0) * 45.0
        return float(np.clip(cost, 5.0, 50.0))

    exit_a_cost = dist_to_cost(dist_a, a_blocked)
    exit_b_cost = dist_to_cost(dist_b, b_blocked)

    # crowd_weight: 전체 잔여 인원 비율 — 출구별 혼잡도 미분화
    remaining_ratio = len(env.people_data) / max(env.n_agents, 1)
    crowd_weight = float(np.clip(0.5 + remaining_ratio * 4.5, 0.5, 5.0))

    return np.array([exit_a_cost, exit_b_cost, crowd_weight], dtype=np.float32)


# ══════════════════════════════════════════════
# 배치 테스트
# ══════════════════════════════════════════════
def run_test(scenario: int, n_agents: int = 10, n_episodes: int = 30,
             save_results: bool = True, render: bool = False):

    cfg = SCENARIO_CONFIGS[scenario]
    render_mode = "human" if render else None

    print(f"\n{'═'*62}")
    print(f"  A* 규칙 기반 베이스라인")
    print(f"  시나리오 {scenario}: {cfg['name']} | {n_agents}명 × {n_episodes}회")
    print(f"{'═'*62}")

    records = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                          render_mode=render_mode)
        _, info = env.reset()

        total_r  = 0.0
        max_fire = int(env.fire_map.sum())

        for _ in range(cfg["max_steps"]):
            action = rule_based_action(env)
            _, reward, terminated, truncated, info = env.step(action)
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

    # 통계
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
        "model":          "astar_rule_baseline",
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
        os.makedirs("result/astar_baseline", exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"s{scenario}_{n_agents}ppl_{ts}"

        csv_path  = f"result/astar_baseline/test_results_{tag}.csv"
        json_path = f"result/astar_baseline/test_summary_{tag}.json"

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

    parser = argparse.ArgumentParser(description="A* 규칙 기반 베이스라인")
    parser.add_argument("--scenario",      type=int,  default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--all-scenarios", action="store_true", help="시나리오 1~4 전부 실행")
    parser.add_argument("--n",             type=int,  default=10, help="인원수")
    parser.add_argument("--episodes",      type=int,  default=30, help="에피소드 수")
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    args = parser.parse_args()

    scenarios = [1, 2, 3, 4] if args.all_scenarios else [args.scenario]

    all_summaries = {}
    for sc in scenarios:
        _, summary = run_test(
            scenario=sc,
            n_agents=args.n,
            n_episodes=args.episodes,
            save_results=not args.no_save,
            render=args.render,
        )
        all_summaries[sc] = summary

    if args.all_scenarios:
        print(f"\n{'═'*62}")
        print("  전체 시나리오 생존율 (A* 규칙 기반 베이스라인)")
        print(f"{'─'*62}")
        for sc, s in all_summaries.items():
            sr = s["survival_rate"]
            print(f"  S{sc} {s['scenario_name']:<8} | "
                  f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                  f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
        print(f"{'═'*62}")
        print("\n  ※ PPO 모델 결과와 비교하려면:")
        print("    python train.py --mode test --test-n 10 --test-scenario X --test-episodes 30")
