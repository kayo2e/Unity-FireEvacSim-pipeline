"""
compute_extra_metrics.py — 표 3 보조 지표 (Exit Balance / Throughput / Path Efficiency)
===========================================================================
exp1_compare.py가 저장한 기존 CSV(exp1_s{N}_s{N}_*.csv)를 읽어 재실험 없이
계산한다 — escaped_A/escaped_B/steps_taken은 이미 CSV에 있음.

지표 정의:
  Exit Balance   = 1 - |A - B| / (A + B)   (1.0 = 완벽한 양쪽 출구 균등 사용,
                   0.0 = 한쪽 출구만 사용) — README의 "F7/F8로 출구 분산 유도"
                   주장을 뒷받침하는 정량 지표
  Throughput     = (A + B) / steps_taken   (스텝당 처리 인원 — 병목 완화 효과)
  Path Efficiency = grid_min_steps / steps_taken (1.0에 가까울수록 이론적
                   최단 스텝에 근접) — grid_min_steps는 에피소드별 실제 시작
                   위치가 CSV에 없어 근사치를 쓴다: 시나리오 스폰 구역에서
                   가장 가까운 출구까지의 평균 BFS 거리(env_core의 BFS 거리맵
                   기준, 화재 무시) — 진짜 에피소드별 최적값이 아니라 "이
                   시나리오에서 화재가 없었다면 대략 몇 스텝이 걸렸을까"에
                   대한 참고선(reference line)이다. 정밀한 값이 필요하면
                   초기 위치를 기록하는 recordings/*.jsonl 기반 재계산이 필요
                   — 이 스크립트는 1차 근사치만 제공한다.

실행:
    python experiments/compute_extra_metrics.py --scenario 4 \
        --csv result/exp1_compare/exp1_s4_s4_20260820_160945.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_core import FireEvacEnv, SCENARIO_CONFIGS


def _grid_min_steps_reference(scenario: int, n_agents: int, n_samples: int = 20) -> float:
    """화재 없는 상태에서 스폰 위치 → 가장 가까운 출구까지의 평균 BFS 거리를
    n_samples번 reset해 추정 — 진짜 최적값이 아니라 근사 참고선."""
    dists = []
    for seed in range(n_samples):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents, hazard_aware=False)
        env.reset(seed=1000 + seed)
        bfs_a = env._dist_to_exit_A
        bfs_b = env._dist_to_exit_B
        for p in env.people_data:
            r, c = p["pos"]
            da = bfs_a[r, c] if bfs_a is not None else np.inf
            db = bfs_b[r, c] if bfs_b is not None else np.inf
            d = min(da, db)
            if np.isfinite(d):
                dists.append(d)
        env.close()
    return float(np.mean(dists)) if dists else float("nan")


def compute(csv_path: str, scenario: int):
    rows_by_model = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows_by_model.setdefault(row["model"], []).append(row)

    n_agents = SCENARIO_CONFIGS[scenario]["n_agents"]
    ref_steps = _grid_min_steps_reference(scenario, n_agents)
    print(f"참고선(화재 없을 때 평균 BFS 거리): {ref_steps:.1f} 스텝\n")

    print(f"{'모델':<10} {'Exit Balance':>14} {'Throughput':>12} {'Path Efficiency':>16}")
    print("-" * 56)
    for model, rows in rows_by_model.items():
        balances, throughputs, efficiencies = [], [], []
        for r in rows:
            a, b = float(r["escaped_A"]), float(r["escaped_B"])
            steps = float(r["steps_taken"])
            total_escaped = a + b
            if total_escaped > 0:
                balances.append(1.0 - abs(a - b) / total_escaped)
                throughputs.append(total_escaped / steps if steps > 0 else 0.0)
            if steps > 0 and np.isfinite(ref_steps):
                efficiencies.append(min(ref_steps / steps, 1.0))
        b_mean = np.mean(balances) if balances else float("nan")
        t_mean = np.mean(throughputs) if throughputs else float("nan")
        e_mean = np.mean(efficiencies) if efficiencies else float("nan")
        print(f"{model:<10} {b_mean:>14.3f} {t_mean:>12.3f} {e_mean:>16.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--csv", type=str, required=True,
                        help="exp1_compare.py가 저장한 시나리오별 CSV 경로")
    args = parser.parse_args()
    compute(args.csv, args.scenario)
