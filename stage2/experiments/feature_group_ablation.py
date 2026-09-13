"""
feature_group_ablation.py — F1~F15 그룹 단위 마스킹 (Task 2, Phase 1 후속)
================================================================================
feature_ablation.py의 단일 피처 마스킹에서 모든 낙폭이 <0.5%p로 나온 결과에
대한 후속 검증. 개별 피처가 안 중요한 게 아니라, 같은 정보(출구 선호/위협)를
여러 피처가 중복 인코딩해서 하나를 지워도 나머지가 메꿔주는 것인지 확인한다.
같은 정보를 담는다고 가정한 피처들을 "묶어서" 동시에 마스킹해 낙폭을 본다.

그룹:
  HAZARD     F1,F2,F12,F13,F14,F15  절대/방향성 화재 위협
  CROWD      F3,F7,F8,F9,F10,F11    출구 선호·혼잡·공황
  PROGRESS   F4,F5,F6               에피소드 진행도
  ALL_BLIND  F1~F15 전체            정책이 관측을 완전히 못 보는 극단 대조군
             (이 조건까지 낙폭이 작으면, 유도등 정책 자체보다 환경의 자체 반응
             동역학이 생존율을 지배한다는 뜻이라 프레임워크 전체의 핵심 주장과
             직결된다 — 반드시 같이 확인해야 함)

실행:
    cd stage2
    python experiments/feature_group_ablation.py --scenarios 2 3 4 5 --episodes 20
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_ablation import run_condition, FEATURE_NAMES
from env_core import SCENARIO_CONFIGS

GROUPS = {
    "HAZARD":   [0, 1, 11, 12, 13, 14],   # F1,F2,F12,F13,F14,F15
    "CROWD":    [2, 6, 7, 8, 9, 10],      # F3,F7,F8,F9,F10,F11
    "PROGRESS": [3, 4, 5],                # F4,F5,F6
    "ALL_BLIND": list(range(15)),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=os.path.join("model", "ppo"))
    parser.add_argument("--model-n", type=int, default=40)
    args = parser.parse_args()

    summary = {}
    for sc in args.scenarios:
        n_agents = SCENARIO_CONFIGS[sc]["n_agents"]
        print(f"\n{'='*70}\nS{sc} {SCENARIO_CONFIGS[sc]['name']}  (n={n_agents})\n{'='*70}")

        baseline = run_condition(args.model_dir, args.model_n, sc, n_agents,
                                  args.episodes, args.seed, mask_idx=None)
        print(f"baseline           : {baseline.mean():.3f} ± {baseline.std():.3f}")

        summary[sc] = {"baseline": (baseline.mean(), baseline.std())}
        for gname, idxs in GROUPS.items():
            rates = run_condition(args.model_dir, args.model_n, sc, n_agents,
                                   args.episodes, args.seed, mask_idx=idxs)
            drop = baseline.mean() - rates.mean()
            summary[sc][gname] = (rates.mean(), rates.std(), drop)
            names = ",".join(FEATURE_NAMES[i].split("_")[0] for i in idxs)
            print(f"{gname:10s}({names:<28s}) masked: {rates.mean():.3f} ± {rates.std():.3f}  "
                  f"(drop vs baseline: {drop:+.3f})")

    print(f"\n{'='*70}\n전체 요약 (baseline 대비 낙폭, %p)\n{'='*70}")
    header = f"{'시나리오':<10}" + "".join(f"{g:>12}" for g in GROUPS)
    print(header)
    for sc in args.scenarios:
        row = f"S{sc:<9}"
        for g in GROUPS:
            drop = summary[sc][g][2] if g in summary[sc] else float("nan")
            row += f"{drop*100:>11.1f}%"
        print(row)
