"""
paired_diff_ci.py — PPO와 비교 방법 간 시드 대응 차이의 95% 신뢰구간
=====================================================================
에피소드 i(같은 시드)마다 dᵢ = PPO 생존율 − 비교 방법 생존율(%p)을 구하고,
평균 차이와 t 분포 기반 95% CI를 보고한다. dᵢ가 정규분포에서 벗어나는
시나리오가 있어 부트스트랩 CI(10,000회)와 Wilcoxon 부호순위 검정을 함께
낸다. 재실험 없이 기존 CSV만 읽는다.

입력:
  - 표 3: result/exp1_compare/ 의 지정 타임스탬프 실행분
          (PPO, Hazard-aware BFS, A*, 정적 유도등)
  - 상수 행동 대조: result/constant_action_control/
          (exp_constant_action_control.py 출력, 있으면 포함)

실행:
    cd stage2
    python experiments/paired_diff_ci.py
"""

import os, csv, glob

import numpy as np
from scipy import stats

STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP1_DIR = os.path.join(STAGE2, "result", "exp1_compare")
CONST_DIR = os.path.join(STAGE2, "result", "constant_action_control")
OUT_PATH = os.path.join(STAGE2, "result", "paired_diff_ci.csv")

# 표 3에 들어간 실행분 (README 표 3 수치와 일치 확인)
TABLE3_RUNS = {2: "20260926_185314", 3: "20260926_190256",
               4: "20260926_191240", 5: "20260926_192430"}


def load(path, model):
    return {int(r["seed"]): 100 * float(r["survival_rate"])
            for r in csv.DictReader(open(path, encoding="utf-8")) if r["model"] == model}


def main():
    rng = np.random.default_rng(0)
    out = []
    for sc, ts in TABLE3_RUNS.items():
        ppo = load(os.path.join(EXP1_DIR, f"exp1_s{sc}_s{sc}_{ts}.csv"), "ppo")
        baselines = [
            ("Hazard-aware BFS", os.path.join(EXP1_DIR, f"exp1_hazard_astar_s{sc}_{ts}.csv"), "hazard_astar"),
            ("A*", os.path.join(EXP1_DIR, f"exp1_simple_astar_s{sc}_{ts}.csv"), "simple_astar"),
            ("정적 유도등", os.path.join(EXP1_DIR, f"exp1_static_s{sc}_{ts}.csv"), "static"),
            ("상수 행동(초기화값)", os.path.join(CONST_DIR, f"const_mid_s{sc}.csv"), "const_mid"),
            ("상수 행동(PPO 평균)", os.path.join(CONST_DIR, f"const_ppo_mean_s{sc}.csv"), "const_ppo_mean"),
        ]
        for name, path, model in baselines:
            if not os.path.exists(path):
                continue
            base = load(path, model)
            seeds = sorted(ppo)
            assert seeds == sorted(base), f"S{sc} {name}: 시드 불일치"
            d = np.array([ppo[s] - base[s] for s in seeds])
            n, m = len(d), d.mean()
            h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            boot = rng.choice(d, (10000, n)).mean(axis=1)
            row = {
                "scenario": sc, "baseline": name, "n": n,
                "ppo_mean": round(np.mean(list(ppo.values())), 2),
                "baseline_mean": round(np.mean(list(base.values())), 2),
                "diff_mean": round(m, 2),
                "t_ci_low": round(m - h, 2), "t_ci_high": round(m + h, 2),
                "boot_ci_low": round(np.percentile(boot, 2.5), 2),
                "boot_ci_high": round(np.percentile(boot, 97.5), 2),
                "p_ttest": round(stats.ttest_1samp(d, 0).pvalue, 4),
                "p_wilcoxon": round(stats.wilcoxon(d).pvalue, 4) if d.any() else 1.0,
                "win": int((d > 0).sum()), "tie": int((d == 0).sum()), "loss": int((d < 0).sum()),
            }
            out.append(row)
            print(f"S{sc} PPO − {name:14s} {m:+6.2f} [{m - h:+6.2f}, {m + h:+6.2f}] "
                  f"p={row['p_ttest']:.4f} W/T/L={row['win']}/{row['tie']}/{row['loss']}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
