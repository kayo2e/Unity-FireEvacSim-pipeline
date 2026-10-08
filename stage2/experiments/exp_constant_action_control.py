"""
exp_constant_action_control.py — 상수 행동 대조 실험
====================================================
제안 정책의 생존율이 "학습된 적응"에서 오는지 확인하기 위해, 같은 환경
(hazard_aware 기본값, 시드 base_seed+ep 페어링)에서 행동을 매 스텝 같은
상수로 고정해 돌린다.

  mid      : action_net bias 초기화값 (27.5, 27.5, 2.75)
  ppo_mean : 제안 정책이 해당 시나리오에서 낸 행동의 평균
             (experiments/exp_action_timeseries.py 출력에서 계산)

결과 CSV는 exp1_compare.py와 같은 열 구성이라 paired_diff_ci.py로 바로
PPO와 대응 비교할 수 있다.

실행:
    cd stage2
    python experiments/exp_action_timeseries.py --scenarios 2 3 4 5   # ppo_mean용
    python experiments/exp_constant_action_control.py --scenarios 2 3 4 5
"""

import sys, os, csv, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from exp1_compare import _make_rec

STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(STAGE2, "result", "constant_action_control")
TIMESERIES_DIR = os.path.join(STAGE2, "result", "action_timeseries")

MID_ACTION = [27.5, 27.5, 2.75]


def ppo_mean_action(scenario):
    path = os.path.join(TIMESERIES_DIR, f"action_timeseries_s{scenario}.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return [float(np.mean([float(r[k]) for r in rows])) for k in ("cA", "cB", "wc")]


def run_constant(scenario, action, n_episodes, base_seed, label):
    cfg = SCENARIO_CONFIGS[scenario]
    n_agents = cfg["n_agents"]
    action = np.array(action, dtype=np.float32)
    records = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        env.reset(seed=seed)
        total_r, info = 0.0, {}
        for _ in range(cfg["max_steps"]):
            _, r, term, trunc, info = env.step(action)
            total_r += r
            if term or trunc:
                break
        env.close()
        records.append({"model": label,
                        **_make_rec(ep + 1, scenario, n_agents, info, total_r, seed)})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    for sc in args.scenarios:
        for label, action in [("const_mid", MID_ACTION), ("const_ppo_mean", ppo_mean_action(sc))]:
            recs = run_constant(sc, action, args.episodes, args.seed, label)
            out = os.path.join(RESULT_DIR, f"{label}_s{sc}.csv")
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
            sr = 100 * np.mean([r["survival_rate"] for r in recs])
            print(f"S{sc} {label:15s} action=({action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}) "
                  f"생존율 {sr:.1f}% | 저장: {out}")


if __name__ == "__main__":
    main()
