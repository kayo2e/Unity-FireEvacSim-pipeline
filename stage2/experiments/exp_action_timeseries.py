"""
exp_action_timeseries.py — 예시 에피소드의 cA/cB/wc 시계열 기록
==================================================================
제안 정책이 매 스텝 출력하는 행동(cA=EXIT A 비용, cB=EXIT B 비용,
wc=혼잡 가중치)을 위협 관측(F1/F2)·출구별 누적 탈출 인원과 함께 기록한다.
"정책이 화재 상황에 따라 실제로 행동을 바꾸는가"를 보여주기 위한 자료.

평가 루프는 exp1_compare.py의 run_ppo()와 동일하다(같은 모델 탐색,
VecNormalize, 시드 base_seed+ep, deterministic, 스무딩 없음). 그래서
에피소드별 생존율이 표 3 CSV와 일치해야 하며, 스크립트가 직접 대조한다.

실행:
    cd stage2
    python experiments/exp_action_timeseries.py --scenarios 2 3 4 5 --feature-set reduced
"""

import sys, os, csv, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from exp1_compare import _find_model

STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(STAGE2, "result", "action_timeseries")
EXP1_DIR = os.path.join(STAGE2, "result", "exp1_compare")

FIELDS = ["episode", "seed", "step", "cA", "cB", "wc", "F1_exitA_threat", "F2_exitB_threat",
          "escaped_A", "escaped_B", "dead", "remaining", "fire_cells"]


def run_episode(model, vecnorm_path, scenario, n_agents, seed, feature_set):
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents, feature_set=feature_set)
    vec = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vec = VecNormalize.load(vecnorm_path, vec)
        vec.training = False
        vec.norm_reward = False
    vec.seed(seed)
    obs = vec.reset()

    rows, info = [], {}
    for _ in range(SCENARIO_CONFIGS[scenario]["max_steps"]):
        # 행동을 고른 시점의 관측(정규화 전 원본)을 행동과 같은 행에 둔다
        raw = vec.get_original_obs()[0] if isinstance(vec, VecNormalize) else obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec.step(action)
        info = infos[0]
        a = action[0]
        rows.append({"seed": seed, "step": info["step"],
                     "cA": float(a[0]), "cB": float(a[1]), "wc": float(a[2]),
                     "F1_exitA_threat": float(raw[0]), "F2_exitB_threat": float(raw[1]),
                     "escaped_A": info["escaped_A"], "escaped_B": info["escaped_B"],
                     "dead": info["dead"], "remaining": info["remaining"],
                     "fire_cells": info["fire_cells"]})
        if done[0]:
            break
    vec.close()
    return rows, info["survival_rate"]


def load_exp1_ppo(scenario, n_agents):
    """표 3 대조용: 해당 시나리오의 가장 최근 시드 페어링 PPO 결과(seed → 생존율)."""
    for path in sorted(glob.glob(os.path.join(EXP1_DIR, f"exp1_s{scenario}_s{scenario}_*.csv")),
                       reverse=True):
        recs = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                if r["model"] == "ppo" and "seed" in r and int(r["n_agents"]) == n_agents]
        if len(recs) >= 30:
            return path, {int(r["seed"]): float(r["survival_rate"]) for r in recs}
    return None, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", choices=["full", "reduced"], default="reduced")
    parser.add_argument("--model-dir", default=os.path.join(STAGE2, "model", "ppo"))
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    for sc in args.scenarios:
        n_agents = SCENARIO_CONFIGS[sc]["n_agents"]
        _, model_path, vecnorm_path = _find_model(args.model_dir, n_agents, args.feature_set)
        model = PPO.load(model_path)
        ref_path, ref = load_exp1_ppo(sc, n_agents)

        all_rows, surv = [], {}
        for ep in range(args.episodes):
            seed = args.seed + ep
            rows, sr = run_episode(model, vecnorm_path, sc, n_agents, seed, args.feature_set)
            for r in rows:
                r["episode"] = ep + 1
            all_rows += rows
            surv[seed] = sr

        out = os.path.join(RESULT_DIR, f"action_timeseries_s{sc}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)

        mismatch = [s for s in surv if s in ref and abs(surv[s] - ref[s]) > 1e-9]
        print(f"S{sc}: 생존율 {100*np.mean(list(surv.values())):.1f}% | "
              f"표 3 대조 {os.path.basename(ref_path) if ref_path else '없음'}: "
              f"{len(surv) - len(mismatch)}/{len(surv)} 일치 | 저장: {out}")
        if mismatch:
            print(f"  [경고] 불일치 시드: {mismatch}")


if __name__ == "__main__":
    main()
