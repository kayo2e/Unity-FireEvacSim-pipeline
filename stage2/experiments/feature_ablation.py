"""
feature_ablation.py — F1~F15 permutation 중요도 측정 (Task 2, Phase 1)
================================================================================
docs/feature-space-optimization-plan.md의 Task 2 1번. 재학습 없이 기존 학습된
40ppl 모델을 그대로 쓰되, 평가 시 F1~F15 중 하나씩 정규화된 관측값을 0(=학습
시 평균)으로 고정하고 생존율이 baseline 대비 얼마나 떨어지는지 측정한다.
낙폭이 거의 없는 피처가 제거 후보다.

VecNormalize가 (raw - running_mean) / running_std로 정규화하므로, 정규화된
값을 0으로 고정하는 것 = 그 피처를 학습 시 평균값으로 고정하는 것과 같다.
raw 평균을 따로 계산할 필요가 없어 이 지점에서 마스킹한다.

실행:
    cd stage2
    python experiments/feature_ablation.py --scenario 4 --episodes 20
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_core import FireEvacEnv, SCENARIO_CONFIGS

FEATURE_NAMES = [
    "F1_exitA_threat", "F2_exitB_threat", "F3_exitA_pref", "F4_escaped_ratio",
    "F5_dead_ratio", "F6_time_ratio", "F7_exitA_congestion", "F8_exitB_congestion",
    "F9_panic", "F10_exitA_dist", "F11_exitB_dist", "F12_fire_row", "F13_fire_col",
    "F14_exitA_delta", "F15_exitB_delta",
]


def run_condition(model_dir, model_n, scenario, n_agents, n_episodes, base_seed, mask_idx=None):
    model_path = os.path.join(model_dir, f"fire_evac_model_{model_n}ppl")
    vecnorm_path = model_path + "_vecnorm.pkl"
    model = PPO.load(model_path)
    cfg = SCENARIO_CONFIGS[scenario]

    survival_rates = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        vec = DummyVecEnv([lambda: env])
        if os.path.exists(vecnorm_path):
            vec = VecNormalize.load(vecnorm_path, vec)
            vec.training = False
            vec.norm_reward = False

        seed = base_seed + ep
        vec.seed(seed)
        obs = vec.reset()
        if mask_idx is not None:
            obs[:, mask_idx] = 0.0
        info = {}

        for _ in range(cfg["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = vec.step(action)
            if mask_idx is not None:
                obs[:, mask_idx] = 0.0
            if infos[0]:
                info = infos[0]
            if done[0]:
                break
        vec.close()
        survival_rates.append(info.get("survival_rate", 0.0))

    return np.array(survival_rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=os.path.join("model", "ppo"))
    parser.add_argument("--model-n", type=int, default=40)
    args = parser.parse_args()

    n_agents = SCENARIO_CONFIGS[args.scenario]["n_agents"]

    print(f"=== Feature ablation: S{args.scenario} ({SCENARIO_CONFIGS[args.scenario]['name']}), "
          f"n_agents={n_agents}, episodes={args.episodes} ===\n")

    baseline = run_condition(args.model_dir, args.model_n, args.scenario, n_agents,
                              args.episodes, args.seed, mask_idx=None)
    print(f"baseline (no mask): {baseline.mean():.3f} ± {baseline.std():.3f}")

    results = {}
    for idx, name in enumerate(FEATURE_NAMES):
        rates = run_condition(args.model_dir, args.model_n, args.scenario, n_agents,
                               args.episodes, args.seed, mask_idx=idx)
        drop = baseline.mean() - rates.mean()
        results[name] = (rates.mean(), rates.std(), drop)
        print(f"{name:22s} masked: {rates.mean():.3f} ± {rates.std():.3f}  "
              f"(drop vs baseline: {drop:+.3f})")

    print("\n=== 요약: baseline 대비 낙폭이 큰 순 ===")
    for name, (mean, std, drop) in sorted(results.items(), key=lambda kv: -kv[1][2]):
        print(f"  {name:22s} drop={drop:+.3f}  masked={mean:.3f}")
