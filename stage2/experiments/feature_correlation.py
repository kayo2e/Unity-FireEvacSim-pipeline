"""
feature_correlation.py — F1~F15 상관관계 분석 (Task 2, Phase 2)
================================================================================
docs/feature-space-optimization-plan.md의 Task 2 2번. 재학습 없이 기존 40ppl
PPO 모델로 S1~S5를 굴리면서 매 스텝 raw F1~F15 벡터를 로깅하고, 15x15 Pearson
상관계수 행렬을 계산한다. 계획 문서가 가설로 세운 중복 후보
(F1/F2 vs F14/F15, F3 vs F10/F11, F4 vs F5)를 실제로 검증한다.

주의: env._get_obs()가 반환하는 값은 raw(정규화 전) 스케일이라 VecNormalize와
무관하게 그대로 로깅한다. 상관관계는 스케일에 불변이므로 문제 없다.

실행:
    cd stage2
    python experiments/feature_correlation.py --episodes 20
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


def collect_obs(model_dir, model_n, scenario, n_agents, n_episodes, base_seed):
    model_path = os.path.join(model_dir, f"fire_evac_model_{model_n}ppl")
    vecnorm_path = model_path + "_vecnorm.pkl"
    model = PPO.load(model_path)
    cfg = SCENARIO_CONFIGS[scenario]

    all_raw = []
    for ep in range(n_episodes):
        env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
        vec = DummyVecEnv([lambda: env])
        if os.path.exists(vecnorm_path):
            vnorm = VecNormalize.load(vecnorm_path, vec)
            vnorm.training = False
            vnorm.norm_reward = False
        else:
            vnorm = None

        seed = base_seed + ep
        raw_obs, _ = env.reset(seed=seed)  # env.reset()이 반환하는 obs를 그대로 사용
        # (env._get_obs()를 별도로 다시 부르면 F14/F15의 prev_f1/prev_f2 상태가
        #  스텝당 두 번 갱신되어 델타가 항상 0으로 오염된다 — reset/step 반환값만 쓴다)
        if vnorm is not None:
            vnorm.seed(seed)
            vnorm.reset()

        for _ in range(cfg["max_steps"]):
            all_raw.append(raw_obs.copy())

            if vnorm is not None:
                norm_obs = vnorm.normalize_obs(np.array([raw_obs]))
                action, _ = model.predict(norm_obs, deterministic=True)
                action = action[0]
            else:
                action, _ = model.predict(raw_obs, deterministic=True)

            raw_obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                break
        env.close()

    return np.array(all_raw)  # (T, 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=os.path.join("model", "ppo"))
    parser.add_argument("--model-n", type=int, default=40)
    args = parser.parse_args()

    all_obs = []
    for sc in args.scenarios:
        n_agents = SCENARIO_CONFIGS[sc]["n_agents"]
        print(f"수집 중: S{sc} ({SCENARIO_CONFIGS[sc]['name']}), n={n_agents}, "
              f"{args.episodes}episodes ...")
        obs = collect_obs(args.model_dir, args.model_n, sc, n_agents,
                           args.episodes, args.seed)
        print(f"  -> {obs.shape[0]} steps 수집")
        all_obs.append(obs)

    X = np.concatenate(all_obs, axis=0)
    print(f"\n총 {X.shape[0]} 스텝, {X.shape[1]}차원")

    corr = np.corrcoef(X, rowvar=False)

    print("\n=== 15x15 상관계수 행렬 ===")
    header = "        " + "".join(f"F{i+1:>3}" for i in range(15))
    print(header)
    for i in range(15):
        row = "".join(f"{corr[i, j]:>4.1f}" for j in range(15))
        print(f"F{i+1:<3}    {row}")

    print("\n=== |r| >= 0.6 인 쌍 (자기 자신 제외) ===")
    pairs = []
    for i in range(15):
        for j in range(i + 1, 15):
            r = corr[i, j]
            if abs(r) >= 0.6:
                pairs.append((abs(r), i, j, r))
    pairs.sort(reverse=True)
    if not pairs:
        print("  (없음 — 15개 피처가 서로 뚜렷하게 구분되는 정보를 담고 있음)")
    for _, i, j, r in pairs:
        print(f"  {FEATURE_NAMES[i]:<22} <-> {FEATURE_NAMES[j]:<22}  r={r:+.3f}")

    # 계획 문서가 세운 가설 쌍 직접 확인
    print("\n=== 사전 가설 쌍 확인 ===")
    hyp_pairs = [(0, 13, "F1 vs F14"), (1, 14, "F2 vs F15"),
                 (2, 9, "F3 vs F10"), (2, 10, "F3 vs F11"),
                 (3, 4, "F4 vs F5")]
    for i, j, label in hyp_pairs:
        print(f"  {label:<12} r={corr[i, j]:+.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "result", "feature_analysis")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "obs_corr_matrix.npy"), corr)
    np.save(os.path.join(out_dir, "obs_raw_samples.npy"), X)
    print(f"\n저장: {out_dir}/obs_corr_matrix.npy, obs_raw_samples.npy")
