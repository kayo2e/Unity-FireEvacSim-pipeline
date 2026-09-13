"""
action_sensitivity_check.py — ALL_BLIND 마스킹이 생존율을 거의 안 바꾼 이유 진단
================================================================================
feature_group_ablation.py에서 F1~F15 전부를 마스킹해도(ALL_BLIND) 생존율이
거의 그대로였다. 두 가설 중 무엇이 맞는지 구분한다.

  가설 A: 정책이 학습 후 입력에 거의 무관하게 비슷한 행동(exit_A_cost,
          exit_B_cost, crowd_weight)을 낸다 (정책이 관측을 사실상 안 씀)
  가설 B: 행동 자체는 관측에 따라 실제로 크게 변하지만, 그 변화가 Dijkstra
          비용장 → 최종 이동방향 단계에서 대부분 흡수돼 생존율까지는 잘 안
          이어진다 (환경 메커니즘이 정책 출력에 안정적/robust)

feature_correlation.py가 저장한 obs_raw_samples.npy(S1~S5 실제 상태 7808개)를
재사용해, (1) 원본 관측 그대로 넣었을 때 행동의 분산과 (2) 전부 0으로 마스킹한
관측을 넣었을 때 행동을 비교한다.

실행:
    cd stage2
    python experiments/action_sensitivity_check.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_core import FireEvacEnv

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "result", "feature_analysis")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "ppo",
                           "fire_evac_model_40ppl")

if __name__ == "__main__":
    X = np.load(os.path.join(RESULT_DIR, "obs_raw_samples.npy"))  # (7808, 15) raw obs
    print(f"샘플 {X.shape[0]}개 로드")

    model = PPO.load(MODEL_PATH)
    vecnorm_path = MODEL_PATH + "_vecnorm.pkl"
    dummy = DummyVecEnv([lambda: FireEvacEnv(scenario=1, n_agents=40)])
    vnorm = VecNormalize.load(vecnorm_path, dummy)
    vnorm.training = False
    vnorm.norm_reward = False

    # 정규화 (raw -> VecNormalize 통계 기준)
    norm_X = vnorm.normalize_obs(X)

    # (1) 원본 관측 그대로
    actions_real, _ = model.predict(norm_X, deterministic=True)

    # (2) 전부 0으로 마스킹 (= 학습 시 평균으로 고정)
    norm_blind = np.zeros_like(norm_X)
    actions_blind, _ = model.predict(norm_blind, deterministic=True)

    names = ["exit_A_cost", "exit_B_cost", "crowd_weight"]
    print("\n=== (1) 실제 관측 7808개에 대한 행동 분포 ===")
    for i, n in enumerate(names):
        col = actions_real[:, i]
        print(f"  {n:14s} mean={col.mean():7.3f}  std={col.std():7.3f}  "
              f"min={col.min():7.3f}  max={col.max():7.3f}")

    print("\n=== (2) 전부 블라인드(0) 처리했을 때 나오는 행동 (모든 샘플 동일해야 정상) ===")
    for i, n in enumerate(names):
        col = actions_blind[:, i]
        print(f"  {n:14s} value={col[0]:7.3f}  (분산={col.std():.6f}, 전부 같아야 함)")

    print("\n=== (1) vs (2): 블라인드 행동이 실제 행동 분포 중 어디쯤에 있는가 ===")
    blind_action = actions_blind[0]
    for i, n in enumerate(names):
        col = actions_real[:, i]
        pct = (col < blind_action[i]).mean() * 100
        print(f"  {n:14s} blind값={blind_action[i]:7.3f}  "
              f"-> 실제 분포에서 하위 {pct:.1f}백분위")

    # 가설 A/B 판정: 실제 관측에서 행동의 표준편차가 액션 range 대비 얼마나 작은지
    print("\n=== 판정 ===")
    ranges = {"exit_A_cost": (5, 50), "exit_B_cost": (5, 50), "crowd_weight": (0.5, 5.0)}
    for i, n in enumerate(names):
        lo, hi = ranges[n]
        col = actions_real[:, i]
        rel_std = col.std() / (hi - lo)
        verdict = "거의 고정값 -> 가설 A 쪽" if rel_std < 0.03 else "관측 따라 실제로 변함 -> 가설 B 쪽 우세"
        print(f"  {n:14s} std/range={rel_std:.3f}  ({verdict})")
