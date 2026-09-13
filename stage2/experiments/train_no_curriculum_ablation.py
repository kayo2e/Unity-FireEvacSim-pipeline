"""
train_no_curriculum_ablation.py — 커리큘럼(시나리오식 학습) 효과 ablation
================================================================================
"시나리오식 학습(S1→S4 커리큘럼)이 실제로 효과가 있는가?"를 확인하기 위한
대조군 학습 스크립트. `ppo_train.py`의 학습 로직(ent_coef, action_net bias
초기화 포함, 2026-09-13 수정판)을 그대로 재사용하되, 커리큘럼 진급 없이
처음부터 가장 어려운 시나리오(S4, 양방향 동시 위협)에서만 바로 학습한다.

비교 대상: model/ppo/fire_evac_model_40ppl.zip (S1→S4 커리큘럼으로 학습)
이 스크립트 결과물: model/ppo_no_curriculum/fire_evac_model_40ppl.zip
                  (S4 직행, 커리큘럼 없음)

공정한 비교를 위해 총 학습 스텝 수를 커리큘럼 버전과 동일하게 맞춰야 한다
(스텝 수가 다르면 "커리큘럼 효과"와 "학습량 차이"가 섞여 결론이 무효화된다
— docs/feature-space-optimization-plan.md Task 2 계획 4번 참고).

평가는 학습 후 experiments/exp1_compare.py --model-dir model/ppo_no_curriculum
로 커리큘럼 버전과 동일 시드(seed=42)·동일 episodes로 S1~S5 생존율을 비교한다.

실행:
    cd stage2
    python experiments/train_no_curriculum_ablation.py --steps 3500000 --n-envs 8
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from train_common import BASE_DIR, EvacTrainCallback, make_vec_env, find_latest_checkpoint

# 커리큘럼 버전과 완전히 분리된 저장 경로 — 기존 model/ppo/를 절대 덮어쓰지 않는다
MODEL_DIR = os.path.join(BASE_DIR, "model", "ppo_no_curriculum")
LOG_DIR   = os.path.join(BASE_DIR, "logs",  "ppo_no_curriculum")

NO_CURRICULUM_SCENARIO = 4  # 양방향 동시 위협 — 커리큘럼 최종 목표 시나리오와 동일


def train(total_timesteps=3_500_000, n_envs=8, ent_coef=0.005):
    n_agents = SCENARIO_CONFIGS[NO_CURRICULUM_SCENARIO]["n_agents"]
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 62)
    print("커리큘럼 ablation — S4 직행 학습 (커리큘럼 없음)")
    print(f"시나리오  : S{NO_CURRICULUM_SCENARIO} ({SCENARIO_CONFIGS[NO_CURRICULUM_SCENARIO]['name']})")
    print(f"인원수    : {n_agents}명")
    print(f"총 스텝   : {total_timesteps:,}  (커리큘럼 버전과 동일해야 공정 비교)")
    print(f"병렬 환경 : {n_envs}개")
    print(f"ent_coef  : {ent_coef}  (커리큘럼 버전과 동일 — 2026-09-13 수정판)")
    print(f"모델 저장 : {MODEL_DIR}")
    print("=" * 62)

    ckpt_dir    = os.path.join(MODEL_DIR, "checkpoints")
    ckpt_prefix = f"ckpt_{n_agents}ppl"
    ckpt_path, vnorm_ckpt, ckpt_steps = find_latest_checkpoint(ckpt_dir, ckpt_prefix)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # start_scenario=max_scenario=NO_CURRICULUM_SCENARIO -> 승급 조건이 항상 거짓이라
    # 커리큘럼 진급 없이 처음부터 끝까지 S4에서만 학습된다 (train_common.py
    # EvacCurriculumWrapper 주석 참고)
    if ckpt_path and ckpt_steps < total_timesteps:
        print(f"\n체크포인트 이어서 학습 ({ckpt_steps:,} → {total_timesteps:,} steps)")
        vec_env = make_vec_env(n_envs=n_envs, n_agents=n_agents,
                                max_scenario=NO_CURRICULUM_SCENARIO,
                                start_scenario=NO_CURRICULUM_SCENARIO)
        if os.path.exists(vnorm_ckpt):
            vec_env = VecNormalize.load(vnorm_ckpt, vec_env.venv)
            vec_env.training = True
        model = PPO.load(ckpt_path, env=vec_env, device=device, tensorboard_log=LOG_DIR)
        remaining = total_timesteps - ckpt_steps
        reset_num = False
    else:
        print("\nS4 직행 학습 시작 (커리큘럼 없음)")
        vec_env = make_vec_env(n_envs=n_envs, n_agents=n_agents,
                                max_scenario=NO_CURRICULUM_SCENARIO,
                                start_scenario=NO_CURRICULUM_SCENARIO)
        model = PPO(
            "MlpPolicy", vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 2048,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            learning_rate   = 3e-4,
            clip_range      = 0.2,
            ent_coef        = ent_coef,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(net_arch=[256, 256]),
            tensorboard_log = LOG_DIR,
        )
        # ppo_train.py와 동일한 action_net bias 초기화 (2026-09-13 수정판과
        # 동일 조건으로 맞춰야 "커리큘럼 유무"만 분리해서 비교할 수 있다)
        action_low  = vec_env.action_space.low
        action_high = vec_env.action_space.high
        box_mid = (action_low + action_high) / 2.0
        with torch.no_grad():
            model.policy.action_net.bias.copy_(
                torch.as_tensor(box_mid, dtype=model.policy.action_net.bias.dtype))
        print(f"  action_net bias 초기화 -> 박스 중간값 {box_mid}")
        remaining = total_timesteps
        reset_num = True

    ckpt_cb = CheckpointCallback(
        save_freq         = max(500_000 // n_envs, 1),
        save_path         = ckpt_dir,
        name_prefix       = ckpt_prefix,
        save_vecnormalize = True,
        verbose           = 1,
    )
    callback = CallbackList([EvacTrainCallback(log_interval=10_000), ckpt_cb])

    model.learn(
        total_timesteps     = remaining,
        callback            = callback,
        tb_log_name         = f"PPO_{n_agents}ppl_nocurriculum",
        progress_bar        = True,
        reset_num_timesteps = reset_num,
    )

    save_path = os.path.join(MODEL_DIR, f"fire_evac_model_{n_agents}ppl")
    model.save(save_path)
    vec_env.save(save_path + "_vecnorm.pkl")
    vec_env.close()
    print(f"\n모델 저장: {save_path}.zip")
    print("\n평가 실행 예시:")
    print(f"  python experiments/exp1_compare.py --scenarios 1 2 3 4 5 "
          f"--model-dir {MODEL_DIR} --include-hazard-astar --episodes 30 --seed 42")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="커리큘럼 ablation: S4 직행 학습")
    parser.add_argument("--steps",    type=int, default=3_500_000,
                         help="커리큘럼 버전과 반드시 동일하게 맞출 것")
    parser.add_argument("--n-envs",   type=int, default=8)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    args = parser.parse_args()

    train(total_timesteps=args.steps, n_envs=args.n_envs, ent_coef=args.ent_coef)
