"""
ppo_train.py — 일반 PPO 학습/테스트
======================================
알고리즘 : PPO (MlpPolicy, net_arch=[256,256])
모델 저장 : model/ppo/
결과 저장 : result/ppo/
로그      : logs/ppo/

Ablation 비교용: recurrent_ppo_train.py와 동일 조건으로 학습 후 성능 비교.

실행:
    python ppo_train.py --mode train --people 10 --steps 300000 --bc-steps 0
    python ppo_train.py --mode test  --model-n 10 --all-scenarios --test-episodes 30
"""

import sys
import os
import platform

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env_core import FireEvacEnv, SCENARIO_CONFIGS, verify_connectivity
from train_common import (BASE_DIR, EvacTrainCallback, make_vec_env,
                          collect_astar_demos, pretrain_bc, test_fire_evac)

MODEL_DIR  = os.path.join(BASE_DIR, "model",  "ppo")
RESULT_DIR = os.path.join(BASE_DIR, "result", "ppo")
LOG_DIR    = os.path.join(BASE_DIR, "logs",   "ppo")


# ══════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════
def train(person_counts=(10,), total_timesteps=300_000,
          n_envs=None, bc_demo_steps=3000, bc_s4_steps=2000, bc_epochs=10):
    import torch

    n_cpu = __import__('multiprocessing').cpu_count()
    if n_envs is None:
        n_envs = max(4, min(n_cpu, 16))

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = FireEvacEnv(scenario=1).observation_space.shape[0]
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 62)
    print("화재대피유도시스템 — PPO 학습")
    print(f"인원수    : {person_counts}명")
    print(f"총 스텝   : {total_timesteps:,} / 모델")
    print(f"병렬 환경 : {n_envs}개")
    print(f"관측 차원 : {obs_dim} (F1~F15)")
    print(f"Policy    : MlpPolicy | net_arch=[256,256]")
    print(f"디바이스  : {device}")
    print(f"모델 저장 : {MODEL_DIR}")
    print("=" * 62)

    for n in person_counts:
        print(f"\n{'─'*62}\n인원수 {n}명 | 커리큘럼 학습 시작\n{'─'*62}")

        vec_env  = make_vec_env(n_agents=n, n_envs=n_envs)
        callback = EvacTrainCallback(log_interval=10_000)

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
            ent_coef        = 0.05,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(net_arch=[256, 256]),
            tensorboard_log = LOG_DIR,
        )

        if bc_demo_steps > 0:
            print(f"\n[BC 사전학습] 데모 수집 중...")
            obs_demo, act_demo = collect_astar_demos(
                n_agents=n, n_envs_demo=4,
                n_steps=bc_demo_steps, s4_steps=bc_s4_steps)
            pretrain_bc(model, obs_demo, act_demo, n_epochs=bc_epochs)

        model.learn(
            total_timesteps = total_timesteps,
            callback        = callback,
            tb_log_name     = f"PPO_{n}ppl",
            progress_bar    = True,
        )

        save_path = os.path.join(MODEL_DIR, f"fire_evac_model_{n}ppl")
        model.save(save_path)
        vec_env.save(save_path + "_vecnorm.pkl")
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip")

    print("\n" + "=" * 62)
    print(f"학습 완료! TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("=" * 62)


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO 화재대피 학습/테스트")
    parser.add_argument("--mode",          choices=["train", "test", "check"], default="train")
    parser.add_argument("--people",        type=int, nargs="+", default=[10])
    parser.add_argument("--steps",         type=int, default=300_000)
    parser.add_argument("--n-envs",        type=int, default=None)
    parser.add_argument("--test-n",        type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--model-n",       type=int, default=None,
                        help="로드할 모델 인원수 (미지정 시 --test-n 사용)")
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--bc-steps",      type=int, default=3000,
                        help="BC 데모 수집 스텝 (0=비활성화)")
    parser.add_argument("--bc-s4-steps",   type=int, default=2000)
    parser.add_argument("--bc-epochs",     type=int, default=10)
    args = parser.parse_args()

    if args.mode == "check":
        verify_connectivity()
        env = FireEvacEnv(scenario=1, n_agents=10)
        check_env(env)
        print(f"관측 크기: {env.observation_space.shape} | 유도등: {env.n_lights}개")
        print("환경 검증 완료!")

    elif args.mode == "train":
        train(
            person_counts  = args.people,
            total_timesteps= args.steps,
            n_envs         = args.n_envs,
            bc_demo_steps  = args.bc_steps,
            bc_s4_steps    = args.bc_s4_steps,
            bc_epochs      = args.bc_epochs,
        )

    elif args.mode == "test":
        scenarios = [1, 2, 3, 4] if args.all_scenarios else [args.test_scenario]
        all_summaries = {}
        for sc in scenarios:
            n = SCENARIO_CONFIGS[sc]["n_agents"] if args.all_scenarios else args.test_n
            _, summary = test_fire_evac(
                ModelCls     = PPO,
                model_dir    = MODEL_DIR,
                result_dir   = RESULT_DIR,
                n_agents     = n,
                scenario     = sc,
                n_episodes   = args.test_episodes,
                save_results = not args.no_save,
                render       = args.render,
                model_n      = args.model_n,
            )
            all_summaries[sc] = summary

        if args.all_scenarios:
            print(f"\n{'═'*62}")
            print("  전체 시나리오 생존율 (PPO)")
            print(f"{'─'*62}")
            for sc, s in all_summaries.items():
                sr = s["survival_rate"]
                print(f"  S{sc} {s['scenario_name']:<8} | "
                      f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                      f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
            print(f"{'═'*62}")
