"""
recurrent_ppo_train.py — RecurrentPPO 학습/테스트
===================================================
알고리즘 : RecurrentPPO (MlpLstmPolicy, lstm_hidden_size=256)
모델 저장 : model/recurrent_ppo/
결과 저장 : result/recurrent_ppo/
로그      : logs/recurrent_ppo/

설치: pip install sb3-contrib

Ablation 비교용: ppo_train.py와 동일 조건으로 학습 후 성능 비교.

실행:
    python recurrent_ppo_train.py --mode train --auto-scenarios --steps 3000000
    python recurrent_ppo_train.py --mode train --people 20 40   --steps 3000000
    python recurrent_ppo_train.py --mode test  --all-scenarios  --test-episodes 30
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise SystemExit(
        "sb3_contrib 미설치.\n"
        "설치: pip install sb3-contrib\n"
        "일반 PPO를 사용하려면: python ppo_train.py"
    )

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from env_core import FireEvacEnv, SCENARIO_CONFIGS, verify_connectivity
from train_common import (BASE_DIR, EvacTrainCallback, EarlyStoppingCallback,
                          make_vec_env, collect_astar_demos, pretrain_bc,
                          test_fire_evac, find_latest_checkpoint)

MODEL_DIR  = os.path.join(BASE_DIR, "model",  "recurrent_ppo")
RESULT_DIR = os.path.join(BASE_DIR, "result", "recurrent_ppo")
LOG_DIR    = os.path.join(BASE_DIR, "logs",   "recurrent_ppo")


# ══════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════
def train(total_timesteps=5_000_000,
          n_envs=None, bc_demo_steps=0, bc_s4_steps=2000, bc_epochs=10,
          lstm_hidden_size=256,
          early_stop=True, patience=500_000, min_delta=0.005):
    import torch

    # S1 인원으로 시작 → 커리큘럼 래퍼가 시나리오별 n_agents 자동 전환
    start_n = SCENARIO_CONFIGS[1]["n_agents"]

    n_cpu = __import__('multiprocessing').cpu_count()
    if n_envs is None:
        n_envs = max(4, min(n_cpu, 16))

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dim = FireEvacEnv(scenario=1).observation_space.shape[0]
    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_dir    = os.path.join(MODEL_DIR, "checkpoints")
    ckpt_prefix = "ckpt"
    ckpt_path, vnorm_ckpt, ckpt_steps = find_latest_checkpoint(ckpt_dir, ckpt_prefix)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 62)
    print("화재대피유도시스템 — RecurrentPPO 학습 (커리큘럼)")
    print(f"시작 인원 : {start_n}명 (S1) → 커리큘럼 자동 전환")
    print(f"총 스텝   : {total_timesteps:,}")
    print(f"병렬 환경 : {n_envs}개")
    print(f"관측 차원 : {obs_dim} (F1~F15)")
    print(f"Policy    : MlpLstmPolicy | lstm_hidden={lstm_hidden_size} | net_arch=[256,256]")
    print(f"디바이스  : {device}")
    print(f"모델 저장 : {MODEL_DIR}/fire_evac_model.zip")
    print("=" * 62)

    if ckpt_path and ckpt_steps < total_timesteps:
        print(f"\n{'─'*62}\n체크포인트 이어서 학습 "
              f"({ckpt_steps:,} → {total_timesteps:,} steps)\n{'─'*62}")
        vec_env = make_vec_env(n_envs=n_envs, n_agents=start_n)
        if os.path.exists(vnorm_ckpt):
            vec_env = VecNormalize.load(vnorm_ckpt, vec_env.venv)
            vec_env.training = True
        model = RecurrentPPO.load(ckpt_path, env=vec_env, device=device,
                                  tensorboard_log=LOG_DIR)
        remaining = total_timesteps - ckpt_steps
        reset_num = False
    else:
        print(f"\n{'─'*62}\n커리큘럼 학습 시작 (S1→S4 자동 전환)\n{'─'*62}")
        vec_env = make_vec_env(n_envs=n_envs, n_agents=start_n)
        model = RecurrentPPO(
            "MlpLstmPolicy", vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 2048,
            batch_size      = 256,
            n_epochs        = 5,
            gamma           = 0.99,
            learning_rate   = 3e-4,
            clip_range      = 0.2,
            ent_coef        = 0.05,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(lstm_hidden_size=lstm_hidden_size,
                                   net_arch=[256, 256]),
            tensorboard_log = LOG_DIR,
        )
        remaining = total_timesteps
        reset_num = True

    if bc_demo_steps > 0 and reset_num:
        print(f"\n[BC 사전학습] 데모 수집 중...")
        obs_demo, act_demo = collect_astar_demos(
            n_agents=start_n, n_envs_demo=4,
            n_steps=bc_demo_steps, s4_steps=bc_s4_steps)
        pretrain_bc(model, obs_demo, act_demo, n_epochs=bc_epochs)

    save_path = os.path.join(MODEL_DIR, "fire_evac_model")
    best_path = os.path.join(MODEL_DIR, "fire_evac_model_best")

    ckpt_cb = CheckpointCallback(
        save_freq         = max(500_000 // n_envs, 1),
        save_path         = ckpt_dir,
        name_prefix       = ckpt_prefix,
        save_vecnormalize = True,
        verbose           = 1,
    )

    callbacks = [EvacTrainCallback(log_interval=10_000), ckpt_cb]

    if early_stop:
        callbacks.append(EarlyStoppingCallback(
            patience       = patience,
            min_delta      = min_delta,
            window         = 100,
            min_steps      = 1_000_000,
            save_best_path = best_path,
        ))
        print(f"  EarlyStopping: patience={patience:,} | "
              f"min_delta={min_delta} | best 저장→ {best_path}.zip")

    model.learn(
        total_timesteps     = remaining,
        callback            = CallbackList(callbacks),
        tb_log_name         = "RecurrentPPO",
        progress_bar        = True,
        reset_num_timesteps = reset_num,
    )

    model.save(save_path)
    vec_env.save(save_path + "_vecnorm.pkl")
    vec_env.close()

    print("\n" + "=" * 62)
    print(f"모델 저장: {save_path}.zip")
    print(f"학습 완료! TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("=" * 62)


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RecurrentPPO 화재대피 학습/테스트")
    parser.add_argument("--mode",           choices=["train", "test", "check"], default="train")
    parser.add_argument("--steps",          type=int, default=5_000_000)
    parser.add_argument("--n-envs",         type=int, default=16)
    parser.add_argument("--test-scenario",  type=int, default=1)
    parser.add_argument("--test-episodes",  type=int, default=30)
    parser.add_argument("--all-scenarios",  action="store_true")
    parser.add_argument("--no-save",        action="store_true")
    parser.add_argument("--render",         action="store_true")
    parser.add_argument("--bc-steps",       type=int, default=0)
    parser.add_argument("--bc-s4-steps",    type=int, default=2000)
    parser.add_argument("--bc-epochs",      type=int, default=10)
    parser.add_argument("--lstm-hidden",    type=int, default=256)
    parser.add_argument("--no-early-stop",  action="store_true")
    parser.add_argument("--patience",       type=int, default=500_000)
    parser.add_argument("--min-delta",      type=float, default=0.005)
    args = parser.parse_args()

    if args.mode == "check":
        verify_connectivity()
        env = FireEvacEnv(scenario=1, n_agents=SCENARIO_CONFIGS[1]["n_agents"])
        check_env(env)
        print(f"관측 크기: {env.observation_space.shape} | 유도등: {env.n_lights}개")
        print("환경 검증 완료!")

    elif args.mode == "train":
        train(
            total_timesteps  = args.steps,
            n_envs           = args.n_envs,
            bc_demo_steps    = args.bc_steps,
            bc_s4_steps      = args.bc_s4_steps,
            bc_epochs        = args.bc_epochs,
            lstm_hidden_size = args.lstm_hidden,
            early_stop       = not args.no_early_stop,
            patience         = args.patience,
            min_delta        = args.min_delta,
        )

    elif args.mode == "test":
        scenarios = [1, 2, 3, 4] if args.all_scenarios else [args.test_scenario]
        all_summaries = {}
        for sc in scenarios:
            n = SCENARIO_CONFIGS[sc]["n_agents"]
            _, summary = test_fire_evac(
                ModelCls     = RecurrentPPO,
                model_dir    = MODEL_DIR,
                result_dir   = RESULT_DIR,
                n_agents     = n,
                scenario     = sc,
                n_episodes   = args.test_episodes,
                save_results = not args.no_save,
                render       = args.render,
            )
            all_summaries[sc] = summary

        if args.all_scenarios:
            print(f"\n{'═'*62}")
            print("  전체 시나리오 생존율 (RecurrentPPO)")
            print(f"{'─'*62}")
            for sc, s in all_summaries.items():
                sr = s["survival_rate"]
                print(f"  S{sc} {s['scenario_name']:<8} | "
                      f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                      f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
            print(f"{'═'*62}")
