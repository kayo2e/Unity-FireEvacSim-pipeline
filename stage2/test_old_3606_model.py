"""
구버전 PPO 모델 (obs=3606) 테스트
===================================
모델: result/test_ppo_10ppl12/fire_evac_model_10ppl.zip
환경: 구버전 FireEvacEnv (6채널×30×20 + 피처6 = 3606차원)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from collections import deque
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ── 상수 ──────────────────────────────────────────────────────────────────
HALL, WALL, EXIT, ROOM = 0, 1, 2, 3
WALKABLE = {HALL, EXIT, ROOM}
N, S, E, W = 0, 1, 2, 3
DELTA = {N: (-1, 0), S: (1, 0), E: (0, 1), W: (0, -1)}

EXIT_CAPACITY = 1
CELL_CAPACITY = 1
QUEUE_RADIUS  = 4

DENSITY_RADIUS     = 1
DENSITY_SLOW_MAX   = 0.80
PANIC_FIRE_DIST    = 15.0
PANIC_RANDOM_MAX   = 0.30

SCENARIO_CONFIGS = {
    1: {"name":"초기 화재",  "fire_count":(1,1),  "spread_prob":0.05, "smoke_radius":0, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":[(3,1)],   "max_steps":200},
    2: {"name":"화재 확산",  "fire_count":(2,3),  "spread_prob":0.12, "smoke_radius":2, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":None,     "max_steps":250},
    3: {"name":"출구 위협",  "fire_count":(1,1),  "spread_prob":0.25, "smoke_radius":4, "exit_block_prob":0.0, "collapse_prob":0.0, "fire_fixed":[(8,10)], "max_steps":350},
    4: {"name":"폭발 붕괴",  "fire_count":(3,6),  "spread_prob":0.20, "smoke_radius":4, "exit_block_prob":0.2, "collapse_prob":0.15, "fire_fixed":None,     "max_steps":600},
}

# 구버전 그리드 (30×20)
BASE_GRID = None  # 아래에서 git에서 추출한 코드로 대체

# ── 구버전 그리드 (git 2bae77a) ────────────────────────────────────────
import importlib.util, types

_OLD_TRAIN_PATH = "/tmp/old_train.py"

def _load_old_env():
    """구버전 train.py에서 FireEvacEnv와 BASE_GRID만 임포트"""
    spec = importlib.util.spec_from_file_location("old_train", _OLD_TRAIN_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load_old_env()
OldFireEvacEnv   = _mod.FireEvacEnv
OldScenarioConfigs = _mod.SCENARIO_CONFIGS

# ── 테스트 실행 ───────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__),
               "result", "test_ppo_10ppl12", "fire_evac_model_10ppl")
VECNORM_PATH = MODEL_PATH + "_vecnorm.pkl"


def run_test(scenario: int, n_agents: int, n_episodes: int, seeds=None):
    env_fn = lambda: OldFireEvacEnv(scenario=scenario, n_agents=n_agents)
    vec    = DummyVecEnv([env_fn])

    model  = PPO.load(MODEL_PATH)

    if os.path.exists(VECNORM_PATH):
        vec = VecNormalize.load(VECNORM_PATH, vec)
        vec.training    = False
        vec.norm_reward = False

    surv_list = []
    for ep in range(n_episodes):
        seed = seeds[ep] if seeds else None
        obs  = vec.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec.step(action)
            done = dones[0]
            info = infos[0]
        surv_list.append(info["survival_rate"] * 100)

    vec.close()
    return float(np.mean(surv_list)), float(np.std(surv_list))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--agents",    type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--episodes",  type=int, default=20)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  구버전 PPO (obs=3606) 테스트")
    print(f"  모델: fire_evac_model_10ppl.zip")
    print(f"  에피소드: {args.episodes}회/조건")
    print(f"{'═'*60}")
    print(f"  {'시나리오':<10} {'인원':>5} {'생존율 평균':>12} {'표준편차':>10}")
    print(f"  {'─'*44}")

    seeds = list(range(1000, 1000 + args.episodes))

    for sc in args.scenarios:
        sc_name = OldScenarioConfigs[sc]["name"]
        for n in args.agents:
            mean, std = run_test(sc, n, args.episodes, seeds=seeds)
            print(f"  S{sc} {sc_name:<8} {n:>4}명  {mean:>10.1f}%  ±{std:>7.1f}%")

    print(f"  {'─'*44}")
    print()


if __name__ == "__main__":
    main()
