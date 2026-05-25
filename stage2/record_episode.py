"""
Unity 없이 시뮬레이션을 돌려 재생용 JSON Lines 파일을 생성합니다.

사용법:
  python record_episode.py --scenario 4
  python record_episode.py --scenario 4 --all-scenarios
"""
import json
import os
import argparse
import pickle
import numpy as np
from stable_baselines3 import PPO
from env_core import FireEvacEnv, SCENARIO_CONFIGS

DEMO_SEEDS = {1: 42, 2: 5, 3: 1, 4: 4, 5: 42, 6: 20}


def load_model(scenario):
    n_agents = SCENARIO_CONFIGS[scenario]["n_agents"]
    model_dir = os.path.join("model", "ppo")
    model_name = f"fire_evac_model_{n_agents}ppl"
    model_path = os.path.join(model_dir, model_name)
    vecnorm_path = f"{model_path}_vecnorm.pkl"

    model = PPO.load(model_path)
    vecnorm = None
    if os.path.exists(vecnorm_path):
        with open(vecnorm_path, "rb") as f:
            vecnorm = pickle.load(f)
        vecnorm.training = False
        vecnorm.norm_reward = False
    return model, vecnorm, n_agents


def record(scenario):
    seed = DEMO_SEEDS.get(scenario, 42)
    model, vecnorm, n_agents = load_model(scenario)

    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    obs, info = env.reset(seed=seed)

    os.makedirs("recordings", exist_ok=True)
    out_path = os.path.join("recordings", f"recording_s{scenario}_seed{seed}.jsonl")

    smooth_action = None

    with open(out_path, "w", encoding="utf-8") as f:
        # init 메시지
        init_msg = {
            "message_type": "init",
            "grid_info": env.get_grid_info(),
            "initial_snapshot": env.get_snapshot(),
        }
        f.write(json.dumps(init_msg, default=str) + "\n")

        step = 0
        while True:
            obs_in = obs.reshape(1, -1)
            if vecnorm is not None:
                obs_in = vecnorm.normalize_obs(obs_in)

            action, _ = model.predict(obs_in, deterministic=True)
            action = action[0].tolist()

            # EMA smoothing
            if smooth_action is None:
                smooth_action = list(action)
            smooth_action = [0.35 * a + 0.65 * s for a, s in zip(action, smooth_action)]
            action = list(smooth_action)

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            snapshot = env.get_snapshot()

            response = {
                "message_type": "step_snapshot",
                "exit_A_cost": float(action[0]),
                "exit_B_cost": float(action[1]),
                "crowd_weight": float(action[2]),
                "snapshot": snapshot,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
                "step": step,
            }
            f.write(json.dumps(response, default=str) + "\n")

            if terminated or truncated:
                end_msg = {"message_type": "episode_end", "final_info": info}
                f.write(json.dumps(end_msg, default=str) + "\n")
                print(f"✅ 시나리오 {scenario} | 생존율: {info['survival_rate']:.1%} | "
                      f"스텝: {step} | 저장: {out_path}")
                break

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=4)
    parser.add_argument("--all-scenarios", action="store_true")
    args = parser.parse_args()

    scenarios = list(DEMO_SEEDS.keys()) if args.all_scenarios else [args.scenario]
    for sc in scenarios:
        record(sc)
