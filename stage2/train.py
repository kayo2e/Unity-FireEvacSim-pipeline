"""
train.py — PPO 학습 / 테스트 진입점
=====================================
환경 로직은 env_core.py에서 관리.
A* 비교는 astar.py, Unity 연동은 unity_bridge.py 참고.
"""

import sys
import numpy as np
import platform

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env_core import FireEvacEnv, SCENARIO_CONFIGS, verify_connectivity


# ══════════════════════════════════════════════
# 커리큘럼 래퍼
# ══════════════════════════════════════════════
class EvacCurriculumWrapper(gym.Wrapper):
    def __init__(self, n_agents: int = 10, threshold: float = 0.50, window: int = 20):
        self.current_scenario = 1
        self.n_agents = n_agents
        env = FireEvacEnv(scenario=1, n_agents=n_agents)
        super().__init__(env)
        self.threshold = threshold
        self.window    = window
        self.recent: list = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            self.recent.append(info["survival_rate"])
            if len(self.recent) > self.window:
                self.recent.pop(0)
            avg = sum(self.recent) / len(self.recent)
            if (len(self.recent) == self.window
                    and avg >= self.threshold
                    and self.current_scenario < len(SCENARIO_CONFIGS)):
                self.current_scenario += 1
                self.env = FireEvacEnv(
                    scenario=self.current_scenario, n_agents=self.n_agents)
                self.recent = []
                print(f"\n[커리큘럼] ★ {self.current_scenario}단계 승급! "
                      f"({self.env.cfg['name']}) | 생존율 {avg:.0%}")
        return obs, rew, term, trunc, info

    def reset(self, **kw): return self.env.reset(**kw)

    @property
    def cfg(self): return self.env.cfg


# ══════════════════════════════════════════════
# 콜백
# ══════════════════════════════════════════════
class EvacTrainCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000):
        super().__init__()
        self.log_interval  = log_interval
        self.ep_rewards:  list = []
        self.ep_survival: list = []
        self._cum_rewards = None

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones   = self.locals["dones"]

        if self._cum_rewards is None:
            self._cum_rewards = np.zeros(len(rewards))
        self._cum_rewards += rewards

        for i, done in enumerate(dones):
            if done:
                self.ep_rewards.append(float(self._cum_rewards[i]))
                self._cum_rewards[i] = 0.0

        for info in self.locals.get("infos", []):
            if "survival_rate" in info:
                self.ep_survival.append(info["survival_rate"])

        if self.num_timesteps % self.log_interval == 0 and self.ep_survival:
            n     = min(len(self.ep_survival), 100)
            avg_s = sum(self.ep_survival[-n:]) / n
            avg_r = (sum(self.ep_rewards[-n:]) / min(len(self.ep_rewards), n)
                     if self.ep_rewards else 0)
            print(f"  Step {self.num_timesteps:>8,} | "
                  f"생존율 {avg_s:>5.1%} | 보상 {avg_r:>+7.1f}")
        return True


# ══════════════════════════════════════════════
# 병렬 환경 팩토리
# ══════════════════════════════════════════════
def make_env(n_agents: int, seed: int):
    def _init():
        env = EvacCurriculumWrapper(n_agents=n_agents)
        env.reset(seed=seed)
        return env
    return _init


# ══════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════
def train_fire_evac(person_counts=(10, 30, 50),
                    total_timesteps=300_000,
                    n_envs=None):
    import torch

    n_cpu = __import__('multiprocessing').cpu_count()
    if n_envs is None:
        n_envs = max(4, min(n_cpu, 16))

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    n_lights = FireEvacEnv(scenario=1).n_lights
    obs_dim  = FireEvacEnv(scenario=1).observation_space.shape[0]

    print("=" * 62)
    print("화재대피유도시스템 Stage 2 — PPO 학습 (MlpPolicy)")
    print(f"인원수         : {person_counts}명")
    print(f"총 스텝        : {total_timesteps:,} / 모델")
    print(f"병렬 환경      : {n_envs}개")
    print(f"유도등         : {n_lights}개 셀")
    print(f"관측 차원      : {obs_dim} (6ch×30×20 + 9 scalar)")
    print(f"Policy         : MlpPolicy | net_arch=[256,256]")
    print(f"학습 디바이스  : {device}")
    print(f"군중 물리      : 밀도 속도 감소(Fruin71) + 공황(Helbing00)")
    print("=" * 62)

    for n in person_counts:
        print(f"\n{'─'*62}\n인원수 {n}명 | 커리큘럼 학습 시작\n{'─'*62}")

        env_fns = [make_env(n_agents=n, seed=i) for i in range(n_envs)]
        raw_env = (DummyVecEnv(env_fns) if platform.system() == "Windows"
                   else SubprocVecEnv(env_fns))
        vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0)
        callback = EvacTrainCallback(log_interval=10_000)

        model = PPO(
            "MlpPolicy",
            vec_env,
            device          = device,
            verbose         = 0,
            n_steps         = 1024,
            batch_size      = 256,
            n_epochs        = 10,
            gamma           = 0.99,
            learning_rate   = 3e-4,
            clip_range      = 0.2,
            ent_coef        = 0.02,
            max_grad_norm   = 0.5,
            policy_kwargs   = dict(net_arch=[256, 256]),
            tensorboard_log = "./fire_evac_log/",
        )

        model.learn(
            total_timesteps = total_timesteps,
            callback        = callback,
            tb_log_name     = f"PPO_{n}ppl",
            progress_bar    = True,
        )

        save_path = f"fire_evac_model_{n}ppl"
        model.save(save_path)
        vec_env.save(f"{save_path}_vecnorm.pkl")
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip | "
              f"정규화 통계: {save_path}_vecnorm.pkl")

    print("\n" + "=" * 62)
    print("학습 완료! TensorBoard: tensorboard --logdir ./fire_evac_log/")
    print("=" * 62)


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def test_fire_evac(n_agents: int = 10, scenario: int = 1,
                   n_episodes: int = 10, save_results: bool = True,
                   render: bool = False):
    import os, csv, json
    from datetime import datetime

    model_path   = f"fire_evac_model_{n_agents}ppl"
    vecnorm_path = f"{model_path}_vecnorm.pkl"
    print(f"\n모델 로드: {model_path}.zip")
    model = PPO.load(model_path)

    render_mode = "human" if render else None
    env     = FireEvacEnv(scenario=scenario, n_agents=n_agents,
                          render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"정규화 통계 로드: {vecnorm_path}")

    records = []
    for ep in range(n_episodes):
        obs      = vec_env.reset()
        total_r  = 0.0
        step_cnt = 0
        max_fire = 0
        print(f"\n[에피소드 {ep+1}/{n_episodes}] "
              f"{env.cfg['name']} | {n_agents}명", end="", flush=True)

        for _ in range(env.cfg["max_steps"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, infos = vec_env.step(action)
            info      = infos[0]
            total_r  += float(r[0])
            step_cnt  = info["step"]
            max_fire  = max(max_fire, info["fire_cells"])
            if render: env.render()
            if done[0]: break

        rec = {
            "episode":        ep + 1,
            "scenario":       scenario,
            "scenario_name":  env.cfg["name"],
            "n_agents":       n_agents,
            "escaped":        info["escaped"],
            "escaped_A":      info.get("escaped_A", 0),
            "escaped_B":      info.get("escaped_B", 0),
            "dead":           info["dead"],
            "remaining":      info["remaining"],
            "survival_rate":  round(info["survival_rate"], 4),
            "total_reward":   round(total_r, 2),
            "steps_taken":    step_cnt,
            "max_fire_cells": max_fire,
            "mean_panic":     round(info.get("mean_panic", 0.0), 4),
            "blocked_exits":  str(info["blocked_exits"]),
        }
        records.append(rec)
        print(f" → 탈출 {rec['escaped']}/{n_agents} | "
              f"생존율 {rec['survival_rate']:.0%} | "
              f"공황 {rec['mean_panic']:.2f} | "
              f"보상 {rec['total_reward']:+.1f} | {step_cnt}스텝")

    vec_env.close()

    def stats(vals):
        a = np.array(vals, dtype=float)
        return {"mean":   round(float(a.mean()), 4),
                "std":    round(float(a.std()),  4),
                "min":    round(float(a.min()),  4),
                "max":    round(float(a.max()),  4),
                "median": round(float(np.median(a)), 4)}

    summary = {
        "model":          model_path,
        "scenario":       scenario,
        "scenario_name":  env.cfg["name"],
        "n_agents":       n_agents,
        "n_episodes":     n_episodes,
        "survival_rate":  stats([r["survival_rate"]  for r in records]),
        "total_reward":   stats([r["total_reward"]   for r in records]),
        "steps_taken":    stats([r["steps_taken"]    for r in records]),
        "escaped":        stats([r["escaped"]        for r in records]),
        "escaped_A":      stats([r["escaped_A"]      for r in records]),
        "escaped_B":      stats([r["escaped_B"]      for r in records]),
        "dead":           stats([r["dead"]           for r in records]),
        "mean_panic":     stats([r["mean_panic"]     for r in records]),
        "max_fire_cells": stats([r["max_fire_cells"] for r in records]),
    }

    print("\n" + "═" * 62)
    print(f"  테스트 결과 요약 | {env.cfg['name']} | {n_agents}명 × {n_episodes}회")
    print("═" * 62)
    for key in ("survival_rate", "total_reward", "steps_taken",
                "escaped", "dead", "mean_panic"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}  median={s['median']:>8}")
    print("═" * 62)

    if save_results:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag       = f"s{scenario}_{n_agents}ppl_{ts}"
        csv_path  = f"test_results_{tag}.csv"
        json_path = f"test_summary_{tag}.json"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader(); writer.writerows(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  에피소드 기록 저장: {csv_path}")
        print(f"  통계 요약 저장   : {json_path}")

    return records, summary


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="화재대피유도시스템 Stage 2")
    parser.add_argument("--mode", choices=["train", "test", "check"],
                        default="train")
    parser.add_argument("--people",        type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--steps",         type=int, default=300_000)
    parser.add_argument("--n-envs",        type=int, default=None)
    parser.add_argument("--test-n",        type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=10)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    args = parser.parse_args()

    if args.mode == "check":
        print("환경 검증 중...")
        verify_connectivity()
        env = FireEvacEnv(scenario=1, n_agents=10)
        check_env(env)
        print(f"관측 크기: {env.observation_space.shape}")
        print(f"유도등 수: {env.n_lights}개")
        print("환경 검증 완료!")

    elif args.mode == "train":
        train_fire_evac(person_counts=args.people,
                        total_timesteps=args.steps,
                        n_envs=args.n_envs)

    elif args.mode == "test":
        test_fire_evac(n_agents=args.test_n,
                       scenario=args.test_scenario,
                       n_episodes=args.test_episodes,
                       save_results=not args.no_save,
                       render=args.render)
