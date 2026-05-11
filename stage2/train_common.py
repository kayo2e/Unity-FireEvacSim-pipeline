"""
train_common.py — 공유 유틸리티
=================================
커리큘럼 래퍼, 콜백, BC 사전학습, 테스트 함수.
ppo_train.py / recurrent_ppo_train.py 양쪽에서 import해 사용.
"""

import sys
import os
import numpy as np
import platform
import gymnasium as gym

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from astar_baseline import rule_based_action

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════
# 커리큘럼 래퍼
# ══════════════════════════════════════════════
class EvacCurriculumWrapper(gym.Wrapper):
    def __init__(self, n_agents: int = 10, threshold: float = 0.85, window: int = 50):
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
                cfg_n = SCENARIO_CONFIGS[self.current_scenario]["n_agents"]
                self.env = FireEvacEnv(
                    scenario=self.current_scenario, n_agents=cfg_n)
                self.recent = []
                print(f"\n[커리큘럼] ★ {self.current_scenario}단계 승급! "
                      f"({self.env.cfg['name']}) | 생존율 {avg:.0%}")
        return obs, rew, term, trunc, info

    def reset(self, **kw):
        return self.env.reset(**kw)

    @property
    def cfg(self):
        return self.env.cfg


# ══════════════════════════════════════════════
# 학습 콜백
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
# 환경 팩토리
# ══════════════════════════════════════════════
def make_env(n_agents: int, seed: int):
    def _init():
        env = EvacCurriculumWrapper(n_agents=n_agents)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(n_agents: int, n_envs: int):
    env_fns = [make_env(n_agents=n_agents, seed=i) for i in range(n_envs)]
    raw = (DummyVecEnv(env_fns) if platform.system() == "Windows"
           else SubprocVecEnv(env_fns))
    return VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)


# ══════════════════════════════════════════════
# BC 사전학습
# ══════════════════════════════════════════════
def collect_astar_demos(n_agents: int, n_envs_demo: int = 4,
                        n_steps: int = 3000, s4_steps: int = 2000) -> tuple:
    env_fns    = [make_env(n_agents=n_agents, seed=200 + i)
                  for i in range(n_envs_demo)]
    demo_raw   = DummyVecEnv(env_fns)
    demo_vnorm = VecNormalize(demo_raw, norm_obs=True, norm_reward=False, clip_obs=10.0)

    all_obs, all_acts = [], []
    obs = demo_vnorm.reset()

    for _ in range(n_steps):
        actions = np.array(
            [rule_based_action(e.env._get_obs()) for e in demo_raw.envs],
            dtype=np.float32,
        )
        all_obs.append(obs.copy())
        all_acts.append(actions.copy())
        obs, _, _, _ = demo_vnorm.step(actions)

    if s4_steps > 0:
        s4_n = SCENARIO_CONFIGS[4]["n_agents"]
        for e in demo_raw.envs:
            e.env = FireEvacEnv(scenario=4, n_agents=s4_n)
        obs = demo_vnorm.reset()
        for _ in range(s4_steps):
            actions = np.array(
                [rule_based_action(e.env._get_obs()) for e in demo_raw.envs],
                dtype=np.float32,
            )
            all_obs.append(obs.copy())
            all_acts.append(actions.copy())
            obs, _, _, _ = demo_vnorm.step(actions)

    demo_vnorm.close()
    return np.concatenate(all_obs), np.concatenate(all_acts)


def pretrain_bc(model, obs_arr: np.ndarray, act_arr: np.ndarray,
                n_epochs: int = 5, batch_size: int = 256, lr: float = 3e-4):
    import torch
    import torch.nn.functional as F

    obs_t = torch.FloatTensor(obs_arr).to(model.device)
    act_t = torch.FloatTensor(act_arr).to(model.device)
    n     = len(obs_t)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=lr)
    print(f"  데모 샘플 수: {n:,} | 배치: {batch_size} | 에폭: {n_epochs}")

    for epoch in range(n_epochs):
        perm       = torch.randperm(n)
        total_loss = 0.0
        n_batches  = 0
        for i in range(0, n - batch_size, batch_size):
            idx  = perm[i: i + batch_size]
            dist = model.policy.get_distribution(obs_t[idx])
            pred = dist.distribution.loc
            loss = F.mse_loss(pred, act_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        print(f"  BC Epoch {epoch+1}/{n_epochs} | "
              f"MSE Loss: {total_loss / max(n_batches, 1):.4f}")
    print("  BC 사전학습 완료.\n")


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def _stats(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    return {
        "mean":   round(float(a.mean()), 4),
        "std":    round(float(a.std()),  4),
        "min":    round(float(a.min()),  4),
        "max":    round(float(a.max()),  4),
        "median": round(float(np.median(a)), 4),
    }


def test_fire_evac(ModelCls, model_dir: str, result_dir: str,
                   n_agents: int = 10, scenario: int = 1,
                   n_episodes: int = 30, save_results: bool = True,
                   render: bool = False, model_n: int = None):
    import csv
    import json
    from datetime import datetime

    model_n      = model_n if model_n is not None else n_agents
    model_path   = os.path.join(model_dir, f"fire_evac_model_{model_n}ppl")
    vecnorm_path = model_path + "_vecnorm.pkl"

    print(f"\n모델 로드: {model_path}.zip  (테스트 인원: {n_agents}명)")
    model = ModelCls.load(model_path)

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
        obs         = vec_env.reset()
        lstm_states = None
        ep_starts   = np.ones((vec_env.num_envs,), dtype=bool)
        total_r     = 0.0
        step_cnt    = 0
        max_fire    = 0
        print(f"\n[에피소드 {ep+1}/{n_episodes}] "
              f"{env.cfg['name']} | {n_agents}명", end="", flush=True)

        for _ in range(env.cfg["max_steps"]):
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=ep_starts, deterministic=True)
            ep_starts = np.zeros((vec_env.num_envs,), dtype=bool)
            obs, r, done, infos = vec_env.step(action)
            info      = infos[0]
            total_r  += float(r[0])
            step_cnt  = info["step"]
            max_fire  = max(max_fire, info["fire_cells"])
            if render:
                env.render()
            if done[0]:
                break

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

    summary = {
        "model":          model_path,
        "scenario":       scenario,
        "scenario_name":  env.cfg["name"],
        "n_agents":       n_agents,
        "n_episodes":     n_episodes,
        "survival_rate":  _stats([r["survival_rate"]  for r in records]),
        "total_reward":   _stats([r["total_reward"]   for r in records]),
        "steps_taken":    _stats([r["steps_taken"]    for r in records]),
        "escaped":        _stats([r["escaped"]        for r in records]),
        "escaped_A":      _stats([r["escaped_A"]      for r in records]),
        "escaped_B":      _stats([r["escaped_B"]      for r in records]),
        "dead":           _stats([r["dead"]           for r in records]),
        "mean_panic":     _stats([r["mean_panic"]     for r in records]),
        "max_fire_cells": _stats([r["max_fire_cells"] for r in records]),
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
        os.makedirs(result_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag       = f"s{scenario}_{n_agents}ppl_{ts}"
        csv_path  = os.path.join(result_dir, f"test_results_{tag}.csv")
        json_path = os.path.join(result_dir, f"test_summary_{tag}.json")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n  에피소드 기록 저장: {csv_path}")
        print(f"  통계 요약 저장   : {json_path}")

    return records, summary
