"""
train_autoregressive.py — AutoregressivePPO (RecurrentPPO + next-token 환경) 학습/테스트
==========================================================================================
알고리즘 : RecurrentPPO (MlpLstmPolicy)
환경     : AutoregressiveEvacEnv  ← 시뮬 1tick = K_MAX gym step으로 분해
모델 저장 : model/autoregressive_ppo/
결과 저장 : result/autoregressive_ppo/
로그      : logs/autoregressive_ppo/

JointPPO 비교용: 동일 커리큘럼/보상 조건, 알고리즘만 다름.
  JointPPO     — MultiDiscrete([2]*64) 동시 결정 + TransformerEncoder
  AutoregressivePPO — Discrete(2) 순차 결정 + LSTM (이 파일)

실행:
    python autoregressive_ppo/train_autoregressive.py --mode train --steps 3000000
    python autoregressive_ppo/train_autoregressive.py --mode test --all-scenarios --test-episodes 30
"""

import sys
import os

_STAGE2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STAGE2)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
torch.set_num_threads(1)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise SystemExit("sb3_contrib 미설치. pip install sb3-contrib")

import numpy as np
import csv
import json
import gymnasium as gym
from datetime import datetime  # test() 결과 저장 타임스탬프에 사용

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

from env_core import SCENARIO_CONFIGS
from joint_ppo.env_joint import K_MAX
from autoregressive_ppo.env_autoregressive import AutoregressiveEvacEnv, OBS_DIM
from train_common import find_latest_checkpoint

BASE_DIR   = _STAGE2
MODEL_DIR  = os.path.join(BASE_DIR, "model",  "autoregressive_ppo")
RESULT_DIR = os.path.join(BASE_DIR, "result", "autoregressive_ppo")
LOG_DIR    = os.path.join(BASE_DIR, "logs",   "autoregressive_ppo")


# ══════════════════════════════════════════════
# 커리큘럼 래퍼
# ══════════════════════════════════════════════
class AutoregressiveCurriculumWrapper(gym.Wrapper):
    """생존율 임계치 도달 시 시나리오를 순서대로 진급."""

    def __init__(self, threshold: float = 0.8, window: int = 50,
                 k_max: int = K_MAX):
        self.current_scenario = 1
        self._k_max           = k_max
        s1_n  = SCENARIO_CONFIGS[1]["n_agents"]
        env   = AutoregressiveEvacEnv(scenario=1, n_agents=s1_n, k_max=k_max)
        super().__init__(env)
        self.threshold = threshold
        self.window    = window
        self.recent:   list = []

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if term or trunc:
            if "survival_rate" in info:
                self.recent.append(info["survival_rate"])
                if len(self.recent) > self.window:
                    self.recent.pop(0)
                avg = sum(self.recent) / len(self.recent)
                if (len(self.recent) == self.window
                        and avg >= self.threshold
                        and self.current_scenario < len(SCENARIO_CONFIGS)):
                    self.current_scenario += 1
                    cfg_n = SCENARIO_CONFIGS[self.current_scenario]["n_agents"]
                    self.env.close()
                    self.env = AutoregressiveEvacEnv(
                        scenario=self.current_scenario,
                        n_agents=cfg_n,
                        k_max=self._k_max)
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
class AutoregressiveCallback(BaseCallback):
    def __init__(self, log_interval: int = 10_000):
        super().__init__()
        self.log_interval  = log_interval
        self.ep_rewards:   list = []
        self.ep_survival:  list = []
        self._cum_rewards  = None

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

        for info, done in zip(self.locals.get("infos", []), dones):
            if done and "survival_rate" in info:
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
# 학습
# ══════════════════════════════════════════════
def train(person_counts=None, total_timesteps: int = 3_000_000,
          n_envs: int = 16, lstm_hidden_size: int = 256,
          k_max: int = K_MAX):
    if person_counts is None:
        person_counts = sorted(set(c["n_agents"] for c in SCENARIO_CONFIGS.values()))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 62)
    print("화재대피유도시스템 — AutoregressivePPO 학습")
    print(f"환경     : AutoregressiveEvacEnv (1tick = {k_max} gym step)")
    print(f"Policy   : MlpLstmPolicy | lstm_hidden={lstm_hidden_size}")
    print(f"Obs 차원 : {OBS_DIM}  (셀8 + 글로벌15 + 위치2)")
    print(f"Action   : Discrete(2)  엔트로피=ln(2)≈0.69")
    print(f"디바이스 : {device} | 병렬 환경: {n_envs}개")
    print(f"인원수   : {person_counts}명")
    print(f"모델 저장: {MODEL_DIR}")
    print("=" * 62)

    def _make_env(seed: int, _k=k_max):
        def _init():
            env = AutoregressiveCurriculumWrapper(k_max=_k)
            env.reset(seed=seed)
            return env
        return _init

    for n in person_counts:
        ckpt_dir    = os.path.join(MODEL_DIR, "checkpoints")
        ckpt_prefix = f"ckpt_{n}ppl"
        ckpt_path, vnorm_ckpt, ckpt_steps = find_latest_checkpoint(ckpt_dir, ckpt_prefix)
        os.makedirs(ckpt_dir, exist_ok=True)

        raw_vec = DummyVecEnv([_make_env(i) for i in range(n_envs)])

        if ckpt_path and ckpt_steps < total_timesteps:
            print(f"\n{'─'*62}\n체크포인트 이어서 학습 "
                  f"({n}ppl | {ckpt_steps:,} → {total_timesteps:,} steps)\n{'─'*62}")
            if os.path.exists(vnorm_ckpt):
                vec_env = VecNormalize.load(vnorm_ckpt, raw_vec)
                vec_env.training = True
            else:
                vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=False,
                                       clip_obs=10.0)
            model = RecurrentPPO.load(ckpt_path, env=vec_env, device=device,
                                      tensorboard_log=LOG_DIR)
            remaining = total_timesteps - ckpt_steps
            reset_num = False
        else:
            s1_n = SCENARIO_CONFIGS[1]["n_agents"]
            print(f"\n{'─'*62}\n커리큘럼 학습 시작 "
                  f"(S1 {s1_n}명 → 이후 시나리오별 인원수 자동 적용, 모델명: {n}ppl)\n{'─'*62}")
            vec_env = VecNormalize(raw_vec, norm_obs=True, norm_reward=False,
                                   clip_obs=10.0)
            model = RecurrentPPO(
                "MlpLstmPolicy", vec_env,
                device          = device,
                verbose         = 0,
                n_steps         = 2048,
                batch_size      = 256,
                n_epochs        = 10,
                gamma           = 0.99,
                learning_rate   = 1e-4,
                clip_range      = 0.2,
                ent_coef        = 0.03,
                max_grad_norm   = 0.5,
                policy_kwargs   = dict(
                    lstm_hidden_size = lstm_hidden_size,
                    net_arch         = [64, 64],
                ),
                tensorboard_log = LOG_DIR,
            )
            remaining = total_timesteps
            reset_num = True

        ckpt_cb = CheckpointCallback(
            save_freq         = max(500_000 // n_envs, 1),
            save_path         = ckpt_dir,
            name_prefix       = ckpt_prefix,
            save_vecnormalize = True,
            verbose           = 1,
        )
        callback = CallbackList([AutoregressiveCallback(log_interval=10_000), ckpt_cb])

        model.learn(
            total_timesteps     = remaining,
            callback            = callback,
            tb_log_name         = f"AutoregressivePPO_{n}ppl",
            progress_bar        = True,
            reset_num_timesteps = reset_num,
        )

        save_path = os.path.join(MODEL_DIR, f"fire_evac_model_{n}ppl")
        model.save(save_path)
        vec_env.save(save_path + "_vecnorm.pkl")
        vec_env.close()
        print(f"\n모델 저장: {save_path}.zip")

    print(f"\n학습 완료! TensorBoard: tensorboard --logdir {LOG_DIR}")


# ══════════════════════════════════════════════
# 테스트
# ══════════════════════════════════════════════
def _stats(vals: list) -> dict:
    a = np.array(vals, dtype=float)
    return {
        "mean":   round(float(a.mean()),     4),
        "std":    round(float(a.std()),      4),
        "min":    round(float(a.min()),      4),
        "max":    round(float(a.max()),      4),
        "median": round(float(np.median(a)), 4),
    }


def test(n_agents: int = 10, scenario: int = 1,
         n_episodes: int = 30, model_n: int = None,
         save_results: bool = True, render: bool = False,
         k_max: int = K_MAX):

    model_n      = model_n if model_n is not None else n_agents
    model_path   = os.path.join(MODEL_DIR, f"fire_evac_model_{model_n}ppl")
    vecnorm_path = model_path + "_vecnorm.pkl"

    print(f"\n모델 로드: {model_path}.zip  (테스트 인원: {n_agents}명)")
    model = RecurrentPPO.load(model_path)

    render_mode = "human" if render else None
    env     = AutoregressiveEvacEnv(scenario=scenario, n_agents=n_agents,
                                    render_mode=render_mode, k_max=k_max)
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"정규화 통계 로드: {vecnorm_path}")

    # 에피소드당 최대 gym step: 시뮬 max_steps × k_max
    max_gym_steps = env.cfg["max_steps"] * k_max

    records = []
    for ep in range(n_episodes):
        obs         = vec_env.reset()
        lstm_states = None
        ep_starts   = np.ones((1,), dtype=bool)
        total_r     = 0.0
        last_info   = {}
        print(f"\n[에피소드 {ep+1}/{n_episodes}] "
              f"{env.cfg['name']} | {n_agents}명", end="", flush=True)

        for _ in range(max_gym_steps):
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=ep_starts, deterministic=True)
            ep_starts = np.zeros((1,), dtype=bool)
            obs, r, done, infos = vec_env.step(action)
            total_r += float(r[0])
            info     = infos[0]
            if info:                        # 틱 완료 시만 info가 채워짐
                last_info = info
            if render:
                env.render()
            if done[0]:
                break

        info = last_info
        rec  = {
            "episode":        ep + 1,
            "scenario":       scenario,
            "scenario_name":  env.cfg["name"],
            "n_agents":       n_agents,
            "escaped":        info.get("escaped",       0),
            "escaped_A":      info.get("escaped_A",     0),
            "escaped_B":      info.get("escaped_B",     0),
            "dead":           info.get("dead",          0),
            "remaining":      info.get("remaining",     0),
            "survival_rate":  round(info.get("survival_rate", 0.0), 4),
            "total_reward":   round(total_r, 2),
            "steps_taken":    info.get("step",          0),
            "max_fire_cells": info.get("fire_cells",    0),
            "mean_panic":     round(info.get("mean_panic", 0.0), 4),
            "blocked_exits":  str(info.get("blocked_exits", [])),
        }
        records.append(rec)
        print(f" → 탈출 {rec['escaped']}/{n_agents} | "
              f"생존율 {rec['survival_rate']:.0%} | "
              f"보상 {rec['total_reward']:+.1f} | {rec['steps_taken']}스텝")

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
    print(f"  테스트 결과 | {env.cfg['name']} | {n_agents}명 × {n_episodes}회")
    print("═" * 62)
    for key in ("survival_rate", "total_reward", "steps_taken",
                "escaped", "dead", "mean_panic"):
        s = summary[key]
        print(f"  {key:<16} mean={s['mean']:>8}  std={s['std']:>7}  "
              f"min={s['min']:>7}  max={s['max']:>7}")
    print("═" * 62)

    if save_results:
        os.makedirs(RESULT_DIR, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag       = f"s{scenario}_{n_agents}ppl_{ts}"
        csv_path  = os.path.join(RESULT_DIR, f"test_results_{tag}.csv")
        json_path = os.path.join(RESULT_DIR, f"test_summary_{tag}.json")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n  에피소드 기록: {csv_path}")
        print(f"  통계 요약   : {json_path}")

    return records, summary


# ══════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoregressivePPO 화재대피 학습/테스트")
    parser.add_argument("--mode",          choices=["train", "test"], default="train")
    parser.add_argument("--auto-scenarios", action="store_true",
                        help="SCENARIO_CONFIGS의 고유 n_agents로 자동 학습 (기본 동작)")
    parser.add_argument("--people",        type=int, nargs="+", default=None,
                        help="학습할 인원수 목록 (미지정 시 --auto-scenarios와 동일)")
    parser.add_argument("--steps",         type=int, default=3_000_000)
    parser.add_argument("--n-envs",        type=int, default=16)
    parser.add_argument("--lstm-hidden",   type=int, default=256)
    parser.add_argument("--k-max",         type=int, default=K_MAX,
                        help=f"제어할 최대 경로 셀 수 (기본 {K_MAX})")
    parser.add_argument("--test-n",        type=int, default=10)
    parser.add_argument("--test-scenario", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=30)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--model-n",       type=int, default=None)
    parser.add_argument("--no-save",       action="store_true")
    parser.add_argument("--render",        action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        if args.people is not None:
            counts = args.people
        elif args.auto_scenarios:
            counts = sorted(set(c["n_agents"] for c in SCENARIO_CONFIGS.values()))
        else:
            counts = None
        train(
            person_counts   = counts,
            total_timesteps = args.steps,
            n_envs          = args.n_envs,
            lstm_hidden_size= args.lstm_hidden,
            k_max           = args.k_max,
        )

    elif args.mode == "test":
        scenarios     = [1, 2, 3, 4] if args.all_scenarios else [args.test_scenario]
        all_summaries = {}
        for sc in scenarios:
            n = SCENARIO_CONFIGS[sc]["n_agents"] if args.all_scenarios else args.test_n
            _, summary = test(
                n_agents     = n,
                scenario     = sc,
                n_episodes   = args.test_episodes,
                model_n      = args.model_n,
                save_results = not args.no_save,
                render       = args.render,
                k_max        = args.k_max,
            )
            all_summaries[sc] = summary

        if args.all_scenarios:
            print(f"\n{'═'*62}")
            print("  전체 시나리오 생존율 (AutoregressivePPO)")
            print(f"{'─'*62}")
            for sc, s in all_summaries.items():
                sr = s["survival_rate"]
                print(f"  S{sc} {s['scenario_name']:<8} | "
                      f"생존율 {sr['mean']:.1%} ± {sr['std']:.1%}  "
                      f"(min {sr['min']:.0%} ~ max {sr['max']:.0%})")
            print(f"{'═'*62}")
