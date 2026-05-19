"""
train_ppo_grid.py — 3,000차원 그리드 관측 PPO 학습/평가
==========================================================
관측 공간: fire_map(1000) + smoke_map(1000) + people_occ(1000) = 3,000차원
동일 조건: net_arch=[256,256], curriculum S1→S4, n_agents=40

실행:
    python3 train_ppo_grid.py --mode train --steps 3000000
    python3 train_ppo_grid.py --mode eval
    python3 train_ppo_grid.py --mode compare   # 15dim vs 3000dim 비교 그래프
"""

import os, sys, time, platform, argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env_core import FireEvacEnv, SCENARIO_CONFIGS
from train_common import (BASE_DIR, EvacTrainCallback, EvacCurriculumWrapper,
                          find_latest_checkpoint)

GRID_MODEL_DIR = os.path.join(BASE_DIR, "model", "ppo_grid")
PPO15_DIR      = os.path.join(BASE_DIR, "model", "ppo")
ROWS, COLS     = 40, 25
OBS_DIM        = ROWS * COLS * 3   # 3,000


# ══════════════════════════════════════════════════════════
# 3,000차원 관측 래퍼
# ══════════════════════════════════════════════════════════
class GridObsWrapper(gym.Wrapper):
    """
    FireEvacEnv(또는 EvacCurriculumWrapper)를 감싸
    fire_map + smoke_map + people_occupancy 를 플래튼한 3,000차원 obs 반환.
    값이 [0,1]로 이미 정규화되어 있어 VecNormalize 불필요.
    """
    def __init__(self, env):
        super().__init__(env)
        n = ROWS * COLS
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 * n,), dtype=np.float32
        )

    def _grid_obs(self):
        inner = self.unwrapped          # 최심부 FireEvacEnv
        fire  = inner.fire_map.flatten().astype(np.float32)
        smoke = inner.smoke_map.flatten().astype(np.float32)
        occ   = np.zeros(ROWS * COLS, dtype=np.float32)
        for p in inner.people_data:
            r, c = p["pos"]
            occ[r * COLS + c] = 1.0
        return np.concatenate([fire, smoke, occ])

    def reset(self, **kw):
        _, info = self.env.reset(**kw)
        return self._grid_obs(), info

    def step(self, action):
        _, reward, term, trunc, info = self.env.step(action)
        return self._grid_obs(), reward, term, trunc, info


# ── 환경 팩토리 ──────────────────────────────────────────
def make_grid_env(seed: int):
    def _init():
        env = GridObsWrapper(EvacCurriculumWrapper(n_agents=40))
        env.reset(seed=seed)
        return env
    return _init


def make_grid_vec_env(n_envs: int):
    fns = [make_grid_env(i) for i in range(n_envs)]
    if platform.system() == "Windows" or n_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns)


# ══════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════
def train(total_timesteps: int = 3_000_000, n_envs: int = 8):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(GRID_MODEL_DIR, exist_ok=True)
    ckpt_dir    = os.path.join(GRID_MODEL_DIR, "checkpoints")
    ckpt_prefix = "ckpt_grid"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path, _, ckpt_steps = find_latest_checkpoint(ckpt_dir, ckpt_prefix)

    print("=" * 62)
    print("3,000차원 그리드 관측 PPO 학습")
    print(f"관측 차원   : {OBS_DIM}")
    print(f"네트워크    : MlpPolicy | net_arch=[256,256]")
    print(f"총 스텝     : {total_timesteps:,}")
    print(f"병렬 환경   : {n_envs}")
    print(f"디바이스    : {device}")
    print(f"VecNormalize: 없음 (관측값 이미 [0,1])")
    print("=" * 62)

    vec_env = make_grid_vec_env(n_envs)

    if ckpt_path and ckpt_steps < total_timesteps:
        print(f"\n체크포인트 이어서 학습 ({ckpt_steps:,} → {total_timesteps:,})")
        model     = PPO.load(ckpt_path, env=vec_env, device=device)
        remaining = total_timesteps - ckpt_steps
        reset_num = False
    else:
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
        )
        remaining = total_timesteps
        reset_num = True

    ckpt_cb = CheckpointCallback(
        save_freq   = max(500_000 // n_envs, 1),
        save_path   = ckpt_dir,
        name_prefix = ckpt_prefix,
        verbose     = 1,
    )
    callback = CallbackList([EvacTrainCallback(log_interval=20_000), ckpt_cb])

    t0 = time.time()
    model.learn(
        total_timesteps     = remaining,
        callback            = callback,
        progress_bar        = True,
        reset_num_timesteps = reset_num,
    )
    elapsed = time.time() - t0

    save_path = os.path.join(GRID_MODEL_DIR, "fire_evac_model_grid")
    model.save(save_path)
    vec_env.close()
    print(f"\n모델 저장: {save_path}.zip")
    print(f"학습 시간: {elapsed/3600:.1f}시간")


# ══════════════════════════════════════════════════════════
# 평가
# ══════════════════════════════════════════════════════════
def eval_grid(n_ep: int = 30, seeds=None) -> dict:
    """3,000차원 모델 시나리오별 평가."""
    path = os.path.join(GRID_MODEL_DIR, "fire_evac_model_grid.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"3000dim 모델 없음: {path}\n먼저 --mode train 실행")
    model = PPO.load(path)

    results = {}
    for sc in [1, 2, 3, 4]:
        cfg   = SCENARIO_CONFIGS[sc]
        n     = cfg["n_agents"]
        surv_list, time_list = [], []
        for ep in range(n_ep):
            env = GridObsWrapper(FireEvacEnv(scenario=sc, n_agents=n))
            env.reset(seed=(seeds[ep] if seeds else None))
            ep_times = []
            for _ in range(cfg["max_steps"]):
                obs = env._grid_obs()
                t0  = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                ep_times.append((time.perf_counter() - t0) * 1000)
                _, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            surv_list.append(info["survival_rate"] * 100)
            time_list.append(float(np.mean(ep_times[1:])) if len(ep_times) > 1 else ep_times[0])
            env.close()
        results[sc] = {
            "surv_mean": float(np.mean(surv_list)),
            "surv_std":  float(np.std(surv_list)),
            "time_mean": float(np.mean(time_list)),
        }
        print(f"  S{sc}: 생존율 {results[sc]['surv_mean']:.1f}% ± {results[sc]['surv_std']:.1f}%"
              f"  추론 {results[sc]['time_mean']:.3f}ms")
    return results


def eval_ppo15(n_ep: int = 30, seeds=None) -> dict:
    """15차원 기존 PPO 시나리오별 평가."""
    path = os.path.join(PPO15_DIR, "fire_evac_model_40ppl.zip")
    pkl  = path.replace(".zip", "_vecnorm.pkl")
    model = PPO.load(path)
    _vn   = None
    if os.path.exists(pkl):
        _tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=1, n_agents=40)])
        _vn  = VecNormalize.load(pkl, _tmp)
        _vn.training = False; _vn.norm_reward = False

    results = {}
    for sc in [1, 2, 3, 4]:
        cfg  = SCENARIO_CONFIGS[sc]
        n    = cfg["n_agents"]
        surv_list, time_list = [], []
        for ep in range(n_ep):
            env = FireEvacEnv(scenario=sc, n_agents=n)
            env.reset(seed=(seeds[ep] if seeds else None))
            ep_times = []
            for _ in range(cfg["max_steps"]):
                raw = env._get_obs()
                obs = _vn.normalize_obs(np.array([raw]))[0] if _vn else raw
                t0  = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                ep_times.append((time.perf_counter() - t0) * 1000)
                _, _, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            surv_list.append(info["survival_rate"] * 100)
            time_list.append(float(np.mean(ep_times[1:])) if len(ep_times) > 1 else ep_times[0])
            env.close()
        results[sc] = {
            "surv_mean": float(np.mean(surv_list)),
            "surv_std":  float(np.std(surv_list)),
            "time_mean": float(np.mean(time_list)),
        }
        print(f"  S{sc}: 생존율 {results[sc]['surv_mean']:.1f}% ± {results[sc]['surv_std']:.1f}%"
              f"  추론 {results[sc]['time_mean']:.3f}ms")
    return results


# ══════════════════════════════════════════════════════════
# 비교 그래프
# ══════════════════════════════════════════════════════════
def plot_compare(r15: dict, r3k: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import matplotlib.gridspec as gridspec

    _NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(_NANUM):
        font_manager.fontManager.addfont(_NANUM)
        matplotlib.rcParams["font.family"] = \
            font_manager.FontProperties(fname=_NANUM).get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False

    sc_names = {1: "S1\n기본", 2: "S2\nEXIT A 위협",
                3: "S3\n진입로 차단", 4: "S4\n양방향 위협"}
    scenarios = [1, 2, 3, 4]
    x     = np.arange(len(scenarios))
    width = 0.32

    surv15  = [r15[s]["surv_mean"] for s in scenarios]
    surv3k  = [r3k[s]["surv_mean"] for s in scenarios]
    std15   = [r15[s]["surv_std"]  for s in scenarios]
    std3k   = [r3k[s]["surv_std"]  for s in scenarios]
    time15  = [r15[s]["time_mean"] for s in scenarios]
    time3k  = [r3k[s]["time_mean"] for s in scenarios]

    # ── 파라미터 수 ──
    n_params_15 = sum(p.numel() for p in PPO.load(
        os.path.join(PPO15_DIR, "fire_evac_model_40ppl.zip")).policy.parameters())
    n_params_3k = sum(p.numel() for p in PPO.load(
        os.path.join(GRID_MODEL_DIR, "fire_evac_model_grid.zip")).policy.parameters())

    fig = plt.figure(figsize=(14, 10), dpi=130)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    ax_surv  = fig.add_subplot(gs[0, :])   # 상단 전체: 생존율 비교
    ax_time  = fig.add_subplot(gs[1, 0])   # 하단 좌: 추론 시간
    ax_param = fig.add_subplot(gs[1, 1])   # 하단 우: 파라미터 수

    BLUE, RED = "#1E90FF", "#FF6B6B"
    err_kw = dict(capsize=4, capthick=1.2, elinewidth=1.2, ecolor="#555")

    # ────── 생존율 ──────────────────────────────────────
    bars_15 = ax_surv.bar(x - width / 2, surv15, width, label="PPO 15차원",
                          color=BLUE, yerr=std15, error_kw=err_kw,
                          edgecolor="white", linewidth=1)
    bars_3k = ax_surv.bar(x + width / 2, surv3k, width, label="PPO 3,000차원",
                          color=RED,  yerr=std3k, error_kw=err_kw,
                          edgecolor="white", linewidth=1)

    for bar, v in zip(bars_15, surv15):
        ax_surv.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    for bar, v in zip(bars_3k, surv3k):
        ax_surv.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=10)

    for i, s in enumerate(scenarios):
        diff = surv15[i] - surv3k[i]
        sign = "+" if diff >= 0 else ""
        clr  = "#1565C0" if diff >= 0 else "#B71C1C"
        top  = max(surv15[i] + std15[i], surv3k[i] + std3k[i]) + 6
        ax_surv.text(i, top, f"15dim {sign}{diff:.1f}%p",
                     ha="center", fontsize=9.5, color=clr, fontweight="bold")

    ax_surv.set_xticks(x)
    ax_surv.set_xticklabels([sc_names[s] for s in scenarios], fontsize=12)
    ax_surv.set_ylabel("생존율 (%)", fontsize=12)
    ax_surv.set_ylim(0, 130)
    ax_surv.set_title(
        "시나리오별 생존율: PPO 15차원 vs PPO 3,000차원 (n_agents=40, 30회 ± std)",
        fontsize=13, fontweight="bold", pad=8)
    ax_surv.legend(fontsize=11, loc="upper right")
    ax_surv.grid(axis="y", alpha=0.3)
    ax_surv.axhline(100, color="#888", lw=0.8, ls="--", alpha=0.4)

    # ────── 추론 시간 ────────────────────────────────────
    xb = np.arange(4)
    ax_time.bar(xb - 0.2, time15, 0.38, label="15차원", color=BLUE,
                edgecolor="white")
    ax_time.bar(xb + 0.2, time3k, 0.38, label="3,000차원", color=RED,
                edgecolor="white")
    for xi, (t1, t3) in enumerate(zip(time15, time3k)):
        ax_time.text(xi - 0.2, t1 + 0.01, f"{t1:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
        ax_time.text(xi + 0.2, t3 + 0.01, f"{t3:.2f}", ha="center",
                     va="bottom", fontsize=8.5)
    ax_time.set_xticks(xb)
    ax_time.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=11)
    ax_time.set_ylabel("추론 시간 (ms)", fontsize=11)
    ax_time.set_title("시나리오별 추론 시간", fontsize=12, fontweight="bold")
    ax_time.legend(fontsize=10)
    ax_time.grid(axis="y", alpha=0.3)

    # ────── 파라미터 수 ─────────────────────────────────
    labels_p = ["PPO 15차원", "PPO 3,000차원"]
    params   = [n_params_15, n_params_3k]
    bars_p   = ax_param.bar(labels_p, params, color=[BLUE, RED], width=0.45,
                             edgecolor="white")
    for bar, v in zip(bars_p, params):
        ax_param.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + max(params) * 0.02,
                      f"{v:,}", ha="center", va="bottom", fontsize=10,
                      fontweight="bold")
    ratio = params[1] / params[0]
    ax_param.annotate(
        f"{ratio:.1f}× 더 많음",
        xy=(1, params[1]), xytext=(0.5, params[1] * 0.65),
        fontsize=11, color="#B71C1C", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_param.set_ylabel("파라미터 수", fontsize=11)
    ax_param.set_title("전체 네트워크 파라미터", fontsize=12, fontweight="bold")
    ax_param.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1e6)}M" if v >= 1e6
                          else f"{int(v/1e3)}K"))
    ax_param.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "PPO 15차원 스칼라 피처 vs PPO 3,000차원 그리드 플래튼 — 전면 비교\n"
        "(동일 아키텍처: net_arch=[256,256] | 동일 학습 조건: 커리큘럼 S1→S4 | n_agents=40)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  그래프 저장: {out_path}")


# ══════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["train", "eval", "compare"], default="train")
    parser.add_argument("--steps",  type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--ep",     type=int, default=30)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.steps, n_envs=args.n_envs)

    elif args.mode == "eval":
        seeds = list(range(1000, 1000 + args.ep))
        print("\n── 15차원 PPO 평가 ──")
        r15 = eval_ppo15(n_ep=args.ep, seeds=seeds)
        print("\n── 3,000차원 PPO 평가 ──")
        r3k = eval_grid(n_ep=args.ep, seeds=seeds)

        out = os.path.join(BASE_DIR, "result", "obs_comparison",
                           f"ppo15_vs_ppo3k_ep{args.ep}.png")
        plot_compare(r15, r3k, out)

    elif args.mode == "compare":
        seeds = list(range(1000, 1000 + args.ep))
        print("\n── 15차원 PPO 평가 ──")
        r15 = eval_ppo15(n_ep=args.ep, seeds=seeds)
        print("\n── 3,000차원 PPO 평가 ──")
        r3k = eval_grid(n_ep=args.ep, seeds=seeds)
        out = os.path.join(BASE_DIR, "result", "obs_comparison",
                           f"ppo15_vs_ppo3k_ep{args.ep}.png")
        plot_compare(r15, r3k, out)


if __name__ == "__main__":
    main()
