"""
에피소드 그리드 시각화 및 저장
================================
실행 예시:
    python3 visualize_episode.py --scenario 1 --baseline bfs
    python3 visualize_episode.py --scenario 2 --baseline astar --gif
    python3 visualize_episode.py --scenario 1 --baseline bfs --every 5 --gif
    python3 visualize_episode.py --scenario 1 --model recurrent_ppo

저장 위치: result/visualize/s{N}_{baseline}_ep{E}/
    - frame_XXXX.png : 스텝별 그리드 이미지
    - episode.gif    : 전체 애니메이션 (--gif 플래그 시)
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_core import (
    FireEvacEnv, SCENARIO_CONFIGS,
    HALL, WALL, EXIT, ROOM,
    N, S, E, W,
    EXIT_A_POS, EXIT_B_POS,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 색상 팔레트 ──────────────────────────────────────
COLORS = {
    "hall":    "#E8E8E8",
    "wall":    "#2C2C2C",
    "exit_a":  "#00CC66",
    "exit_b":  "#009944",
    "room":    "#D4C9A8",
    "fire":    "#FF4500",
    "smoke":   "#B0B0B0",
    "person":  "#1E90FF",
    "blocked": "#8B0000",
}

DIR_ARROW = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}
DIR_LABEL = {N: "↑", S: "↓", E: "→", W: "←"}
EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)


# ── 단일 프레임 렌더링 ────────────────────────────────
def render_frame(env: FireEvacEnv, step: int, ax: plt.Axes, title_extra: str = ""):
    ROWS, COLS = env.ROWS, env.COLS

    # 배경 레이어: RGBA 이미지 직접 구성
    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)

    def hex2rgba(h, a=1.0):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]

    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL:
                img[r, c] = hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                if (r, c) in env.blocked_exits:
                    img[r, c] = hex2rgba(COLORS["blocked"])
                else:
                    img[r, c] = hex2rgba(COLORS["exit_a"])
            elif (r, c) in EXIT_B_SET:
                if (r, c) in env.blocked_exits:
                    img[r, c] = hex2rgba(COLORS["blocked"])
                else:
                    img[r, c] = hex2rgba(COLORS["exit_b"])
            elif cell == ROOM:
                img[r, c] = hex2rgba(COLORS["room"])
            else:
                img[r, c] = hex2rgba(COLORS["hall"])

    # 연기 (반투명 회색)
    smoke_mask = env.smoke_map > 0
    for r, c in zip(*np.where(smoke_mask)):
        img[r, c] = hex2rgba(COLORS["smoke"], 0.55)

    # 화재 (불투명 빨강)
    fire_mask = env.fire_map > 0
    for r, c in zip(*np.where(fire_mask)):
        img[r, c] = hex2rgba(COLORS["fire"])

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # 유도등 화살표
    lmap = {env.light_cells[i]: int(env.light_dirs[i]) for i in range(env.n_lights)}
    arrow_opts = dict(color="#555555", width=0.008, head_width=0.22, head_length=0.18,
                      length_includes_head=True)
    for (r, c), d in lmap.items():
        dx, dy = DIR_ARROW[d]
        ax.arrow(c, r, dx * 0.35, dy * 0.35, **arrow_opts)

    # 사람 (파란 원)
    for p in env.people_data:
        pr, pc = p["pos"]
        ax.plot(pc, pr, "o", color=COLORS["person"],
                markersize=4, markeredgecolor="white", markeredgewidth=0.5)

    # 축 설정
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    alive   = len(env.people_data)
    escaped = env.escaped
    dead    = env.dead
    fire_n  = int(fire_mask.sum())
    title = (f"Step {step:>3} | "
             f"alive={alive} escaped={escaped} dead={dead} fire={fire_n}"
             + (f" | {title_extra}" if title_extra else ""))
    ax.set_title(title, fontsize=8, pad=3)

    # 범례
    legend_items = [
        mpatches.Patch(color=COLORS["exit_a"],  label="Exit A"),
        mpatches.Patch(color=COLORS["exit_b"],  label="Exit B"),
        mpatches.Patch(color=COLORS["blocked"], label="Blocked Exit"),
        mpatches.Patch(color=COLORS["fire"],    label="Fire"),
        mpatches.Patch(color=COLORS["smoke"],   label="Smoke"),
        mpatches.Patch(color=COLORS["person"],  label="Person"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              fontsize=5, framealpha=0.7, ncol=2,
              handlelength=1, handletextpad=0.4, columnspacing=0.6)


# ── 에피소드 실행 + 저장 ──────────────────────────────
def run_and_save(
    scenario:   int   = 1,
    n_agents:   int   = None,
    baseline:   str   = "bfs",
    model_path: str   = None,
    every:      int   = 1,
    make_gif:   bool  = True,
    out_dir:    str   = None,
    episode:    int   = 1,
    seed:       int   = None,
):
    cfg      = SCENARIO_CONFIGS[scenario]
    n_agents = n_agents or cfg["n_agents"]

    # 정책 불러오기
    policy_fn = None
    policy_label = baseline

    if baseline == "bfs":
        from baselines.astar_baseline import bfs_action
        policy_fn = bfs_action
    elif baseline == "astar":
        from baselines.astar_real import astar_action
        policy_fn = astar_action
    elif baseline == "simple_astar":
        from baselines.astar_simple_baseline import simple_astar_action
        policy_fn = simple_astar_action
    elif baseline == "model" or model_path:
        policy_label = "model"
        _path = model_path
        if _path is None:
            for sub in ("recurrent_ppo", "ppo"):
                candidate = os.path.join(_ROOT, "model", sub, f"s{scenario}_best.zip")
                if os.path.exists(candidate):
                    _path = candidate
                    policy_label = sub
                    break
        if _path is None or not os.path.exists(_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {_path}")
        print(f"  모델 로드: {_path}")

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = PPO.load(_path)
        vecnorm_pkl = _path.replace(".zip", "_vecnorm.pkl")
        _vn = None
        if os.path.exists(vecnorm_pkl):
            _tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=scenario, n_agents=n_agents)])
            _vn = VecNormalize.load(vecnorm_pkl, _tmp)
            _vn.training    = False
            _vn.norm_reward = False
            print(f"  VecNorm 로드: {vecnorm_pkl}")

        def policy_fn(env):
            raw_obs = env._get_obs()
            obs = _vn.normalize_obs(np.array([raw_obs]))[0] if _vn else raw_obs
            action, _ = model.predict(obs, deterministic=True)
            return action

    # 출력 디렉토리
    if out_dir is None:
        out_dir = os.path.join(_ROOT, "result", "visualize",
                               f"episode_s{scenario}_{policy_label}_ep{episode}")
    os.makedirs(out_dir, exist_ok=True)

    # 환경 초기화
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    obs, info = env.reset(seed=seed)

    print(f"\n  Scenario {scenario} | agents={n_agents} | policy={policy_label} | seed={seed}")
    print(f"  Output: {out_dir}")

    frame_paths = []
    step = 0

    # 초기 프레임
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    render_frame(env, step, ax)
    fpath = os.path.join(out_dir, f"frame_{step:04d}.png")
    fig.savefig(fpath, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    frame_paths.append(fpath)

    for _ in range(cfg["max_steps"]):
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if step % every == 0 or terminated or truncated:
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            render_frame(env, step, ax)
            fpath = os.path.join(out_dir, f"frame_{step:04d}.png")
            fig.savefig(fpath, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            frame_paths.append(fpath)
            print(f"  [{step:>4}스텝] 탈출 {info['escaped']}/{n_agents} "
                  f"사망 {info['dead']} 화재 {info['fire_cells']}")

        if terminated or truncated:
            break

    env.close()
    print(f"  {len(frame_paths)} frames saved")

    # GIF 생성
    if make_gif and len(frame_paths) > 1:
        gif_path = os.path.join(out_dir, "episode.gif")
        frames_pil = [Image.open(p) for p in frame_paths]
        frames_pil[0].save(
            gif_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=120,
            loop=0,
        )
        print(f"  GIF saved: {gif_path}")
        # 메모리 해제
        for f in frames_pil:
            f.close()

    # 요약 이미지 (첫·중간·마지막 프레임 3개 비교)
    if len(frame_paths) >= 3:
        picks = [
            frame_paths[0],
            frame_paths[len(frame_paths) // 2],
            frame_paths[-1],
        ]
        labels = ["Initial", "Mid", "Final"]
        fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=100)
        for ax, fp, lbl in zip(axes, picks, labels):
            img = Image.open(fp)
            ax.imshow(np.array(img))
            ax.set_title(lbl, fontsize=11)
            ax.axis("off")
        fig.suptitle(
            f"Scenario {scenario} | agents={n_agents} | policy={policy_label} | "
            f"escaped={info['escaped']} dead={info['dead']}",
            fontsize=11,
        )
        summary_path = os.path.join(out_dir, "summary.png")
        fig.savefig(summary_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary image: {summary_path}")

    return out_dir, info


# ── 진입점 ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="에피소드 그리드 시각화 저장")
    parser.add_argument("--scenario",  type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--n",         type=int, default=None, help="인원수")
    parser.add_argument("--baseline",  type=str, default="bfs",
                        choices=["bfs", "astar", "simple_astar", "model"],
                        help="policy: bfs / astar / simple_astar / model")
    parser.add_argument("--model-path", type=str, default=None,
                        help="모델 .zip 경로 (--baseline model 시 자동 탐색)")
    parser.add_argument("--every",     type=int, default=1,
                        help="N스텝마다 프레임 저장 (기본=1, 전체)")
    parser.add_argument("--no-gif",    action="store_true", help="GIF 생성 안 함")
    parser.add_argument("--episode",   type=int, default=1, help="에피소드 번호(출력 폴더용)")
    parser.add_argument("--seed",      type=int, default=None,
                        help="랜덤 시드 고정 (같은 값이면 모든 모델이 동일한 에피소드 실행)")
    parser.add_argument("--all-scenarios", action="store_true", help="시나리오 1~4 전부")
    args = parser.parse_args()

    scenarios = [1, 2, 3, 4] if args.all_scenarios else [args.scenario]

    for sc in scenarios:
        run_and_save(
            scenario=sc,
            n_agents=args.n,
            baseline=args.baseline,
            model_path=args.model_path,
            every=args.every,
            make_gif=not args.no_gif,
            episode=args.episode,
            seed=args.seed,
        )