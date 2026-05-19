"""
모델별 경로 유도 비교 시각화
================================
옵션 1: 같은 시드·같은 스텝에서 BFS / A* / PPO 나란히 비교
옵션 2: 유도등 셀을 출구 A(초록) / B(파랑)으로 색칠한 라우팅 히트맵

실행 예시:
    python3 visualize_compare.py --scenario 4 --seed 125
    python3 visualize_compare.py --scenario 4 --seed 125 --gif
    python3 visualize_compare.py --scenario 1 --seed 0 --every 5

저장 위치: result/visualize/compare_s{N}_seed{S}/
    - frame_XXXX_compare.png : 스텝별 모델 비교 이미지
    - compare.gif            : 전체 비교 애니메이션
    - summary_compare.png    : 초기 / 중간 / 최종 3시점 요약
"""

import io
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_core import (
    FireEvacEnv, SCENARIO_CONFIGS,
    HALL, WALL, EXIT, ROOM,
    N, S, E, W,
    EXIT_A_POS, EXIT_B_POS,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

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
    "route_a": "#00CC66",
    "route_b": "#1E90FF",
    "route_unknown": "#999999",
}

DIR_ARROW  = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}
EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)


def hex2rgba(h, a=1.0):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]


def classify_exit(lmap, start_r, start_c):
    """유도등 화살표 체인을 따라가서 어느 출구로 연결되는지 반환 ('A' / 'B' / None)."""
    visited = set()
    r, c = start_r, start_c
    for _ in range(120):
        if (r, c) in visited:
            return None
        visited.add((r, c))
        if (r, c) in EXIT_A_SET:
            return 'A'
        if (r, c) in EXIT_B_SET:
            return 'B'
        if (r, c) not in lmap:
            return None
        dr, dc = DIR_ARROW[lmap[(r, c)]]
        r, c = r + dr, c + dc
    return None


# ── 단일 패널 렌더링 ─────────────────────────────────
def render_guidance_frame(env: FireEvacEnv, step: int, ax: plt.Axes, model_name: str):
    ROWS, COLS = env.ROWS, env.COLS

    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL:
                img[r, c] = hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                color = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit_a"]
                img[r, c] = hex2rgba(color)
            elif (r, c) in EXIT_B_SET:
                color = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit_b"]
                img[r, c] = hex2rgba(color)
            elif cell == ROOM:
                img[r, c] = hex2rgba(COLORS["room"])
            else:
                img[r, c] = hex2rgba(COLORS["hall"])

    for r, c in zip(*np.where(env.smoke_map > 0)):
        img[r, c] = hex2rgba(COLORS["smoke"], 0.55)
    for r, c in zip(*np.where(env.fire_map > 0)):
        img[r, c] = hex2rgba(COLORS["fire"])

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # 유도등: 라우팅 색 오버레이(옵션 2) + 화살표(옵션 1)
    lmap = {env.light_cells[i]: int(env.light_dirs[i]) for i in range(env.n_lights)}
    for (r, c), d in lmap.items():
        dest = classify_exit(lmap, r, c)
        color = (COLORS["route_a"] if dest == 'A'
                 else COLORS["route_b"] if dest == 'B'
                 else COLORS["route_unknown"])
        rect = plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            facecolor=color, alpha=0.30, zorder=2, linewidth=0,
        )
        ax.add_patch(rect)
        dx, dy = DIR_ARROW[d]
        ax.arrow(
            c, r, dx * 0.35, dy * 0.35,
            color="#333333", width=0.008,
            head_width=0.22, head_length=0.18,
            length_includes_head=True, zorder=3,
        )

    for p in env.people_data:
        pr, pc = p["pos"]
        ax.plot(pc, pr, "o", color=COLORS["person"],
                markersize=4, markeredgecolor="white",
                markeredgewidth=0.5, zorder=4)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    fire_n = int(env.fire_map.sum())
    ax.set_title(
        f"[{model_name}]  Step {step:>3}\n"
        f"escaped={env.escaped}  dead={env.dead}  fire={fire_n}",
        fontsize=8, pad=3,
    )


def _fig_to_pil(fig) -> Image.Image:
    """matplotlib Figure → PIL Image (메모리, 디스크 없음)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── 단일 모델 에피소드 실행 → 프레임 리스트 반환 ──────
def _run_model_episode(
    scenario:  int,
    n_agents:  int,
    seed:      int,
    policy_fn,
    model_name: str,
    every:     int,
    cfg:       dict,
) -> tuple[list, dict]:
    """
    한 모델을 독립적으로 실행하고 PIL Image 리스트와 최종 info를 반환.
    각 모델을 순차 실행해 전역 random state 공유 문제를 방지.
    """
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    env.reset(seed=seed)

    frames = []  # list of (step, PIL Image)

    # 초기 프레임 (action 이전)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    render_guidance_frame(env, 0, ax, model_name)
    frames.append((0, _fig_to_pil(fig)))

    info = {}
    for _ in range(cfg["max_steps"]):
        action = policy_fn(env)
        _, _, term, trunc, info = env.step(action)
        step = env.step_count

        if step % every == 0 or term or trunc:
            fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
            render_guidance_frame(env, step, ax, model_name)
            frames.append((step, _fig_to_pil(fig)))

        if term or trunc:
            break

    env.close()
    return frames, info


# ── 프레임 합성 및 저장 ──────────────────────────────
def _compose_frame(
    frame_sets: dict,       # {model_name: (step, PIL Image)}
    model_names: list,
    scenario: int,
    seed: int,
    out_dir: str,
    frame_idx: int,
) -> str:
    n = len(model_names)
    panels = [np.array(frame_sets[name][1]) for name in model_names]
    steps  = [frame_sets[name][0]           for name in model_names]

    # 패널 높이 맞추기
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.ones((max_h - p.shape[0], p.shape[1], p.shape[2]),
                          dtype=p.dtype) * 255
            p = np.vstack([p, pad])
        padded.append(p)

    combined = np.hstack(padded)
    fig, ax = plt.subplots(figsize=(combined.shape[1] / 100,
                                    combined.shape[0] / 100 + 0.6), dpi=100)
    ax.imshow(combined)
    ax.axis("off")

    legend_items = [
        mpatches.Patch(color=COLORS["route_a"], alpha=0.7, label="Routing → Exit A"),
        mpatches.Patch(color=COLORS["route_b"], alpha=0.7, label="Routing → Exit B"),
        mpatches.Patch(color=COLORS["fire"],    label="Fire"),
        mpatches.Patch(color=COLORS["smoke"],   label="Smoke"),
        mpatches.Patch(color=COLORS["person"],  label="Person"),
        mpatches.Patch(color=COLORS["blocked"], label="Blocked Exit"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=6, fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.5, 0.0))

    step_str = " / ".join(f"{n}:s{s}" for n, s in zip(model_names, steps))
    fig.suptitle(
        f"Scenario {scenario}  |  seed={seed}  |  {step_str}",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    fpath = os.path.join(out_dir, f"frame_{frame_idx:04d}_compare.png")
    fig.savefig(fpath, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return fpath


# ── 메인 실행 ─────────────────────────────────────────
def run_comparison(
    scenario:  int  = 4,
    n_agents:  int  = None,
    seed:      int  = 125,
    every:     int  = 1,
    make_gif:  bool = True,
    out_dir:   str  = None,
    ppo_path:  str  = None,
):
    cfg      = SCENARIO_CONFIGS[scenario]
    n_agents = n_agents or cfg["n_agents"]

    from baselines.astar_baseline import bfs_action
    from baselines.astar_real     import astar_action

    policies = {
        "BFS": bfs_action,
        "A*":  astar_action,
    }

    # PPO 자동 탐색
    _ppo_path = ppo_path
    if _ppo_path is None:
        suffix = "20ppl" if n_agents <= 20 else "40ppl"
        for sub in ("ppo", "recurrent_ppo"):
            for fname in (f"s{scenario}_best.zip", f"fire_evac_model_{suffix}.zip"):
                cand = os.path.join(_ROOT, "model", sub, fname)
                if os.path.exists(cand):
                    _ppo_path = cand
                    break
            if _ppo_path:
                break

    if _ppo_path and os.path.exists(_ppo_path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        model = PPO.load(_ppo_path)
        vecnorm_pkl = _ppo_path.replace(".zip", "_vecnorm.pkl")
        _vn = None
        if os.path.exists(vecnorm_pkl):
            _tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=scenario, n_agents=n_agents)])
            _vn = VecNormalize.load(vecnorm_pkl, _tmp)
            _vn.training    = False
            _vn.norm_reward = False

        def _make_ppo_fn(m, vn):
            def _fn(env):
                raw_obs = env._get_obs()
                obs = vn.normalize_obs(np.array([raw_obs]))[0] if vn else raw_obs
                action, _ = m.predict(obs, deterministic=True)
                return action
            return _fn

        policies["PPO"] = _make_ppo_fn(model, _vn)
        print(f"  PPO 로드: {_ppo_path}")
    else:
        print("  [경고] PPO 모델을 찾을 수 없습니다. BFS / A* 만 비교합니다.")

    model_names = list(policies.keys())

    if out_dir is None:
        out_dir = os.path.join(
            _ROOT, "result", "visualize",
            f"compare_s{scenario}_seed{seed}",
        )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  Scenario {scenario} | agents={n_agents} | seed={seed} | models={model_names}")
    print(f"  Output: {out_dir}")

    # ── 모델별 순차 실행 (전역 random state 독립 보장) ──
    all_frames: dict[str, list] = {}
    final_infos: dict[str, dict] = {}

    for name in model_names:
        print(f"  [{name}] 에피소드 실행 중...")
        frames, info = _run_model_episode(
            scenario, n_agents, seed,
            policies[name], name, every, cfg,
        )
        all_frames[name]   = frames
        final_infos[name]  = info
        esc  = info.get("escaped", "?")
        dead = info.get("dead", "?")
        steps = info.get("step", "?")
        print(f"  [{name}] 완료 → escaped={esc}  dead={dead}  steps={steps}")

    # ── 프레임 합성 ──
    max_len     = max(len(v) for v in all_frames.values())
    frame_paths = []

    for i in range(max_len):
        frame_set = {}
        for name in model_names:
            frames = all_frames[name]
            idx    = min(i, len(frames) - 1)
            frame_set[name] = frames[idx]

        fpath = _compose_frame(
            frame_set, model_names, scenario, seed, out_dir, i,
        )
        frame_paths.append(fpath)

    print(f"  {len(frame_paths)} comparison frames saved")

    # ── GIF ──
    if make_gif and len(frame_paths) > 1:
        gif_path = os.path.join(out_dir, "compare.gif")
        frames_pil = [Image.open(p) for p in frame_paths]
        frames_pil[0].save(
            gif_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=150,
            loop=0,
        )
        print(f"  GIF saved: {gif_path}")
        for f in frames_pil:
            f.close()

    # ── 요약: 초기 / 중간 / 최종 3시점 ──
    if len(frame_paths) >= 3:
        picks  = [
            frame_paths[0],
            frame_paths[len(frame_paths) // 2],
            frame_paths[-1],
        ]
        labels = ["Initial", "Mid", "Final"]

        fig, axes = plt.subplots(3, 1,
                                 figsize=(8 * len(model_names), 7 * 3), dpi=80)
        for ax, fp, lbl in zip(axes, picks, labels):
            ax.imshow(np.array(Image.open(fp)))
            ax.set_title(lbl, fontsize=12)
            ax.axis("off")

        result_str = "  |  ".join(
            f"{n}: esc={final_infos[n].get('escaped','?')} "
            f"dead={final_infos[n].get('dead','?')}"
            for n in model_names
        )
        fig.suptitle(
            f"Scenario {scenario}  |  seed={seed}\n{result_str}",
            fontsize=11,
        )
        summary_path = os.path.join(out_dir, "summary_compare.png")
        fig.savefig(summary_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary: {summary_path}")

    return out_dir, final_infos


# ── 진입점 ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델별 경로 유도 비교 시각화")
    parser.add_argument("--scenario",      type=int, default=4,   choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--n",             type=int, default=None, help="인원수")
    parser.add_argument("--seed",          type=int, default=125,  help="고정 시드 (기본=125)")
    parser.add_argument("--every",         type=int, default=1,    help="N스텝마다 프레임 저장")
    parser.add_argument("--no-gif",        action="store_true",    help="GIF 생성 안 함")
    parser.add_argument("--ppo-path",      type=str, default=None, help="PPO .zip 경로")
    parser.add_argument("--all-scenarios", action="store_true",    help="시나리오 1~4,6 전부")
    args = parser.parse_args()

    scenarios = [1, 2, 3, 4, 6] if args.all_scenarios else [args.scenario]

    for sc in scenarios:
        run_comparison(
            scenario=sc,
            n_agents=args.n,
            seed=args.seed,
            every=args.every,
            make_gif=not args.no_gif,
            ppo_path=args.ppo_path,
        )
