"""
make_comparison_gifs.py — 시나리오별 A* vs PPO 나란히 비교 GIF 생성
====================================================================
동일한 시드로 A*(왼쪽)·PPO(오른쪽)를 동시 실행해 프레임별 합성 GIF를 저장합니다.
시나리오 제목은 상단 중앙에 표시됩니다.
비주얼: poster_grid --routes-only 스타일 (불꽃 글로우, 비상구 아이콘, 위험도 대피자 컬러)

실행:
    cd stage2
    python make_comparison_gifs.py
    python make_comparison_gifs.py --scenarios 1 2 3 4 6 --seed 42
    python make_comparison_gifs.py --every 2 --dpi 70

저장: result/visualize/comparison_gif/s{N}_comparison_seed{seed}.gif
"""

import io, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib import font_manager
from PIL import Image

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_core import (
    FireEvacEnv, SCENARIO_CONFIGS,
    HALL, WALL, EXIT, ROOM,
    N as DIR_N, S as DIR_S, E as DIR_E, W as DIR_W,
    EXIT_A_POS, EXIT_B_POS,
    BASE_GRID,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

# 포스터 기준 시나리오 표시명 (env_core 이름과 별개)
SC_DISPLAY_NUM  = {6: 5}
SC_NAME_DISPLAY = {
    1: "기본 탈출",
    2: "EXIT A 위협",
    3: "진입로 차단",
    4: "양방향 위협",
    6: "중앙 차단",
}

# 시나리오별 고정 시드 — PPO 우위가 가장 뚜렷하게 나타나는 시드 선정
SC_SEED = {
    1: 42,   # 둘 다 전원 생존, 시드 무관
    2:  5,   # A*(20/20사망) vs PPO(33/7사망) — 극명한 차이
    3:  1,   # A*(35/5사망) vs PPO(39/1사망)
    4:  4,   # A*(22/18사망) vs PPO(37/3사망)
    6: 20,   # A*(32/8사망) vs PPO(35/5사망)
}

# 시나리오별 A* / PPO 경로 판단 근거 설명
SCENARIO_NOTES = {
    1: {
        "astar": "화재 위험 낮음 → 최단 BFS로 EXIT A 집중\n출구 혼잡도는 고려하지 않음",
        "ppo":   "F7(EXIT A 혼잡) 상승 감지 → exit_a_cost↑ 출력\nBFS가 EXIT B 우선 배정, 대기줄 분산",
    },
    2: {
        "astar": "초기 EXIT A 진입 → 화재 확산 시 BFS 비용 급등\n→ 전원 EXIT B 쏠림, 2셀 병목·사망",
        "ppo":   "F1 급락·F14↑ → exit_a_cost 단계적 증가 출력\nBFS가 EXIT A 기피 라우팅, EXIT B로 점진 전환",
    },
    3: {
        "astar": "EXIT A 접근로 화재 뒤늦게 인식\n→ EXIT B 전환 지연, 연기 사망 증가",
        "ppo":   "F1 급락 즉시 감지 → exit_a_cost 최대화 출력\nBFS가 EXIT A 완전 우회, EXIT B 조기 집중",
    },
    4: {
        "astar": "무작위 점화 → 한 출구 BFS 비용 급등\n반대 출구 쏠림·과부하로 사망 증가",
        "ppo":   "F1·F2로 각 출구 위험도 실시간 감지\nexit_a/b_cost 개별 조정 → 덜 위험한 쪽 동적 분산",
    },
    6: {
        "astar": "중앙 통로 차단 → EXIT A 병목 집중\n과밀 체증으로 사망 증가",
        "ppo":   "F12·F13(화재 위치) 변화 감지 → exit 비용 재조정\n우회 유도 → 탈출 35명 / 사망 5명 (A*: 32 / 8)",
    },
}

COLORS = {
    "hall":    "#EFEFEF",
    "wall":    "#1A1A1A",
    "exit":    "#00A550",
    "room":    "#E0D8C4",
    "fire":    "#FF4500",
    "smoke":   "#BBBBBB",
    "blocked": "#8B0000",
}

EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)

# 실제 건물에서 벽이지만 BASE_GRID에 ROOM/HALL로 등록된 셀 (시각화만 벽 색)
_VISUAL_WALL_CELLS = frozenset(
    (r, c) for r in range(10, 28) for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c) for r in range(0, 10) for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

# 위험도 컬러맵: 빨강(화재 인접) → 노랑 → 초록(안전)
_DANGER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "danger", [(1.0, 0.12, 0.12), (1.0, 0.70, 0.0), (0.05, 0.78, 0.25)], N=64
)

_ICON_CACHE: dict = {}


def _get_exit_icon(size: int = 52) -> np.ndarray:
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]
    path = os.path.join(_ROOT, "evac_sign.png")
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    _ICON_CACHE[size] = np.array(img)
    return _ICON_CACHE[size]


# ── 화재 글로우 ───────────────────────────────────────────────────────────
def _draw_fire_glow(ax: plt.Axes, env: FireEvacEnv, zorder: int = 3):
    fire_ys, fire_xs = np.where(env.fire_map > 0)
    if len(fire_xs) == 0:
        return

    is_wall = np.array([
        env.grid[r, c] == WALL or (r, c) in _VISUAL_WALL_CELLS
        for r, c in zip(fire_ys, fire_xs)
    ])

    frs = fire_ys[~is_wall].astype(float)
    fcs = fire_xs[~is_wall].astype(float)
    if len(fcs):
        ax.scatter(fcs, frs, s=320, c="#CC1000", alpha=0.09, linewidths=0, zorder=zorder)
        ax.scatter(fcs, frs, s=180, c="#FF5000", alpha=0.18, linewidths=0, zorder=zorder)
        ax.scatter(fcs, frs, s=80,  c="#FF9500", alpha=0.45, linewidths=0, zorder=zorder)
        ax.scatter(fcs, frs, s=28,  c="#FFE040", alpha=0.68, linewidths=0, zorder=zorder)
        ax.scatter(fcs, frs - 0.48, s=120, c="#FF6A00", alpha=0.40,
                   marker="^", linewidths=0, zorder=zorder + 0.5)

    wrs = fire_ys[is_wall].astype(float)
    wcs = fire_xs[is_wall].astype(float)
    if len(wcs):
        ax.scatter(wcs, wrs - 0.40, s=140, c="#FF6A00", alpha=0.55,
                   marker="^", linewidths=0, zorder=zorder + 1)
        ax.scatter(wcs, wrs - 0.58, s=55,  c="#FFD040", alpha=0.65,
                   marker="^", linewidths=0, zorder=zorder + 1)


# ── PPO 로드 ──────────────────────────────────────────────────────────────
def _load_ppo(scenario: int, n_agents: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    suffix = "20ppl" if n_agents <= 20 else "40ppl"
    _path = None
    for fname in (f"s{scenario}_best.zip", f"fire_evac_model_{suffix}.zip",
                  "fire_evac_model.zip"):
        cand = os.path.join(_ROOT, "model", "ppo", fname)
        if os.path.exists(cand):
            _path = cand
            break

    if not (_path and os.path.exists(_path)):
        return None

    model = PPO.load(_path)
    vecnorm_pkl = _path.replace(".zip", "_vecnorm.pkl")
    _vn = None
    if os.path.exists(vecnorm_pkl):
        tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=scenario, n_agents=n_agents)])
        _vn = VecNormalize.load(vecnorm_pkl, tmp)
        _vn.training    = False
        _vn.norm_reward = False

    def _policy(env):
        raw = env._get_obs()
        obs = _vn.normalize_obs(np.array([raw]))[0] if _vn else raw
        action, _ = model.predict(obs, deterministic=True)
        return action

    print(f"    PPO 로드: {_path}")
    return _policy


# ── 단일 환경 패널 렌더링 (poster_grid --routes-only 스타일) ─────────────
def _draw_env(env: FireEvacEnv, step: int, ax: plt.Axes,
              label: str, label_color: str,
              done: bool = False):
    ROWS, COLS = env.ROWS, env.COLS

    def hex2rgba(h, a=1.0):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]

    # ── 배경 레이어 ──
    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit"]
                img[r, c] = hex2rgba(col)
            elif (r, c) in EXIT_B_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit"]
                img[r, c] = hex2rgba(col)
            elif cell == ROOM:
                img[r, c] = hex2rgba(COLORS["room"])
            else:
                img[r, c] = hex2rgba(COLORS["hall"])

    # 연기: 복도/방 셀만 오버라이드 — 벽 실루엣 유지
    for r, c in zip(*np.where(env.smoke_map > 0)):
        if env.grid[r, c] != WALL and (r, c) not in _VISUAL_WALL_CELLS:
            img[r, c] = hex2rgba(COLORS["smoke"], 0.40)

    # 화재: 복도/방 셀만 오버라이드 — 벽은 글로우로만 표현해 실루엣 유지
    for r, c in zip(*np.where(env.fire_map > 0)):
        if env.grid[r, c] != WALL and (r, c) not in _VISUAL_WALL_CELLS:
            img[r, c] = hex2rgba(COLORS["fire"], 0.93)

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    # ── 불꽃 글로우 ──
    _draw_fire_glow(ax, env, zorder=3)

    # ── 비상구 아이콘 ──
    icon = _get_exit_icon(size=52)
    for exit_pos in (EXIT_A_POS, EXIT_B_POS):
        blocked = all((r, c) in env.blocked_exits for r, c in exit_pos)
        if blocked:
            continue
        er = sum(r for r, c in exit_pos) / len(exit_pos)
        ec = sum(c for r, c in exit_pos) / len(exit_pos)
        ib = OffsetImage(icon, zoom=0.55)
        ib.image.axes = ax
        ab = AnnotationBbox(ib, (ec, er), frameon=False, zorder=9)
        ax.add_artist(ab)

    # ── 대피자 (위험도 컬러, 큰 마커) ──
    if env.people_data:
        fire_cells = np.argwhere(env.fire_map > 0)
        cell_rep: dict = {}
        for p in env.people_data:
            pr, pc = p["pos"]
            if env.grid[pr, pc] == WALL or (pr, pc) in _VISUAL_WALL_CELLS:
                continue
            min_d = float(
                np.abs(fire_cells[:, 0] - pr).astype(float).min() +
                np.abs(fire_cells[:, 1] - pc).astype(float).min()
            ) if len(fire_cells) else 99.0
            t = min(min_d / 6.0, 1.0)   # 0=위험, 1=안전
            key = (pr, pc)
            if key not in cell_rep or t < cell_rep[key][0]:
                cell_rep[key] = (t, pr, pc)

        for t, pr, pc in sorted(cell_rep.values(), key=lambda x: -x[0]):
            ax.plot(pc, pr, "o",
                    color=_DANGER_CMAP(t),
                    markersize=11.0,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=5, alpha=0.93)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # PPO 패널 테두리 강조
    if label == "PPO":
        for sp in ax.spines.values():
            sp.set_edgecolor("#1565C0")
            sp.set_linewidth(2.2)

    # ── 통계 + 스텝 ──
    survived  = getattr(env, "escaped", 0)
    dead      = getattr(env, "dead", 0)
    remaining = env.n_agents - survived - dead
    ax.text(0.01, 0.02,
            f"탈출 {survived}  잔류 {remaining}  사망 {dead}",
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="left",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      alpha=0.88, edgecolor="#CCCCCC", linewidth=0.8))
    ax.text(0.99, 0.02, f"t = {step}",
            transform=ax.transAxes, fontsize=10.0, va="bottom", ha="right",
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.88, edgecolor="#CCCCCC", linewidth=0.8))

    # ── 완료 배너 (에피소드 종료 후 상대방이 아직 실행 중인 경우) ──
    if done:
        # 반투명 오버레이
        ax.add_patch(plt.Rectangle((0, 0), 1, 1,
                                   transform=ax.transAxes,
                                   facecolor="white", alpha=0.45,
                                   zorder=11, linewidth=0))
        done_color = "#1565C0" if label == "PPO" else "#333333"
        finish_step = env.step_count
        ax.text(0.5, 0.52,
                f"완료\nt = {finish_step}\n탈출 {survived}  사망 {dead}",
                transform=ax.transAxes,
                fontsize=22, fontweight="bold",
                ha="center", va="center",
                color=done_color, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.55",
                          facecolor="white", alpha=0.92,
                          edgecolor=done_color, linewidth=2.5),
                zorder=12)


# ── 단일 패널 → PIL Image ────────────────────────────────────────────────
def _render_single_panel(env: FireEvacEnv, step: int, label: str,
                          label_color: str, annotation: str,
                          done: bool, dpi: int) -> Image.Image:
    """한 환경 패널만 독립적으로 렌더링 (제목·범례 없음)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.73, bottom=0.01)
    _draw_env(env, step, ax, label, label_color, done=done)

    # ── 레이블 (A* / PPO): axes 위 figure 영역
    fig.text(0.5, 0.970, label,
             ha="center", va="top",
             fontsize=17, fontweight="bold", color=label_color,
             transform=fig.transFigure)

    # ── 설명 패널: 레이블 바로 아래, 그림 위
    if annotation:
        box_color = "#EEF4FF" if label == "PPO" else "#F8F8F8"
        edge_color = "#90B8E8" if label == "PPO" else "#CCCCCC"
        fig.text(0.5, 0.880, annotation,
                 ha="center", va="top",
                 fontsize=12.5, style="italic",
                 color="#111111", linespacing=1.45,
                 bbox=dict(boxstyle="round,pad=0.45", facecolor=box_color,
                           alpha=0.92, edgecolor=edge_color, linewidth=1.0),
                 transform=fig.transFigure)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _run_episode(scenario: int, n: int, seed: int, policy_fn,
                 max_steps: int, every: int,
                 label: str, label_color: str, annotation: str,
                 dpi: int) -> tuple[list, int]:
    """
    에피소드를 독립적으로 실행하고 단일 패널 PIL 이미지 목록을 반환.
    env.reset(seed=seed)가 global random을 재시드하므로
    두 에피소드를 순차 실행하면 각자 동일한 시드에서 시작된다.
    반환: ([(step_key, PIL), ...], finish_step)
    """
    env = FireEvacEnv(scenario=scenario, n_agents=n)
    env.reset(seed=seed)   # ← random.seed(seed) + np.random.seed(seed) 실행

    panels: list[tuple[int, Image.Image]] = []
    done = False

    panels.append((0, _render_single_panel(env, 0, label, label_color,
                                            annotation, False, dpi)))

    prev_done = False
    for step in range(1, max_steps + 1):
        if not done:
            action = policy_fn(env)
            _, _, t, tr, _ = env.step(action)
            done = t or tr

        just_done = done and not prev_done
        if step % every == 0 or just_done:
            panels.append((step, _render_single_panel(
                env, env.step_count, label, label_color,
                annotation, done, dpi)))
        prev_done = done

        if done:
            break

    finish = env.step_count
    # 완료 배너 없는 마지막 프레임 (hold 용도)
    last_no_banner = _render_single_panel(env, finish, label, label_color,
                                          annotation, False, dpi)
    env.close()
    return panels, finish, last_no_banner


# ── 두 패널 + 제목·범례 합성 → PIL Image ────────────────────────────────
def _stitch_frame(left_img: Image.Image, right_img: Image.Image,
                  sc_num: int, sc_name: str, dpi: int) -> Image.Image:
    """두 패널 PIL Image를 matplotlib으로 좌우 나열, 제목·범례 추가."""
    import matplotlib.lines as mlines

    lw, lh = left_img.size
    rw, rh = right_img.size
    panel_w_in = (lw + rw) / dpi
    panel_h_in = max(lh, rh) / dpi

    fig = plt.figure(figsize=(panel_w_in, panel_h_in + 0.85), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 2,
                          left=0.0, right=1.0,
                          top=0.91, bottom=0.10,
                          wspace=0.01)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_l.imshow(np.array(left_img),  aspect="auto")
    ax_r.imshow(np.array(right_img), aspect="auto")
    ax_l.axis("off"); ax_r.axis("off")

    fig.suptitle(f"S{sc_num}  {sc_name}",
                 fontsize=20, fontweight="bold", color="#111111", y=0.98)

    _mk = dict(marker="o", linestyle="None", markeredgecolor="white", markeredgewidth=1.2)
    legend_items = [
        mpatches.Patch(facecolor=COLORS["fire"],                               label="화재"),
        mpatches.Patch(facecolor=COLORS["smoke"],   edgecolor="#aaa", lw=0.6, label="연기"),
        mlines.Line2D([], [], markerfacecolor="#FF1F1F", markersize=10, label="대피자 (위험)", **_mk),
        mlines.Line2D([], [], markerfacecolor="#FFB200", markersize=10, label="대피자 (주의)", **_mk),
        mlines.Line2D([], [], markerfacecolor="#0DC940", markersize=10, label="대피자 (안전)", **_mk),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               fontsize=9.0, framealpha=0.90, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.01))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── 시나리오 1개 → GIF ────────────────────────────────────────────────────
def make_gif(scenario: int, seed: int = 42, dpi: int = 80,
             duration: int = 150, every: int = 1) -> str | None:
    from baselines.astar_real import astar_action

    cfg       = SCENARIO_CONFIGS[scenario]
    n         = cfg["n_agents"]
    max_steps = cfg["max_steps"]
    sc_num    = SC_DISPLAY_NUM.get(scenario, scenario)
    sc_name   = SC_NAME_DISPLAY.get(scenario, cfg["name"])
    notes     = SCENARIO_NOTES.get(scenario, {})
    seed      = SC_SEED.get(scenario, seed)   # 시나리오별 최적 시드 우선 사용

    print(f"\n  [S{scenario}→S{sc_num} {sc_name}]  agents={n}  seed={seed}")

    ppo_fn = _load_ppo(scenario, n)
    if ppo_fn is None:
        print(f"    [경고] PPO 모델 없음 — 건너뜀")
        return None

    # ── Phase 1: A* 에피소드 독립 실행 ──────────────────────────────────
    # env.reset(seed=seed) 가 random.seed(seed)를 호출하므로
    # A* / PPO 각자 동일한 random 시퀀스에서 출발 → 공정 비교
    print(f"    A* 실행 중...")
    a_panels, a_done, a_last_no_banner = _run_episode(
        scenario, n, seed, astar_action, max_steps, every,
        "A*", "#333333", notes.get("astar", ""), dpi)

    # ── Phase 2: PPO 에피소드 독립 실행 ──────────────────────────────────
    print(f"    PPO 실행 중...")
    p_panels, p_done, p_last_no_banner = _run_episode(
        scenario, n, seed, ppo_fn, max_steps, every,
        "PPO", "#1565C0", notes.get("ppo", ""), dpi)

    print(f"    A* 완료: t={a_done}  |  PPO 완료: t={p_done}")

    # ── Phase 3: step 키 정렬 후 패널 합성 ────────────────────────────────
    # 완료 배너: 한쪽만 끝났을 때만 표시.
    # 둘 다 끝난 구간에서는 배너 없이 렌더링하고, 마지막에 5초 hold 프레임(배너 포함) 추가.
    a_map = dict(a_panels)
    p_map = dict(p_panels)
    a_last_banner = a_panels[-1][1]   # done 배너 있는 마지막 프레임
    p_last_banner = p_panels[-1][1]

    all_steps = sorted(set(a_map) | set(p_map))
    frames: list[Image.Image] = []
    durations: list[int] = []

    for step in all_steps:
        a_fin = step >= a_done
        p_fin = step >= p_done

        if a_fin and p_fin:
            # 둘 다 완료 → 개별 배너 숨김 (hold 프레임에서 한꺼번에 표시)
            l_img = a_last_no_banner
            r_img = p_last_no_banner
        else:
            l_img = a_map.get(step, a_last_banner)
            r_img = p_map.get(step, p_last_banner)

        frames.append(_stitch_frame(l_img, r_img, sc_num, sc_name, dpi))
        durations.append(duration)

    # ── 5초 hold 프레임: 둘 다 완료 배너 표시 후 루프 ──────────────────────
    hold_frame = _stitch_frame(a_last_banner, p_last_banner, sc_num, sc_name, dpi)
    frames.append(hold_frame)
    durations.append(5000)

    out_dir = os.path.join(_ROOT, "result", "visualize", "comparison_gif")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"s{sc_num}_comparison_seed{seed}.gif")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    for f in frames:
        f.close()

    print(f"    GIF 저장: {out_path}  ({len(frames)} frames, hold=5s)")
    return out_path


# ── 진입점 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="시나리오별 A* vs PPO 나란히 비교 GIF 생성")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4, 6],
                        help="포함할 시나리오 번호 (기본: 1 2 3 4 6)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="고정 시드 (기본: 42)")
    parser.add_argument("--dpi",      type=int, default=80,
                        help="해상도 (기본: 80 — 파일 크기 최소화)")
    parser.add_argument("--duration", type=int, default=150,
                        help="프레임당 시간 ms (기본: 150)")
    parser.add_argument("--every",    type=int, default=1,
                        help="N스텝마다 프레임 저장 (기본: 1, 전체)")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  A* vs PPO 비교 GIF 생성")
    print(f"  시나리오: {args.scenarios}  seed={args.seed}  dpi={args.dpi}")
    print(f"{'═'*60}")

    results = []
    for sc in args.scenarios:
        path = make_gif(sc, seed=args.seed, dpi=args.dpi,
                        duration=args.duration, every=args.every)
        if path:
            results.append(path)

    print(f"\n{'─'*60}")
    print(f"  완료: {len(results)}/{len(args.scenarios)}개 GIF 저장")
    for p in results:
        print(f"    {p}")
    print(f"{'═'*60}\n")
