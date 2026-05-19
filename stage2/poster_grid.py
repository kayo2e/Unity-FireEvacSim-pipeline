"""
poster_grid.py — 포스터용 A* vs PPO 2행×N열 정성 비교 그리드
=============================================================
각 시나리오의 "핵심 장면"(에피소드 capture_frac 지점)에서
A*(위 행) vs PPO(아래 행) 라우팅 유도등·에이전트를 나란히 보여준다.

실행:
    cd stage2
    python poster_grid.py
    python poster_grid.py --seed 42
    python poster_grid.py --scenarios 1 2 3 4 6 --capture-frac 0.35
    python poster_grid.py --seed 77 --dpi 200

저장: result/visualize/poster_grid_seed{SEED}.png
"""

import io, os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 한글 폰트 설정 (NanumGothic 우선, 없으면 DejaVu Sans fallback)
import matplotlib.font_manager as _fm
_korean_fonts = [f.name for f in _fm.fontManager.ttflist
                 if any(k in f.name for k in ("Nanum", "Malgun", "Gothic", "NotoSansCJK"))]
if _korean_fonts:
    matplotlib.rcParams["font.family"] = _korean_fonts[0]
matplotlib.rcParams["axes.unicode_minus"] = False

from env_core import (
    FireEvacEnv, SCENARIO_CONFIGS,
    HALL, WALL, EXIT, ROOM,
    N, S, E, W,
    EXIT_A_POS, EXIT_B_POS,
    BASE_GRID,
)

# 실제 건물에서 벽이지만 BASE_GRID에 ROOM/HALL로 등록된 셀
# → 시뮬레이션 동작은 그대로, 시각화만 벽 색으로 표시
_VISUAL_WALL_CELLS = frozenset(
    (r, c)
    for r in range(10, 28)
    for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c)
    for r in range(0, 10)
    for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 시나리오별 기본 캡처 스텝 (capture_frac 로 덮어쓸 수 있음) ──
CAPTURE_STEPS = {1: 60, 2: 45, 3: 40, 4: 30, 5: 40, 6: 40}

# 경로 전용 포스터: PPO 성능이 두드러지는 스텝
# 화재가 충분히 확산된 후 → A* 약점(쏠림·지연) vs PPO 강점(분산·선제) 최대화
PPO_SHOWCASE_STEPS = {
    1: 70,   # S1: 기본 탈출 — 화재 확산 중, 에이전트 분포 차이 확인
    2: 90,   # S2: EXIT A 위협 확산 — A* EXIT B 병목 vs PPO 분산
    3: 28,   # S3: 진입로 차단 초반 — PPO 에이전트 모두 안전거리 유지 (seed42 기준 전원 녹색)
    4: 80,   # S4: 양방향 위협 최고조 — PPO 균형 분배 vs A* 단일 회피
    5: 80,   # S5: EXIT B 위협
    6: 50,   # S6: OOD 중앙 차단 직후 — PPO 우회 경로 vs A* 직진 시도
}

# 포스터 표시용 시나리오 이름/번호 오버라이드 (env_core 값과 별개)
SC_NAME_OVERRIDE = {
    4: "양방향 위협",
    5: "중앙 차단",
    6: "중앙 차단",
}
SC_DISPLAY_NUM = {6: 5}   # S6 → 포스터에서 S5로 표시

EXIT_COLOR = "#00A550"   # ISO 7010 비상구 녹색
ROUTE_COLOR = "#1E6EBB"  # 단일 경로 색 (A/B 구분 없음)

COLORS = {
    "hall":          "#EFEFEF",
    "wall":          "#1A1A1A",
    "exit_a":        EXIT_COLOR,
    "exit_b":        EXIT_COLOR,
    "room":          "#E0D8C4",
    "fire":          "#FF4500",
    "smoke":         "#BBBBBB",
    "person":        "#FF7700",
    "blocked":       "#8B0000",
    "route_a":       ROUTE_COLOR,
    "route_b":       ROUTE_COLOR,
    "route_unknown": "#888888",
}

DIR_ARROW  = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}
EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)

_EXIT_ICON_CACHE: dict = {}

def _make_exit_icon(size: int = 52) -> np.ndarray:
    """evac_sign.png를 리사이즈해서 RGBA numpy array로 반환."""
    if size in _EXIT_ICON_CACHE:
        return _EXIT_ICON_CACHE[size]
    from PIL import Image
    icon_path = os.path.join(_ROOT, "evac_sign.png")
    img = Image.open(icon_path).convert("RGBA").resize((size, size), Image.LANCZOS)
    _EXIT_ICON_CACHE[size] = np.array(img)
    return _EXIT_ICON_CACHE[size]


def _hex2rgba(h, a=1.0):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]


def _classify_exit(lmap, start_r, start_c):
    visited = set()
    r, c = start_r, start_c
    for _ in range(120):
        if (r, c) in visited:
            return None
        visited.add((r, c))
        if (r, c) in EXIT_A_SET:
            return "A"
        if (r, c) in EXIT_B_SET:
            return "B"
        if (r, c) not in lmap:
            return None
        dr, dc = DIR_ARROW[lmap[(r, c)]]
        r, c = r + dr, c + dc
    return None


def _trace_chain(lmap, start_r, start_c):
    """유도등 체인을 따라가며 출구까지의 셀 좌표 리스트와 목적지를 반환."""
    path, visited = [], set()
    r, c = start_r, start_c
    for _ in range(200):
        if (r, c) in visited:
            break
        visited.add((r, c))
        path.append((c, r))          # matplotlib (x=col, y=row)
        if (r, c) in EXIT_A_SET:
            return path, "A"
        if (r, c) in EXIT_B_SET:
            return path, "B"
        if (r, c) not in lmap:
            break
        dr, dc = DIR_ARROW[lmap[(r, c)]]
        r, c = r + dr, c + dc
    return path, None


def _draw_routing_chains(env, ax, halo: bool = True):
    """트래픽 가중 굵기로 주요 복도를 굵은 화살표 선으로 표시.
    halo=True  → 불/연기 배경 위에서 보이도록 흰 후광 추가 (존 오버레이 패널용)
    halo=False → 클린 배경 전용, 후광 없이 더 진하게 (라우팅 선 패널용)"""
    lmap = {env.light_cells[i]: int(env.light_dirs[i]) for i in range(env.n_lights)}

    edge_info: dict[tuple, list] = {}
    for (start_r, start_c) in lmap:
        path, dest = _trace_chain(lmap, start_r, start_c)
        for i in range(len(path) - 1):
            key = (*path[i], *path[i + 1])
            if key not in edge_info:
                edge_info[key] = [dest, 0]
            edge_info[key][1] += 1

    if not edge_info:
        return

    max_t = max(v[1] for v in edge_info.values())
    MAIN = {"A": ROUTE_COLOR, "B": ROUTE_COLOR}

    for (x1, y1, x2, y2), (dest, cnt) in edge_info.items():
        if dest not in MAIN:          # 출구 미연결 체인 제외
            continue
        ratio = cnt / max_t
        if ratio < 0.12:
            continue

        lw    = (1.5 + 8.5 * ratio) if halo else (1.2 + 6.8 * ratio)  # 클린 배경은 더 가늘게
        alpha = 0.50 + 0.45 * ratio  # 0.50 ~ 0.95

        if halo:
            ax.plot([x1, x2], [y1, y2], "-",
                    color="white", alpha=0.90,
                    linewidth=lw + 4.0, solid_capstyle="round", zorder=3.8)

        ax.plot([x1, x2], [y1, y2], "-",
                color=MAIN[dest], alpha=alpha,
                linewidth=lw, solid_capstyle="round", zorder=3.9)

        # 화살촉: 불/연기 배경 패널에서만 (클린 배경에서는 선만으로 충분)
        if halo and ratio > 0.28:
            ax.annotate("",
                xy=(x1 + (x2-x1)*0.68, y1 + (y2-y1)*0.68),
                xytext=(x1 + (x2-x1)*0.32, y1 + (y2-y1)*0.32),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=MAIN[dest],
                    alpha=alpha,
                    lw=max(1.0, lw * 0.45),
                    mutation_scale=10 + 10 * ratio,
                ),
                zorder=4.0,
            )


def _bfs_fire_aware(env, exits: list) -> np.ndarray:
    """화재·연기 셀을 장애물로 취급한 BFS 거리맵 계산."""
    from collections import deque
    ROWS, COLS = env.ROWS, env.COLS
    INF = float("inf")
    dist = np.full((ROWS, COLS), INF)
    q = deque()
    for r, c in exits:
        if env.fire_map[r, c] == 0:   # 출구 자체가 불타지 않은 경우만
            dist[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if env.grid[nr, nc] == WALL:
                continue
            if env.fire_map[nr, nc] > 0:   # 화재 셀 통과 금지
                continue
            if dist[nr, nc] == INF:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _draw_fire_glow(ax, env, base_zorder: int = 3, y_offset: float = 0.0):
    """화재 셀 위에 레이어드 글로우 + 불꽃 팁 scatter 오버레이.
    y_offset > 0 이면 scatter를 아래로 시프트 (시각적 강조, S1 전용).
    벽 셀: 그리드를 훼손하지 않도록 팁만 표시."""
    fire_ys, fire_xs = np.where(env.fire_map > 0)
    if len(fire_xs) == 0:
        return

    is_wall = np.array([
        env.grid[r, c] == WALL or (r, c) in _VISUAL_WALL_CELLS
        for r, c in zip(fire_ys, fire_xs)
    ])

    # ── 바닥(복도·방) 화재: 글로우 + 팁 ──
    frs = fire_ys[~is_wall].astype(float) + y_offset
    fcs = fire_xs[~is_wall].astype(float)
    if len(fcs) > 0:
        z = base_zorder
        ax.scatter(fcs, frs, s=320, c="#CC1000", alpha=0.09, linewidths=0, zorder=z)
        ax.scatter(fcs, frs, s=180, c="#FF5000", alpha=0.18, linewidths=0, zorder=z)
        ax.scatter(fcs, frs, s=80,  c="#FF9500", alpha=0.45, linewidths=0, zorder=z)
        ax.scatter(fcs, frs, s=28,  c="#FFE040", alpha=0.68, linewidths=0, zorder=z)
        ax.scatter(fcs, frs - 0.48, s=120, c="#FF6A00", alpha=0.40,
                   marker="^", linewidths=0, zorder=z + 0.5)

    # ── 벽 화재: 팁만 위로 솟음 (그리드 유지) ──
    wrs = fire_ys[is_wall].astype(float) + y_offset
    wcs = fire_xs[is_wall].astype(float)
    if len(wcs) > 0:
        z = base_zorder + 1
        ax.scatter(wcs, wrs - 0.40, s=140, c="#FF6A00", alpha=0.55,
                   marker="^", linewidths=0, zorder=z)
        ax.scatter(wcs, wrs - 0.58, s=55,  c="#FFD040", alpha=0.65,
                   marker="^", linewidths=0, zorder=z)


def render_routes_panel(env: FireEvacEnv, ax: plt.Axes, show_agents: bool = False):
    """라우팅 선 전용 패널 — 화재 회피 BFS 경로 흐름을 트래픽 가중 선으로 표시.
    화재·연기 셀을 장애물로 취급하므로 실제 대피 가능 경로만 표시된다.
    show_agents=True 시 에이전트 현재 위치도 표시."""
    ROWS, COLS = env.ROWS, env.COLS

    # 클린 플로어플랜 배경 (화재는 연하게 표시)
    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = _hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                img[r, c] = _hex2rgba(COLORS["exit_a"])
            elif (r, c) in EXIT_B_SET:
                img[r, c] = _hex2rgba(COLORS["exit_b"])
            elif cell == ROOM:
                img[r, c] = _hex2rgba(COLORS["room"])
            else:
                img[r, c] = _hex2rgba(COLORS["hall"])
    # 화재 영역 연하게 오버레이 (경로가 화재를 피한다는 걸 맥락으로 보여줌)
    for r, c in zip(*np.where(env.fire_map > 0)):
        img[r, c] = _hex2rgba(COLORS["fire"], 0.18)
    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")
    _draw_fire_glow(ax, env, base_zorder=3)

    # 출구 비상구 아이콘 (그룹 중심 1개씩)
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    _icon = _make_exit_icon(size=52)
    _cell_px = ax.get_window_extent().width / max(env.COLS, 1) if ax.get_window_extent().width > 0 else 10
    _zoom = max(0.28, min(0.55, _cell_px * 1.6 / 52))
    for exit_pos in (EXIT_A_POS, EXIT_B_POS):
        if exit_pos:
            er = sum(r for r, c in exit_pos) / len(exit_pos)
            ec = sum(c for r, c in exit_pos) / len(exit_pos)
            ib = OffsetImage(_icon, zoom=_zoom)
            ib.image.axes = ax
            ab = AnnotationBbox(ib, (ec, er), frameon=False, zorder=9)
            ax.add_artist(ab)

    # 화재 회피 BFS 거리맵
    da = _bfs_fire_aware(env, list(EXIT_A_POS))
    db = _bfs_fire_aware(env, list(EXIT_B_POS))
    INF = float("inf")

    edge_info: dict[tuple, list] = {}  # (x1,y1,x2,y2) → [dest, count]

    for r in range(ROWS):
        for c in range(COLS):
            if env.grid[r, c] == WALL or (r, c) in _VISUAL_WALL_CELLS:
                continue
            if env.fire_map[r, c] > 0:
                continue
            if (r, c) in EXIT_A_SET or (r, c) in EXIT_B_SET:
                continue
            d_a = da[r, c] if da[r, c] < INF else INF
            d_b = db[r, c] if db[r, c] < INF else INF
            if d_a == INF and d_b == INF:
                continue
            dest     = "A" if d_a <= d_b else "B"
            dist_map = da if dest == "A" else db

            visited_path: set = set()
            cr, cc = r, c
            for _ in range(300):
                if (cr, cc) in visited_path:
                    break
                visited_path.add((cr, cc))
                if (cr, cc) in EXIT_A_SET or (cr, cc) in EXIT_B_SET:
                    break
                best_d = dist_map[cr, cc]
                nr2, nc2 = None, None
                for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    tr, tc = cr + dr2, cc + dc2
                    if (0 <= tr < ROWS and 0 <= tc < COLS
                            and env.grid[tr, tc] != WALL
                            and env.fire_map[tr, tc] == 0):
                        if dist_map[tr, tc] < best_d:
                            best_d = dist_map[tr, tc]
                            nr2, nc2 = tr, tc
                if nr2 is None:
                    break
                key = (cc, cr, nc2, nr2)
                if key not in edge_info:
                    edge_info[key] = [dest, 0]
                edge_info[key][1] += 1
                cr, cc = nr2, nc2

    if not edge_info:
        ax.set_xlim(-0.5, COLS - 0.5); ax.set_ylim(ROWS - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        return

    max_t = max(v[1] for v in edge_info.values())
    MAIN = {"A": ROUTE_COLOR, "B": ROUTE_COLOR}

    for (x1, y1, x2, y2), (dest, cnt) in edge_info.items():
        if dest not in MAIN:
            continue
        ratio = cnt / max_t
        if ratio < 0.04:
            continue
        pass  # 경로선 숨김 — 범례만 유지, 그림에는 미표시

    if show_agents:
        import matplotlib.colors as _mc
        # 위험도 컬러맵: 빨강(화재 인접) → 노랑 → 초록(안전)
        _danger_cmap = _mc.LinearSegmentedColormap.from_list(
            "danger", [(1.0, 0.12, 0.12), (1.0, 0.70, 0.0), (0.05, 0.78, 0.25)], N=64)

        if env.people_data:
            fire_cells = np.argwhere(env.fire_map > 0)
            agents_with_danger = []
            for p in env.people_data:
                pr, pc = p["pos"]
                if env.grid[pr, pc] == WALL or (pr, pc) in _VISUAL_WALL_CELLS:
                    continue
                if len(fire_cells) > 0:
                    dists = (np.abs(fire_cells[:, 0] - pr)
                             + np.abs(fire_cells[:, 1] - pc)).astype(float)
                    min_d = float(dists.min())
                else:
                    min_d = 99.0
                # 0~6 스텝 → 0.0(위험)~1.0(안전)
                t = min(min_d / 6.0, 1.0)
                agents_with_danger.append((t, pr, pc))
            # 같은 셀에 여러 에이전트 → 가장 위험한 것만 대표로 표시
            cell_rep: dict = {}
            for t, pr, pc in agents_with_danger:
                key = (pr, pc)
                if key not in cell_rep or t < cell_rep[key][0]:
                    cell_rep[key] = (t, pr, pc)
            deduped = sorted(cell_rep.values(), key=lambda x: -x[0])  # 안전→위험 순(위험이 맨 위)
            for t, pr, pc in deduped:
                agent_color = _danger_cmap(t)
                ax.plot(pc, pr, "o", color=agent_color,
                        markersize=13.0, markeredgecolor="white",
                        markeredgewidth=1.5, zorder=5, alpha=0.92)

        # 생존 현황 텍스트 박스 — people_data 유무와 무관하게 항상 표시
        survived  = getattr(env, "escaped", 0)
        dead      = getattr(env, "dead", 0)
        remaining = env.n_agents - survived - dead
        if remaining == 0:
            stats_str = (f"탈출 완료 t={env.step_count}"
                         f"  |  탈출 {survived}  |  잔류 {remaining}  |  사망 {dead}")
            box_edge  = "#1565C0"
            box_face  = "#e8f0fb"
        else:
            stats_str = f"탈출 {survived}  |  잔류 {remaining}  |  사망 {dead}"
            box_edge  = "#aaaaaa"
            box_face  = "white"
        ax.text(0.99, 1.02, stats_str,
                transform=ax.transAxes,
                fontsize=7.0, ha="right", va="bottom", fontweight="bold",
                color="#111111", clip_on=False,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=box_face, alpha=0.92,
                          edgecolor=box_edge, linewidth=1.2),
                zorder=6)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def render_panel(env: FireEvacEnv, ax: plt.Axes, actual_step: int):
    """포스터용 단일 패널 렌더링.
    레이어 순서: 건물 → 연기/화재 → 라우팅 체인(굵은 화살표 선) → 에이전트"""
    ROWS, COLS = env.ROWS, env.COLS

    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = env.grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = _hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit_a"]
                img[r, c] = _hex2rgba(col)
            elif (r, c) in EXIT_B_SET:
                col = COLORS["blocked"] if (r, c) in env.blocked_exits else COLORS["exit_b"]
                img[r, c] = _hex2rgba(col)
            elif cell == ROOM:
                img[r, c] = _hex2rgba(COLORS["room"])
            else:
                img[r, c] = _hex2rgba(COLORS["hall"])

    for r, c in zip(*np.where(env.smoke_map > 0)):
        img[r, c] = _hex2rgba(COLORS["smoke"], 0.45)
    for r, c in zip(*np.where(env.fire_map > 0)):
        img[r, c] = _hex2rgba(COLORS["fire"], 0.92)  # 진하게 — 경로선 아래로도 선명히 보임

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")
    _fire_y_off = 0.42 if getattr(env, "scenario", 0) == 1 else 0.0
    _draw_fire_glow(ax, env, base_zorder=3, y_offset=_fire_y_off)

    # 라우팅 존 오버레이: BFS 거리맵으로 모든 복도 셀을 출구 권역으로 채움
    da = env._dist_to_exit_A   # shape (ROWS, COLS), inf = unreachable
    db = env._dist_to_exit_B
    INF = 1e9
    for r in range(ROWS):
        for c in range(COLS):
            if env.grid[r, c] == WALL or (r, c) in _VISUAL_WALL_CELLS:
                continue
            if (r, c) in EXIT_A_SET or (r, c) in EXIT_B_SET:
                continue
            d_a = da[r, c] if da[r, c] < INF else INF
            d_b = db[r, c] if db[r, c] < INF else INF
            if d_a == INF and d_b == INF:
                continue
            if d_a <= d_b:
                color, alpha = COLORS["route_a"], 0.22
            else:
                color, alpha = COLORS["route_b"], 0.22
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color, alpha=alpha, zorder=2, linewidth=0,
            ))

    # 에이전트
    for p in env.people_data:
        pr, pc = p["pos"]
        ax.plot(pc, pr, "o", color=COLORS["person"],
                markersize=3.0, markeredgecolor="white",
                markeredgewidth=0.4, zorder=5, alpha=0.85)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(COLS - 0.8, ROWS - 0.8,
            f"t={actual_step}", fontsize=6, color="#888888",
            ha="right", va="bottom", zorder=5)


# ── PPO 로드 ──────────────────────────────────────────
def _load_ppo(scenario: int, n_agents: int, ppo_path: str | None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    _path = ppo_path
    if _path is None:
        suffix = "20ppl" if n_agents <= 20 else "40ppl"
        for sub in ("ppo", "recurrent_ppo"):
            for fname in (f"s{scenario}_best.zip",
                          f"fire_evac_model_{suffix}.zip"):
                cand = os.path.join(_ROOT, "model", sub, fname)
                if os.path.exists(cand):
                    _path = cand
                    break
            if _path:
                break

    if not (_path and os.path.exists(_path)):
        return None

    model = PPO.load(_path)
    vecnorm_pkl = _path.replace(".zip", "_vecnorm.pkl")
    _vn = None
    if os.path.exists(vecnorm_pkl):
        tmp = DummyVecEnv([lambda: FireEvacEnv(scenario=scenario,
                                               n_agents=n_agents)])
        _vn = VecNormalize.load(vecnorm_pkl, tmp)
        _vn.training = False
        _vn.norm_reward = False

    def _policy(env):
        raw = env._get_obs()
        obs = _vn.normalize_obs(np.array([raw]))[0] if _vn else raw
        action, _ = model.predict(obs, deterministic=True)
        return action

    print(f"    PPO 로드: {_path}")
    return _policy


# ── 한 시나리오·한 모델 실행 → 목표 스텝에서 env 스냅샷 ──
def _run_to_step(scenario: int, n_agents: int, seed: int,
                 policy_fn, target_step: int, cfg: dict) -> tuple:
    """
    target_step 까지 환경을 실행하고 그 시점의 env 와 실제 스텝을 반환.
    에피소드가 target_step 전에 끝나면 마지막 상태 반환.
    """
    env = FireEvacEnv(scenario=scenario, n_agents=n_agents)
    env.reset(seed=seed)

    actual_step = 0
    for _ in range(min(target_step, cfg["max_steps"])):
        action = policy_fn(env)
        _, _, term, trunc, _ = env.step(action)
        actual_step = env.step_count
        if term or trunc:
            break

    return env, actual_step


# ── 메인 ─────────────────────────────────────────────
def build_poster_grid(
    scenarios:    list  = None,
    seed:         int   = 42,
    capture_frac: float = None,
    ppo_path:     str   = None,
    dpi:          int   = 150,
    out_path:     str   = None,
    show_routes:  bool  = False,   # True → 각 시나리오마다 [존오버레이 | 라우팅선] 쌍
    routes_only:  bool  = False,   # True → 경로 패널만 표시 (PPO showcase 스텝, 큰 글씨)
):
    if scenarios is None:
        scenarios = [1, 2, 3, 4, 6]

    from baselines.astar_real import astar_action

    n_sc = len(scenarios)

    # ── 레이아웃 파라미터 분기 ──────────────────────────────
    if routes_only:
        p_per    = 1
        panel_w  = 1.6
        fig_h    = 6.2
        wspace   = 0.02
        fs_title = 12.0
        fs_row   = 14     # 기준값; 실제 표시는 ×4
        fs_step  = 9
        fs_leg   = 15.0
        top_m    = 0.87
        bottom_m = 0.06   # 하단 여백 최소화 (범례 오른쪽 이동)
        step_src = PPO_SHOWCASE_STEPS
        label_w  = 0.55   # 가로 행 레이블을 위한 추가 폭 (inches)
        leg_w    = 2.2    # 오른쪽 범례 영역 폭 (inches)
    elif show_routes:
        p_per    = 2
        panel_w  = 2.2
        fig_h    = 5.8
        wspace   = 0.18
        fs_title = 8.0
        fs_row   = 11
        fs_step  = 6
        fs_leg   = 7.5
        top_m    = 0.88
        bottom_m = 0.12
        step_src = CAPTURE_STEPS
        label_w  = 0.4
    else:
        p_per    = 1
        panel_w  = 2.6
        fig_h    = 5.8
        wspace   = 0.04
        fs_title = 8.0
        fs_row   = 11
        fs_step  = 6
        fs_leg   = 7.5
        top_m    = 0.88
        bottom_m = 0.12
        step_src = CAPTURE_STEPS
        label_w  = 0.4

    n_cols  = n_sc * p_per
    _leg_w  = leg_w if routes_only else 0.0
    fig_w   = panel_w * n_cols + 0.5 + label_w + _leg_w
    left_m  = label_w / fig_w               # 행 레이블 공간 비율
    right_m = 1.0 - _leg_w / fig_w if routes_only else 0.99  # 패널 우측 경계

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        2, n_cols,
        left=left_m, right=right_m,
        top=top_m, bottom=bottom_m,
        hspace=0.22,
        wspace=wspace,
    )

    row_labels = ["A*", "PPO"]
    row_colors = ["#333333", "#1565C0"]
    ppo_ax_list = []   # PPO 행 배경 박스용 축 수집

    for col_idx, sc in enumerate(scenarios):
        cfg     = SCENARIO_CONFIGS[sc]
        n       = cfg["n_agents"]
        sc_name = SC_NAME_OVERRIDE.get(sc, cfg["name"])

        if capture_frac is not None:
            target = int(cfg["max_steps"] * capture_frac)
        else:
            target = step_src.get(sc, int(cfg["max_steps"] * 0.35))

        print(f"\n  S{sc} [{sc_name}]  agents={n}  target_step={target}")

        ppo_fn = _load_ppo(sc, n, ppo_path)
        if ppo_fn is None:
            print(f"    [경고] PPO 모델 없음 — PPO 패널 건너뜀")

        policies = [astar_action, ppo_fn]

        ax_top       = None
        ax_top_route = None

        for row_idx, (label, policy_fn, color) in enumerate(
                zip(row_labels, policies, row_colors)):

            zone_col  = col_idx * p_per
            route_col = col_idx * p_per + 1

            ax = fig.add_subplot(gs[row_idx, zone_col])
            if row_idx == 0:
                ax_top = ax

            if policy_fn is None:
                ax.text(0.5, 0.5, "모델 없음",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_facecolor("#F5F5F5")
                if show_routes:
                    ax2 = fig.add_subplot(gs[row_idx, route_col])
                    ax2.set_xticks([]); ax2.set_yticks([])
                    ax2.set_facecolor("#F5F5F5")
                    if row_idx == 0:
                        ax_top_route = ax2
            else:
                print(f"    [{label}] 실행 중...")
                env, actual = _run_to_step(sc, n, seed, policy_fn, target, cfg)

                if routes_only:
                    # 경로 전용 모드: 라우팅 선 패널 + 에이전트 위치
                    render_routes_panel(env, ax, show_agents=True)
                    # 스텝 표시 (오른쪽 하단) — 조기 완료 시 ★완료 표시
                    ROWS_e, COLS_e = env.ROWS, env.COLS
                    if actual < target:
                        step_label = f"t={actual}  ★완료"
                        step_color = "#1565C0"
                    else:
                        step_label = f"t={actual}"
                        step_color = "#888888"
                    ax.text(COLS_e - 0.8, ROWS_e - 0.8,
                            step_label, fontsize=fs_step,
                            color=step_color, fontweight="bold" if actual < target else "normal",
                            ha="right", va="bottom", zorder=6)
                elif show_routes:
                    render_panel(env, ax, actual)
                    ax2 = fig.add_subplot(gs[row_idx, route_col])
                    render_routes_panel(env, ax2)
                    if row_idx == 0:
                        ax_top_route = ax2
                else:
                    render_panel(env, ax, actual)

                env.close()

            # PPO 축 수집 (배경 박스용)
            if row_idx == 1:
                ppo_ax_list.append(ax)

            # 왼쪽 첫 열에만 행 레이블 (가로, 4배 크기)
            if col_idx == 0:
                ax.text(-0.06, 0.5, label,
                        transform=ax.transAxes,
                        fontsize=25,
                        fontweight="bold",
                        color=color,
                        ha="right", va="center",
                        rotation=0, clip_on=False)

        # 열 헤더
        sc_disp = SC_DISPLAY_NUM.get(sc, sc)
        if ax_top is not None:
            if routes_only:
                title_str = f"S{sc_disp}  {sc_name}\n(t = {step_src.get(sc, '?')})"
            else:
                title_str = f"S{sc_disp}  {sc_name}" + ("  (상황)" if show_routes else "")
            ax_top.set_title(title_str, fontsize=fs_title,
                             fontweight="bold", pad=5, color="#222222")
        if show_routes and ax_top_route is not None:
            ax_top_route.set_title(
                "경로", fontsize=fs_title, fontweight="bold",
                pad=4, color="#555555",
            )

    # ── PPO 행 배경 강조 박스 ────────────────────────────
    if ppo_ax_list:
        _pos  = [a.get_position() for a in ppo_ax_list]
        _x0   = min(p.x0 for p in _pos)
        _y0   = min(p.y0 for p in _pos)
        _x1   = max(p.x1 for p in _pos)
        _y1   = max(p.y1 for p in _pos)
        _px     = 0.007
        _py_bot = 0.02        # 하단 여백
        _py_top = 0.005       # 상단 여백 (A* 행과 겹침 방지)
        _lpad   = left_m + 0.07  # 좌측 확장 (PPO 레이블 + S1 stats 박스 포함)
        fig.add_artist(mpatches.FancyBboxPatch(
            (_x0 - _lpad, _y0 - _py_bot),
            (_x1 - _x0 + _lpad + _px),
            (_y1 - _y0 + _py_top + _py_bot),
            boxstyle="round,pad=0.003",
            transform=fig.transFigure,
            facecolor="#dae3f3",
            edgecolor="none", linewidth=0,
            zorder=0, clip_on=False,
        ))
        for _a in ppo_ax_list:
            for _sp in _a.spines.values():
                _sp.set_visible(False)

    # ── 범례 ──────────────────────────────────────────────
    # 비상구 아이콘 범례 핸들러
    from matplotlib.legend_handler import HandlerBase as _HB
    class _IconHandler(_HB):
        def __init__(self, icon_arr): self._icon = icon_arr; super().__init__()
        def create_artists(self, legend, handle, xd, yd, w, h, fs, trans):
            from matplotlib.offsetbox import AnnotationBbox, OffsetImage
            zoom = h / self._icon.shape[0] * 1.4
            ib = OffsetImage(self._icon, zoom=zoom)
            ab = AnnotationBbox(ib, (xd + w*0.5, yd + h*0.5),
                                xycoords=trans, frameon=False)
            return [ab]

    _exit_handle = mpatches.Patch(facecolor=EXIT_COLOR, label="출구 (비상구)")
    _icon_handler_map = {_exit_handle: _IconHandler(_make_exit_icon(size=52))}

    _mk = dict(marker="o", linestyle="None", markeredgecolor="white", markeredgewidth=1.2)
    _route_handle = mlines.Line2D([], [], color=ROUTE_COLOR, linestyle="--",
                                  linewidth=2.5, label="대피 경로")
    if routes_only:
        legend_items = [
            _route_handle,
            _exit_handle,
            mpatches.Patch(facecolor=COLORS["fire"], label="화재"),
            mlines.Line2D([], [], markerfacecolor="#FF1F1F", markersize=11, label="대피자 (위험)", **_mk),
            mlines.Line2D([], [], markerfacecolor="#FFB200", markersize=11, label="대피자 (주의)", **_mk),
            mlines.Line2D([], [], markerfacecolor="#0DC940", markersize=11, label="대피자 (안전)", **_mk),
        ]
        ncol_leg = 2   # 3행×2열
        # 오른쪽 범례 영역 중앙 (figure 좌표)
        _leg_cx = right_m + _leg_w / fig_w * 0.5
    else:
        legend_items = [
            mpatches.Patch(facecolor=ROUTE_COLOR,       label="대피 경로"),
            _exit_handle,
            mpatches.Patch(facecolor=COLORS["fire"],    label="화재"),
            mpatches.Patch(facecolor=COLORS["smoke"],   label="연기"),
            mpatches.Patch(facecolor=COLORS["person"],  label="대피자"),
        ]
        ncol_leg = len(legend_items)
        _leg_cx  = 0.5

    if routes_only:
        fig.legend(
            handles=legend_items,
            handler_map=_icon_handler_map,
            loc="center left",          # 범례 왼쪽 끝을 anchor에 맞춤 → 패널과 겹침 없음
            ncol=ncol_leg,
            fontsize=fs_leg,
            framealpha=0.95,
            bbox_to_anchor=(right_m + 0.01, 0.5),  # 패널 우측 경계 바로 오른쪽
        )
    else:
        fig.legend(
            handles=legend_items,
            handler_map=_icon_handler_map,
            loc="lower center",
            ncol=ncol_leg,
            fontsize=fs_leg,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.01),
        )

    if out_path is None:
        out_dir  = os.path.join(_ROOT, "result", "visualize")
        os.makedirs(out_dir, exist_ok=True)
        sc_tag   = "".join(str(s) for s in scenarios)
        if routes_only:
            suffix = "_routes_only"
        elif show_routes:
            suffix = "_routes"
        else:
            suffix = "_lines"
        out_path = os.path.join(out_dir,
                                f"poster_grid_s{sc_tag}_seed{seed}{suffix}.png")

    fig.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"\n  저장 완료: {out_path}")
    return out_path


# ── 줌인 비교 패널 ────────────────────────────────────
def build_zoom_comparison(
    scenario:  int   = 2,
    seed:      int   = 42,
    step:      int   = 90,
    zoom_rows: tuple = (15, 40),
    zoom_cols: tuple = (0, 25),
    dpi:       int   = 150,
    out_path:  str   = None,
    ppo_path:  str   = None,
):
    """A* vs PPO 특정 구역 줌인 나란히 비교 패널."""
    from baselines.astar_real import astar_action

    cfg   = SCENARIO_CONFIGS[scenario]
    n     = cfg["n_agents"]
    r0, r1 = zoom_rows
    c0, c1 = zoom_cols

    # 줌 종횡비에 맞춰 패널 크기 계산
    panel_w = 4.0
    panel_h = panel_w * (r1 - r0) / max(c1 - c0, 1)
    fig_w   = panel_w * 2 + 0.5
    fig_h   = panel_h + 1.0    # 타이틀 여백

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98,
                          top=0.88, bottom=0.04, wspace=0.06)

    ppo_fn   = _load_ppo(scenario, n, ppo_path)
    policies = [astar_action, ppo_fn]
    labels   = ["A*", "PPO"]
    colors   = ["#333333", "#1565C0"]

    for i, (label, policy_fn, color) in enumerate(zip(labels, policies, colors)):
        ax = fig.add_subplot(gs[0, i])

        if policy_fn is None:
            ax.text(0.5, 0.5, "모델 없음", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        print(f"  [{label}] S{scenario} 실행 중 (step={step})...")
        env, actual = _run_to_step(scenario, n, seed, policy_fn, step, cfg)
        render_routes_panel(env, ax, show_agents=True)

        # 줌 적용
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(r1 - 0.5, r0 - 0.5)

        # PPO 패널 배경 강조
        if i == 1:
            for sp in ax.spines.values():
                sp.set_edgecolor("#4472c4")
                sp.set_linewidth(2.5)

        ax.set_title(label, fontsize=24, fontweight="bold",
                     color=color, pad=7)
        env.close()

    sc_name = SC_NAME_OVERRIDE.get(scenario, cfg["name"])
    fig.suptitle(f"S{SC_DISPLAY_NUM.get(scenario, scenario)}  {sc_name}  (t = {step})",
                 fontsize=13, fontweight="bold", color="#222222", y=0.97)

    if out_path is None:
        out_dir  = os.path.join(_ROOT, "result", "visualize")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f"zoom_s{scenario}_seed{seed}_t{step}.png")

    fig.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"\n  저장 완료: {out_path}")
    return out_path


# ── 진입점 ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="포스터용 A* vs PPO 2×N 정성 비교 그리드 생성")
    parser.add_argument("--scenarios",     type=int, nargs="+",
                        default=[1, 2, 3, 4, 6],
                        help="포함할 시나리오 번호 (기본: 1 2 3 4 6)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="고정 시드 (기본: 42)")
    parser.add_argument("--capture-frac",  type=float, default=None,
                        help="에피소드 캡처 지점 비율 (기본: 시나리오별 고정값)")
    parser.add_argument("--ppo-path",      type=str,   default=None,
                        help="PPO 모델 .zip 경로 (없으면 자동 탐색)")
    parser.add_argument("--dpi",           type=int,   default=150,
                        help="출력 해상도 (기본: 150)")
    parser.add_argument("--out",           type=str,   default=None,
                        help="출력 파일 경로 (없으면 result/visualize/ 에 자동 저장)")
    parser.add_argument("--show-routes",   action="store_true",
                        help="각 패널 옆에 라우팅 선 전용 패널을 나란히 추가")
    parser.add_argument("--routes-only",   action="store_true",
                        help="경로 패널만 표시 (PPO showcase 스텝, 큰 글씨, 에이전트 위치 포함)")
    parser.add_argument("--zoom",          action="store_true",
                        help="줌인 비교 패널 생성 (--zoom-scenario, --zoom-step 참조)")
    parser.add_argument("--zoom-scenario", type=int, default=2,
                        help="줌인 대상 시나리오 (기본: 2)")
    parser.add_argument("--zoom-step",     type=int, default=90,
                        help="줌인 캡처 스텝 (기본: 90)")
    parser.add_argument("--zoom-rows",     type=int, nargs=2, default=[15, 40],
                        metavar=("R0", "R1"), help="줌인 행 범위 (기본: 15 40)")
    parser.add_argument("--zoom-cols",     type=int, nargs=2, default=[0, 25],
                        metavar=("C0", "C1"), help="줌인 열 범위 (기본: 0 25)")
    args = parser.parse_args()

    if args.zoom:
        build_zoom_comparison(
            scenario  = args.zoom_scenario,
            seed      = args.seed,
            step      = args.zoom_step,
            zoom_rows = tuple(args.zoom_rows),
            zoom_cols = tuple(args.zoom_cols),
            dpi       = args.dpi,
            out_path  = args.out,
            ppo_path  = args.ppo_path,
        )
    else:
        build_poster_grid(
            scenarios=args.scenarios,
            seed=args.seed,
            capture_frac=args.capture_frac,
            ppo_path=args.ppo_path,
            dpi=args.dpi,
            out_path=args.out,
            show_routes=args.show_routes,
            routes_only=args.routes_only,
        )