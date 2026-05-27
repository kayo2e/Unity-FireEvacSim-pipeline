"""
make_comparison_gifs.py — 시나리오별 A* vs PPO 나란히 비교 GIF 생성
====================================================================
유도등:
  - 분기점 자동 검출 (시각화 전용)
  - 매 프레임 화재 반영 BFS로 화살표 방향 동적 재계산
  - A* 패널: 화재만 회피
  - PPO 패널: 화재 + 짙은 연기까지 회피 (지능형 차별화)

실행:
    cd stage2
    python make_comparison_gifs.py
    python make_comparison_gifs.py --scenarios 6 --dpi 80
"""

import io, os, sys, argparse
from collections import deque
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

SC_DISPLAY_NUM  = {6: 5}
SC_NAME_DISPLAY = {
    1: "기본 탈출",
    2: "EXIT A 위협",
    3: "진입로 차단",
    4: "양방향 위협",
    6: "중앙 차단",
}

SC_SEED = {1: 42, 2: 5, 3: 1, 4: 4, 6: 20}

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

_VISUAL_WALL_CELLS = frozenset(
    (r, c) for r in range(10, 28) for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c) for r in range(0, 10) for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

_DANGER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "danger", [(1.0, 0.12, 0.12), (1.0, 0.70, 0.0), (0.05, 0.78, 0.25)], N=64
)

_ICON_CACHE: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# 분기점 유도등 시스템 (시각화 전용)
# ═══════════════════════════════════════════════════════════════════════════

def _is_walkable(r: int, c: int, grid: np.ndarray) -> bool:
    """벽이 아닌 모든 셀을 walkable로 인정 (HALL/ROOM/EXIT 모두)."""
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if (r, c) in _VISUAL_WALL_CELLS:
        return False
    return grid[r, c] != WALL


def _find_junctions(grid: np.ndarray) -> list:
    junctions = []
    ROWS, COLS = grid.shape
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] != HALL or (r, c) in _VISUAL_WALL_CELLS:
                continue
            n_walk = sum(1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                         if _is_walkable(r + dr, c + dc, grid))
            if n_walk >= 3:
                junctions.append((r, c))
    return junctions


def _cluster_junctions(junctions: list, radius: int = 2) -> list:
    if not junctions:
        return []
    pts = list(junctions)
    used = [False] * len(pts)
    clusters = []
    for i, p in enumerate(pts):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            q = pts[j]
            if abs(p[0] - q[0]) <= radius and abs(p[1] - q[1]) <= radius:
                cluster.append(q)
                used[j] = True
        cr = round(sum(p[0] for p in cluster) / len(cluster))
        cc = round(sum(p[1] for p in cluster) / len(cluster))
        clusters.append((cr, cc))
    return clusters


def _enforce_min_spacing(points: list, min_dist: int = 5) -> list:
    """
    탐욕적으로 최소 간격 유지 — 이미 선택된 점에서 chebyshev 거리가
    min_dist 이상인 점들만 남긴다.
    EXIT 근처(EXIT_A_POS, EXIT_B_POS의 평균 위치)는 우선 보존.
    """
    if not points:
        return []

    # EXIT 중심점 (필수 유지)
    exit_centers = []
    if EXIT_A_POS:
        exit_centers.append((
            round(sum(r for r, c in EXIT_A_POS) / len(EXIT_A_POS)),
            round(sum(c for r, c in EXIT_A_POS) / len(EXIT_A_POS)),
        ))
    if EXIT_B_POS:
        exit_centers.append((
            round(sum(r for r, c in EXIT_B_POS) / len(EXIT_B_POS)),
            round(sum(c for r, c in EXIT_B_POS) / len(EXIT_B_POS)),
        ))

    # EXIT 중심에 가장 가까운 분기점부터 강제 선택
    selected = []
    remaining = list(points)
    for ec in exit_centers:
        if not remaining:
            break
        nearest = min(remaining,
                      key=lambda p: abs(p[0]-ec[0]) + abs(p[1]-ec[1]))
        selected.append(nearest)
        remaining.remove(nearest)

    # 나머지는 row 순으로 정렬하면서 최소 간격 유지
    remaining.sort(key=lambda p: (p[0], p[1]))
    for p in remaining:
        ok = all(max(abs(p[0]-s[0]), abs(p[1]-s[1])) >= min_dist
                 for s in selected)
        if ok:
            selected.append(p)
    return selected


_GUIDANCE_POSITIONS = _enforce_min_spacing(
    _cluster_junctions(_find_junctions(BASE_GRID), radius=2),
    min_dist=5,
)
print(f"[init] 유도등 위치: {len(_GUIDANCE_POSITIONS)}개")


def _bfs_dist_from_exits(grid: np.ndarray, exit_set: set,
                          blocked: set,
                          fire_cells: set = None,
                          fire_penalty: float = 50.0,
                          near_fire_penalty: float = 5.0) -> np.ndarray:
    """
    Dijkstra 기반 거리맵 — 화재 근처는 비용은 높지만 통과 가능.
    이렇게 하면 BFS가 '도달 불가' 판정을 거의 안 내고, 화살표가 항상 살아있음.

    blocked         : 절대 통과 불가 (보통 비워둠 — fire는 penalty로 처리)
    fire_cells      : 화재 셀 (cost +fire_penalty)
    near_fire_penalty: 화재 인접 셀 cost 가산
    """
    import heapq
    if fire_cells is None:
        fire_cells = set()

    ROWS, COLS = grid.shape
    dist = np.full((ROWS, COLS), np.inf, dtype=np.float32)
    pq: list = []

    for (r, c) in exit_set:
        if (r, c) in blocked:
            continue
        dist[r, c] = 0.0
        heapq.heappush(pq, (0.0, r, c))

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if grid[nr, nc] == WALL or (nr, nc) in _VISUAL_WALL_CELLS:
                continue
            if (nr, nc) in blocked:
                continue
            # 이동 비용 결정
            step_cost = 1.0
            if (nr, nc) in fire_cells:
                step_cost = fire_penalty   # 화재 통과는 매우 비싸지만 가능
            else:
                # 인접 4셀 중 하나라도 화재면 페널티
                for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if (nr + ddr, nc + ddc) in fire_cells:
                        step_cost += near_fire_penalty
                        break
            nd = d + step_cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                heapq.heappush(pq, (nd, nr, nc))
    return dist


def _compute_arrow(pos: tuple, dist_map: np.ndarray,
                    grid: np.ndarray):
    r, c = pos
    if not np.isfinite(dist_map[r, c]):
        return None
    best = None
    best_d = dist_map[r, c]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            continue
        if grid[nr, nc] == WALL or (nr, nc) in _VISUAL_WALL_CELLS:
            continue
        d = dist_map[nr, nc]
        if d < best_d:
            best_d = d
            best = (dc, dr)  # (x방향=col, y방향=row)
    return best


# ═══════════════════════════════════════════════════════════════════════════
# 화재 글로우 / 비상구 아이콘
# ═══════════════════════════════════════════════════════════════════════════

def _get_exit_icon(size: int = 52) -> np.ndarray:
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]
    path = os.path.join(_ROOT, "evac_sign.png")
    img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
    _ICON_CACHE[size] = np.array(img)
    return _ICON_CACHE[size]


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


# ═══════════════════════════════════════════════════════════════════════════
# 유도등 렌더링 — 화살표 노랑 + 사각형 밖으로 뻗음
# ═══════════════════════════════════════════════════════════════════════════

# 이전 프레임 화살표 방향 캐시 (label별로 따로 저장)
_PREV_ARROWS: dict = {}


def _draw_guidance_lights(ax: plt.Axes, env: FireEvacEnv,
                            label: str, zorder: int = 4):
    """
    분기점 유도등을 그리고 가장 가까운 EXIT 방향으로 화살표 표시.

    A*  : 화재 페널티 약함 (그래서 화재 위치를 잘 못 피함)
    PPO : 화재 페널티 강함 (멀찍이서부터 우회 유도)

    이전 프레임과 화살표 방향이 바뀐 분기점은 노란 펄스 + 흰 테두리로 강조.
    """
    fire_cells = set()
    fire_ys, fire_xs = np.where(env.fire_map > 0)
    for r, c in zip(fire_ys, fire_xs):
        fire_cells.add((int(r), int(c)))

    blocked = set(env.blocked_exits)   # blocked는 막힌 EXIT만

    # 정책별 화재 페널티 차별화 (PPO가 더 멀리서부터 우회)
    if label == "PPO":
        fire_pen = 80.0
        near_pen = 10.0
        # PPO는 짙은 연기도 비싸게 (연기 1셀당 +3)
        smoke_ys, smoke_xs = np.where(env.smoke_map > 0.6)
        for r, c in zip(smoke_ys, smoke_xs):
            fire_cells.add((int(r), int(c)))  # smoke를 fire_cells에 합쳐서 near_penalty 적용
    else:  # A*
        fire_pen = 30.0
        near_pen = 2.0

    dist_a = _bfs_dist_from_exits(env.grid, EXIT_A_SET, blocked,
                                   fire_cells, fire_pen, near_pen)
    dist_b = _bfs_dist_from_exits(env.grid, EXIT_B_SET, blocked,
                                   fire_cells, fire_pen, near_pen)

    # 이번 프레임 방향 캐시
    prev_dirs = _PREV_ARROWS.get(label, {})
    curr_dirs: dict = {}

    for (r, c) in _GUIDANCE_POSITIONS:
        da = dist_a[r, c] if np.isfinite(dist_a[r, c]) else np.inf
        db = dist_b[r, c] if np.isfinite(dist_b[r, c]) else np.inf

        if da == np.inf and db == np.inf:
            # 거의 발생 안 함 (soft cost라 도달 불가는 EXIT가 완전 막혔을 때만)
            ax.plot(c, r, "x", color="#D32F2F", markersize=7,
                    markeredgewidth=1.8, zorder=zorder + 1)
            continue

        dm = dist_a if da <= db else dist_b
        direction = _compute_arrow((r, c), dm, env.grid)
        if direction is None:
            continue
        dx, dy = direction
        curr_dirs[(r, c)] = (dx, dy)

        # 이전과 방향 바뀌었는지 체크
        changed = (r, c) in prev_dirs and prev_dirs[(r, c)] != (dx, dy)

        # 방향 변경 시 노란 펄스 (글로우 효과)
        if changed:
            for s, a in [(0.95, 0.18), (0.70, 0.30), (0.50, 0.45)]:
                ax.add_patch(plt.Rectangle(
                    (c - s/2, r - s/2), s, s,
                    facecolor="#FFEB3B", alpha=a,
                    edgecolor="none",
                    zorder=zorder - 0.5,
                ))

        # 유도등 본체 — 작은 녹색 사각형
        body_color = "#FFC107" if changed else "#1B5E20"   # 변경 시 황색 본체
        edge_color = "#FFEB3B" if changed else "white"
        edge_w     = 1.4 if changed else 0.6
        ax.add_patch(plt.Rectangle(
            (c - 0.22, r - 0.22), 0.44, 0.44,
            facecolor=body_color, alpha=0.92,
            edgecolor=edge_color, linewidth=edge_w,
            zorder=zorder,
        ))
        # 화살표 — 파란색
        ax.annotate(
            "",
            xy=(c + dx * 0.55, r + dy * 0.55),
            xytext=(c - dx * 0.02, r - dy * 0.02),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.5,head_width=0.40",
                color="#2196F3",
                lw=2.0,
                mutation_scale=12,
            ),
            zorder=zorder + 1,
        )

    # 이번 프레임 방향 저장 (다음 호출에서 비교용)
    _PREV_ARROWS[label] = curr_dirs


# ═══════════════════════════════════════════════════════════════════════════
# PPO 로드
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# 단일 환경 패널 렌더링
# ═══════════════════════════════════════════════════════════════════════════

def _draw_env(env: FireEvacEnv, step: int, ax: plt.Axes,
              label: str, label_color: str,
              done: bool = False):
    ROWS, COLS = env.ROWS, env.COLS

    def hex2rgba(h, a=1.0):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16)/255 for i in (0, 2, 4)] + [a]

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

    for r, c in zip(*np.where(env.smoke_map > 0)):
        if env.grid[r, c] != WALL and (r, c) not in _VISUAL_WALL_CELLS:
            img[r, c] = hex2rgba(COLORS["smoke"], 0.40)

    for r, c in zip(*np.where(env.fire_map > 0)):
        if env.grid[r, c] != WALL and (r, c) not in _VISUAL_WALL_CELLS:
            img[r, c] = hex2rgba(COLORS["fire"], 0.93)

    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    _draw_fire_glow(ax, env, zorder=3)
    _draw_guidance_lights(ax, env, label, zorder=4)

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
            t = min(min_d / 6.0, 1.0)
            key = (pr, pc)
            if key not in cell_rep or t < cell_rep[key][0]:
                cell_rep[key] = (t, pr, pc)
        for t, pr, pc in sorted(cell_rep.values(), key=lambda x: -x[0]):
            ax.plot(pc, pr, "o",
                    color=_DANGER_CMAP(t),
                    markersize=11.0,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    zorder=6, alpha=0.93)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if label == "PPO":
        for sp in ax.spines.values():
            sp.set_edgecolor("#1565C0")
            sp.set_linewidth(2.2)

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

    if done:
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


def _render_single_panel(env: FireEvacEnv, step: int, label: str,
                          label_color: str, annotation: str,
                          done: bool, dpi: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.73, bottom=0.01)
    _draw_env(env, step, ax, label, label_color, done=done)

    fig.text(0.5, 0.970, label,
             ha="center", va="top",
             fontsize=17, fontweight="bold", color=label_color,
             transform=fig.transFigure)

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
                 dpi: int):
    env = FireEvacEnv(scenario=scenario, n_agents=n)
    env.reset(seed=seed)

    # 이전 화살표 캐시 초기화 (이 에피소드의 첫 프레임은 모두 '변화 없음')
    _PREV_ARROWS[label] = {}

    panels = []
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
    last_no_banner = _render_single_panel(env, finish, label, label_color,
                                          annotation, False, dpi)
    env.close()
    return panels, finish, last_no_banner


def _stitch_frame(left_img: Image.Image, right_img: Image.Image,
                  sc_num: int, sc_name: str, dpi: int) -> Image.Image:
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
        mpatches.Patch(facecolor=COLORS["fire"],                                label="화재"),
        mpatches.Patch(facecolor=COLORS["smoke"],   edgecolor="#aaa",  lw=0.6,  label="연기"),
        mpatches.Patch(facecolor="#1B5E20",         edgecolor="white", lw=0.6,  label="유도등 (→ 방향)"),
        mlines.Line2D([], [], markerfacecolor="#FF1F1F", markersize=10, label="대피자 (위험)", **_mk),
        mlines.Line2D([], [], markerfacecolor="#FFB200", markersize=10, label="대피자 (주의)", **_mk),
        mlines.Line2D([], [], markerfacecolor="#0DC940", markersize=10, label="대피자 (안전)", **_mk),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=6,
               fontsize=9.0, framealpha=0.90, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.01))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def make_gif(scenario: int, seed: int = 42, dpi: int = 80,
             duration: int = 150, every: int = 1):
    from baselines.astar_real import astar_action

    cfg       = SCENARIO_CONFIGS[scenario]
    n         = cfg["n_agents"]
    max_steps = cfg["max_steps"]
    sc_num    = SC_DISPLAY_NUM.get(scenario, scenario)
    sc_name   = SC_NAME_DISPLAY.get(scenario, cfg["name"])
    notes     = SCENARIO_NOTES.get(scenario, {})
    seed      = SC_SEED.get(scenario, seed)

    print(f"\n  [S{scenario}→S{sc_num} {sc_name}]  agents={n}  seed={seed}")

    ppo_fn = _load_ppo(scenario, n)
    if ppo_fn is None:
        print(f"    [경고] PPO 모델 없음 — 건너뜀")
        return None

    print(f"    A* 실행 중...")
    a_panels, a_done, a_last_no_banner = _run_episode(
        scenario, n, seed, astar_action, max_steps, every,
        "A*", "#333333", notes.get("astar", ""), dpi)

    print(f"    PPO 실행 중...")
    p_panels, p_done, p_last_no_banner = _run_episode(
        scenario, n, seed, ppo_fn, max_steps, every,
        "PPO", "#1565C0", notes.get("ppo", ""), dpi)

    print(f"    A* 완료: t={a_done}  |  PPO 완료: t={p_done}")

    a_map = dict(a_panels)
    p_map = dict(p_panels)
    a_last_banner = a_panels[-1][1]
    p_last_banner = p_panels[-1][1]

    all_steps = sorted(set(a_map) | set(p_map))
    frames = []
    durations = []

    for step in all_steps:
        a_fin = step >= a_done
        p_fin = step >= p_done

        if a_fin and p_fin:
            l_img = a_last_no_banner
            r_img = p_last_no_banner
        else:
            l_img = a_map.get(step, a_last_banner)
            r_img = p_map.get(step, p_last_banner)

        frames.append(_stitch_frame(l_img, r_img, sc_num, sc_name, dpi))
        durations.append(duration)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="시나리오별 A* vs PPO 나란히 비교 GIF 생성")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--dpi",      type=int, default=80)
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--every",    type=int, default=1)
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