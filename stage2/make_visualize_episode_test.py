"""
make_visualize_episode.py — 유도등 위치 + 방향 화살표 시각화 (단일 파일 버전)
======================================================================
make_comparison_gifs.py와 같은 폴더(stage2/)에 두고 실행:
    python make_visualize_episode.py

저장:
    result/visualize/junction_preview.png       — 분기점 검출 결과
    result/visualize/guidance_preview_nearest.png — 가장 가까운 EXIT 방향
    result/visualize/guidance_preview_exit_a.png  — 모두 EXIT A 방향
    result/visualize/guidance_preview_exit_b.png  — 모두 EXIT B 방향
"""

import os, sys
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_NANUM = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(_NANUM):
    font_manager.fontManager.addfont(_NANUM)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_NANUM).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_core import (
    HALL, WALL, EXIT, ROOM,
    EXIT_A_POS, EXIT_B_POS,
    BASE_GRID,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))

# 색상 팔레트
COLORS = {
    "hall":    "#EFEFEF",
    "wall":    "#1A1A1A",
    "exit":    "#00A550",
    "room":    "#E0D8C4",
}

# 시각화 전용 벽 셀 (make_comparison_gifs.py와 동일)
_VISUAL_WALL_CELLS = frozenset(
    (r, c) for r in range(10, 28) for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c) for r in range(0, 10) for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)


# ── 공통: 통과 가능 셀 판정 ──────────────────────────────────────────────
def is_walkable(r: int, c: int, grid: np.ndarray) -> bool:
    """대피자가 지나갈 수 있는 셀인지 (HALL 또는 EXIT, 시각적 벽 제외)."""
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if (r, c) in _VISUAL_WALL_CELLS:
        return False
    cell = grid[r, c]
    return cell == HALL or cell == EXIT


# ── 분기점 자동 검출 ─────────────────────────────────────────────────────
def find_junctions(grid: np.ndarray) -> list[tuple[int, int]]:
    """HALL 셀 중 인접 walkable 이웃이 3개 이상인 곳 (T자/십자)."""
    junctions = []
    ROWS, COLS = grid.shape
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] != HALL:
                continue
            if (r, c) in _VISUAL_WALL_CELLS:
                continue
            n_walk = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if is_walkable(r + dr, c + dc, grid)
            )
            if n_walk >= 3:
                junctions.append((r, c))
    return junctions


def cluster_junctions(junctions: list[tuple[int, int]],
                       radius: int = 2) -> list[tuple[int, int]]:
    """인접한 분기점들을 클러스터링해서 대표점 하나만 남김."""
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


# ── BFS 거리맵 계산 ───────────────────────────────────────────────────────
def bfs_distance_from_exits(grid: np.ndarray,
                              exit_cells: set,
                              blocked: set = None) -> np.ndarray:
    """
    EXIT 셀들로부터 역방향 BFS — 모든 walkable 셀의 출구까지 최단거리.
    blocked: 추가로 통과 불가 처리할 셀(예: 화재 셀)
    반환: dist_map (출구=0, 도달불가=inf)
    """
    if blocked is None:
        blocked = set()

    ROWS, COLS = grid.shape
    dist = np.full((ROWS, COLS), np.inf, dtype=np.float32)
    q = deque()

    for (r, c) in exit_cells:
        if (r, c) in blocked:
            continue
        dist[r, c] = 0.0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not is_walkable(nr, nc, grid):
                continue
            if (nr, nc) in blocked:
                continue
            if dist[nr, nc] > dist[r, c] + 1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


# ── 유도등 방향 결정 ─────────────────────────────────────────────────────
def compute_arrow(pos: tuple, dist_map: np.ndarray,
                   grid: np.ndarray) -> tuple[int, int] | None:
    """유도등 위치의 4방향 이웃 중 dist_map이 가장 작은 방향 반환."""
    r, c = pos
    best = None
    best_d = dist_map[r, c]
    if not np.isfinite(best_d):
        return None
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if not is_walkable(nr, nc, grid):
            continue
        d = dist_map[nr, nc]
        if d < best_d:
            best_d = d
            best = (dr, dc)
    return best


# ── 시각화 1: 분기점 검출 결과 ───────────────────────────────────────────
def draw_map_with_junctions(grid: np.ndarray,
                              raw_junctions: list[tuple[int, int]],
                              clustered: list[tuple[int, int]],
                              out_path: str):
    ROWS, COLS = grid.shape

    def hex2rgba(h, a=1.0):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)] + [a]

    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET or (r, c) in EXIT_B_SET:
                img[r, c] = hex2rgba(COLORS["exit"])
            elif cell == ROOM:
                img[r, c] = hex2rgba(COLORS["room"])
            else:
                img[r, c] = hex2rgba(COLORS["hall"])

    fig, ax = plt.subplots(figsize=(8, 9), dpi=110)
    fig.patch.set_facecolor("white")
    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    for r, c in raw_junctions:
        ax.plot(c, r, "o", color="#999999", markersize=4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    for i, (r, c) in enumerate(clustered, start=1):
        ax.add_patch(plt.Rectangle(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            facecolor="#E53935", alpha=0.85,
            edgecolor="white", linewidth=1.5,
            zorder=6
        ))
        ax.text(c, r, str(i),
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                zorder=7)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"분기점 자동 검출 결과\n"
        f"raw (회색 점): {len(raw_junctions)}개  →  "
        f"clustered (빨강 □): {len(clustered)}개",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ── 시각화 2: 유도등 + 화살표 ────────────────────────────────────────────
def draw_with_arrows(grid: np.ndarray,
                       junctions: list[tuple[int, int]],
                       dist_a: np.ndarray, dist_b: np.ndarray,
                       out_path: str,
                       strategy: str = "nearest"):
    ROWS, COLS = grid.shape

    def hex2rgba(h, a=1.0):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)] + [a]

    img = np.zeros((ROWS, COLS, 4), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            cell = grid[r, c]
            if cell == WALL or (r, c) in _VISUAL_WALL_CELLS:
                img[r, c] = hex2rgba(COLORS["wall"])
            elif (r, c) in EXIT_A_SET or (r, c) in EXIT_B_SET:
                img[r, c] = hex2rgba(COLORS["exit"])
            elif cell == ROOM:
                img[r, c] = hex2rgba(COLORS["room"])
            else:
                img[r, c] = hex2rgba(COLORS["hall"])

    fig, ax = plt.subplots(figsize=(8, 9), dpi=110)
    fig.patch.set_facecolor("white")
    ax.imshow(img, origin="upper", aspect="equal", interpolation="nearest")

    for (r, c) in junctions:
        if strategy == "exit_a":
            dm = dist_a
            arrow_color = "#FF7043"
        elif strategy == "exit_b":
            dm = dist_b
            arrow_color = "#42A5F5"
        else:  # nearest
            if dist_a[r, c] <= dist_b[r, c]:
                dm = dist_a
                arrow_color = "#FF7043"
            else:
                dm = dist_b
                arrow_color = "#42A5F5"

        ax.add_patch(plt.Rectangle(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            facecolor="#1B5E20", alpha=0.92,
            edgecolor="white", linewidth=1.2,
            zorder=6
        ))

        direction = compute_arrow((r, c), dm, grid)
        if direction is None:
            ax.plot(c, r, "x", color="red", markersize=10, zorder=8)
            continue
        dr, dc = direction
        ax.annotate(
            "",
            xy=(c + dc * 0.42, r + dr * 0.42),
            xytext=(c - dc * 0.18, r - dr * 0.18),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.5,head_width=0.45",
                color=arrow_color,
                lw=2.4,
                mutation_scale=12,
            ),
            zorder=8,
        )

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    title_map = {
        "nearest": "전략: 가장 가까운 EXIT  (주황=A, 파랑=B)",
        "exit_a":  "전략: 모두 EXIT A 방향",
        "exit_b":  "전략: 모두 EXIT B 방향",
    }
    ax.set_title(
        f"유도등 화살표 시각화 ({len(junctions)}개)\n{title_map[strategy]}",
        fontsize=12.5, fontweight="bold", pad=10
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


# ── 진입점 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  유도등 위치 + 방향 시각화")
    print("=" * 60)
    print(f"  Grid shape: {BASE_GRID.shape}")

    # 1) 분기점 검출
    raw = find_junctions(BASE_GRID)
    junctions = cluster_junctions(raw, radius=2)
    print(f"  raw 분기점: {len(raw)}개  →  clustered: {len(junctions)}개")

    # 2) 각 EXIT에서 BFS
    dist_a = bfs_distance_from_exits(BASE_GRID, EXIT_A_SET)
    dist_b = bfs_distance_from_exits(BASE_GRID, EXIT_B_SET)
    print(f"  EXIT A 도달가능 셀: {int(np.isfinite(dist_a).sum())}")
    print(f"  EXIT B 도달가능 셀: {int(np.isfinite(dist_b).sum())}")

    out_dir = os.path.join(_ROOT, "result", "visualize")
    os.makedirs(out_dir, exist_ok=True)

    # 3) 분기점 검출 미리보기
    draw_map_with_junctions(
        BASE_GRID, raw, junctions,
        os.path.join(out_dir, "junction_preview.png")
    )

    # 4) 3가지 전략 화살표
    for strat in ("nearest", "exit_a", "exit_b"):
        draw_with_arrows(
            BASE_GRID, junctions, dist_a, dist_b,
            os.path.join(out_dir, f"guidance_preview_{strat}.png"),
            strategy=strat,
        )

    print("=" * 60)