"""
test_junctions.py — 맵의 분기점을 자동 검출하고 시각화
==================================================================
make_comparison_gifs.py와 같은 폴더(stage2/)에 두고 실행:
    python test_junctions.py

저장: result/visualize/junction_preview.png
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# make_comparison_gifs.py와 동일한 색상 팔레트
COLORS = {
    "hall":    "#EFEFEF",
    "wall":    "#1A1A1A",
    "exit":    "#00A550",
    "room":    "#E0D8C4",
}

# 시각화 전용 벽 셀 (make_comparison_gifs.py에서 그대로 가져옴)
_VISUAL_WALL_CELLS = frozenset(
    (r, c) for r in range(10, 28) for c in range(8, 13)
    if BASE_GRID[r, c] == ROOM
) | frozenset(
    (r, c) for r in range(0, 10) for c in range(18, 25)
    if BASE_GRID[r, c] == HALL
)

EXIT_A_SET = set(EXIT_A_POS)
EXIT_B_SET = set(EXIT_B_POS)


# ── 핵심: 분기점 자동 검출 ────────────────────────────────────────────────
def is_walkable(r: int, c: int, grid: np.ndarray) -> bool:
    """대피자가 지나갈 수 있는 셀인지 (HALL 또는 EXIT, 시각적 벽 제외)."""
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        return False
    if (r, c) in _VISUAL_WALL_CELLS:
        return False
    cell = grid[r, c]
    return cell == HALL or cell == EXIT


def find_junctions(grid: np.ndarray) -> list[tuple[int, int]]:
    """
    분기점 검출: HALL 셀 중 인접 walkable 이웃이 3개 이상인 곳.
    (T자, 십자 교차로)
    """
    junctions = []
    ROWS, COLS = grid.shape
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] != HALL:
                continue
            if (r, c) in _VISUAL_WALL_CELLS:
                continue
            # 4방향 이웃 중 walkable 개수
            n_walk = sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if is_walkable(r + dr, c + dc, grid)
            )
            if n_walk >= 3:
                junctions.append((r, c))
    return junctions


def cluster_junctions(junctions: list[tuple[int, int]],
                       radius: int = 2) -> list[tuple[int, int]]:
    """
    인접한 분기점들을 클러스터링해서 대표점 하나만 남김.
    (T자 교차로가 2~3칸 연속해서 검출되는 걸 방지)
    """
    if not junctions:
        return []

    # union-find 간단 버전
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
        # 클러스터 중심 (반올림)
        cr = round(sum(p[0] for p in cluster) / len(cluster))
        cc = round(sum(p[1] for p in cluster) / len(cluster))
        clusters.append((cr, cc))

    return clusters


# ── 시각화 ────────────────────────────────────────────────────────────────
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

    # 1) 클러스터링 전 모든 분기점 — 작은 회색 점
    for r, c in raw_junctions:
        ax.plot(c, r, "o", color="#999999", markersize=4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # 2) 클러스터링 후 대표 분기점 — 큰 빨간 사각형 + 번호
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
    print(f"저장: {out_path}")


# ── 진입점 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  분기점 자동 검출 테스트")
    print("=" * 60)
    print(f"  Grid shape: {BASE_GRID.shape}")

    raw = find_junctions(BASE_GRID)
    print(f"  raw 분기점 개수: {len(raw)}")
    for r, c in raw:
        print(f"    ({r}, {c})")

    clustered = cluster_junctions(raw, radius=2)
    print(f"\n  클러스터링 후: {len(clustered)}개")
    for i, (r, c) in enumerate(clustered, start=1):
        print(f"    #{i}: ({r}, {c})")

    out_dir = os.path.join(_ROOT, "result", "visualize")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "junction_preview.png")
    draw_map_with_junctions(BASE_GRID, raw, clustered, out_path)

    print("=" * 60)