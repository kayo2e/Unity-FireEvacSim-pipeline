"""
build_base_grid.py — Stage 1 원본 이미지 → Stage 2 BASE_GRID(40x25) 변환기
============================================================================
gridcell_extract.py가 만드는 grid_map.npy(209x109 이진 벽 지도)에는 EXIT 위치가
없다 — 기존 색상 마스크가 빨강/파랑을 전부 "노이즈"로 버려서다. 이 스크립트는
초록색 "비상구" 픽토그램 아이콘을 별도로 검출해 EXIT 위치까지 자동으로 채워,
env_core.py의 BASE_GRID(40x25, HALL/WALL/EXIT/ROOM)와 같은 형식으로 만든다.

ROOM/HALL 구분은 만들지 않는다 — env_core.py에서 WALKABLE = {HALL, EXIT, ROOM}로
셋을 동일하게 취급해 시뮬레이션 로직에 아무 영향이 없기 때문이다(렌더링에서 문자만
다름). 벽이 아니고 출구도 아닌 모든 칸은 HALL로 채운다.

실행:
    cd stage1
    python build_base_grid.py
    # 출력: base_grid.npy (40x25), base_grid_vis.jpg
"""
import cv2
import numpy as np
from collections import deque

HALL, WALL, EXIT, ROOM = 0, 1, 2, 3  # env_core.py와 동일한 인코딩
TARGET_ROWS, TARGET_COLS = 40, 25


def load_wall_mask(image_path="image.jpg"):
    # gridcell_extract.py와 동일한 벽 검출 로직 재사용(색 노이즈 제거 → 어두운 픽셀 → 벽)
    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ranges = [
        ((0, 45, 60), (10, 255, 255)),
        ((165, 45, 60), (179, 255, 255)),
        ((90, 35, 50), (135, 255, 255)),
    ]
    noise_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        noise_mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    noise_mask = cv2.morphologyEx(noise_mask, cv2.MORPH_CLOSE, kernel)
    noise_mask = cv2.dilate(noise_mask, kernel)

    rb_removed = img.copy()
    rb_removed[noise_mask != 0] = 255
    gray_temp = cv2.cvtColor(rb_removed, cv2.COLOR_BGR2GRAY)
    rb_removed[gray_temp < 150] = 0
    gray = cv2.cvtColor(rb_removed, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray, 0, 130)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    clean = np.zeros_like(black_mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 80:
            continue
        if w < 15 and h < 15:
            continue
        clean[labels == i] = 255
    return clean


def find_exit_icons(image_path="image.jpg", min_area=500, top_k=4):
    # "비상구" 픽토그램은 표준 소방 표지 색(초록 바탕)이라 별도 HSV 범위로 검출 가능.
    # 원본 해상도 그대로 검출(리사이즈된 벽 마스크와는 별도 좌표계 — 비율로 매핑함).
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 60, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    icons = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        icons.append((area, cx, cy))
    icons.sort(reverse=True)
    return icons[:top_k], img.shape[:2]  # (h, w)


def pool_to_grid(binary_mask, target_rows, target_cols, threshold=0.05):
    # gridcell_extract.py와 같은 기준(흰색 비율 > 5% → 벽). 209x109 중간 단계 없이
    # 리사이즈된 이진 마스크에서 40x25로 직접 풀링해 반올림 오차를 한 번만 태운다.
    h, w = binary_mask.shape
    row_edges = np.linspace(0, h, target_rows + 1).astype(int)
    col_edges = np.linspace(0, w, target_cols + 1).astype(int)
    grid = np.zeros((target_rows, target_cols), dtype=np.int32)
    for r in range(target_rows):
        for c in range(target_cols):
            cell = binary_mask[row_edges[r]:row_edges[r + 1], col_edges[c]:col_edges[c + 1]]
            if cell.size == 0:
                continue
            white_ratio = np.count_nonzero(cell == 255) / cell.size
            grid[r, c] = WALL if white_ratio > threshold else HALL
    return grid


def map_icon_to_grid(cx, cy, full_hw, target_rows, target_cols):
    h, w = full_hw
    row = int(cy / h * target_rows)
    col = int(cx / w * target_cols)
    return max(0, min(target_rows - 1, row)), max(0, min(target_cols - 1, col))


def nearest_walkable(grid, r, c, max_radius=6):
    # 아이콘 좌표는 표지판이 붙은 벽 위치일 수 있어 실제 통행 가능한 셀이 아닐 수 있다.
    # BFS로 가장 가까운 HALL 셀을 찾아 그 자리를 EXIT로 표시한다.
    rows, cols = grid.shape
    if grid[r, c] == HALL:
        return r, c
    seen = {(r, c)}
    q = deque([(r, c, 0)])
    while q:
        cr, cc, d = q.popleft()
        if d > max_radius:
            break
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                seen.add((nr, nc))
                if grid[nr, nc] == HALL:
                    return nr, nc
                q.append((nr, nc, d + 1))
    return r, c


def build_base_grid(image_path="image.jpg"):
    wall_mask = load_wall_mask(image_path)
    grid = pool_to_grid(wall_mask, TARGET_ROWS, TARGET_COLS)

    icons, full_hw = find_exit_icons(image_path)
    exit_cells = []
    for area, cx, cy in icons:
        r, c = map_icon_to_grid(cx, cy, full_hw, TARGET_ROWS, TARGET_COLS)
        r2, c2 = nearest_walkable(grid, r, c)
        exit_cells.append((r2, c2))
        grid[r2, c2] = EXIT

    return grid, exit_cells, icons


def save_visualization(grid, path="base_grid_vis.jpg", cell_px=30):
    rows, cols = grid.shape
    colors = {
        WALL: (30, 30, 30),
        EXIT: (60, 180, 60),
        HALL: (200, 185, 155),
        ROOM: (200, 185, 155),
    }
    vis = np.full((rows * cell_px, cols * cell_px, 3), 240, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_px, (r + 1) * cell_px
            x1, x2 = c * cell_px, (c + 1) * cell_px
            vis[y1:y2, x1:x2] = colors[grid[r, c]]
    for r in range(rows + 1):
        cv2.line(vis, (0, r * cell_px), (cols * cell_px, r * cell_px), (120, 120, 120), 1)
    for c in range(cols + 1):
        cv2.line(vis, (c * cell_px, 0), (c * cell_px, rows * cell_px), (120, 120, 120), 1)
    cv2.imwrite(path, vis)


if __name__ == "__main__":
    grid, exits, icons = build_base_grid()
    np.save("base_grid.npy", grid)
    save_visualization(grid)

    vals, counts = np.unique(grid, return_counts=True)
    names = {HALL: "HALL", WALL: "WALL", EXIT: "EXIT", ROOM: "ROOM"}
    print(f"grid shape: {grid.shape}")
    print("cell counts:", {names[v]: c for v, c in zip(vals.tolist(), counts.tolist())})
    print(f"detected {len(icons)} green exit icons (area, cx, cy):", icons)
    print("mapped to grid cells (EXIT):", exits)
    print("출력: base_grid.npy, base_grid_vis.jpg")
