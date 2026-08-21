"""
build_ground_truth.py — image.jpg 전용 수작업 정답(ground truth) 그리드 제작
================================================================================
Phase A(docs/progress.md 2026-08-21 계획 참고). image_grid_overlay.jpg(40x25
그리드를 사진 위에 겹친 시각화)를 3등분해서 직접 읽고 방/복도/출구 경계를
좌표로 옮긴 것. WALL/HALL만 구분한다 — env_core.py에서 ROOM은 HALL과 기능적으로
동일해(WALKABLE 집합) 구분 실익이 없다(이미 확인됨, docs/feature-space... 아님,
hazard-aware-ablation.md 계열 확인 사항).

정확한 셀 단위 라벨링이 아니라 구조적 블록(방/복도/코어) 단위 근사다. 40x25
해상도에서는 문 폭 같은 미세 구조가 셀 하나보다 작아 사람이 봐도 셀 경계에서
판단이 갈리는 부분이 있다 — 이건 정답 자체의 한계로 문서화해둔다(Phase C의
"관대한 매칭"이 이런 경계 오차를 흡수하기 위한 것).

실행:
    cd stage1
    python build_ground_truth.py
    # 출력: ground_truth_grid.npy (40x25), ground_truth_vis.jpg
"""
import cv2
import numpy as np

HALL, WALL, EXIT = 0, 1, 2
ROWS, COLS = 40, 25

# door row positions found by build_base_grid.py's find_door_openings() —
# 재사용해 정답에도 같은 자리를 문으로 뚫는다(육안으로 대조해 확인함,
# 도면상 각 방 하나당 문 하나인 게 맞음).
DOOR_ROWS = [11, 18, 5, 16, 10, 30, 28, 24, 22, 33, 38]
DOOR_COL = 6


def build_ground_truth():
    g = np.full((ROWS, COLS), HALL, dtype=np.int32)

    # 1) 상단 해칭 기계실/옥상 구조물 (row 0-3, col 0-13)
    g[0:4, 0:14] = WALL
    # 2) 건물 바깥(오른쪽 위 요철 컷아웃) — row 0-13, col 14-24
    g[0:14, 14:25] = WALL

    # 3) 왼쪽 외벽 (col 0), 하단 외벽(row 39), 상단 일부는 이미 처리됨
    g[4:40, 0] = WALL
    g[39, :] = WALL

    # 4) 201~209호와 복도를 가르는 벽 (col 6), 문 위치만 뚫음
    g[4:39, DOOR_COL] = WALL
    for r in DOOR_ROWS:
        g[r, DOOR_COL] = HALL

    # 5) 중앙 코어(엘리베이터/계단, 해칭 박스) — row 18-24, col 7-11
    g[18:24, 7:12] = WALL
    # 6) 하단 엘리베이터 코어 — row 30-34, col 9-11
    g[30:34, 9:12] = WALL

    # 7) 오른쪽 외벽(비스듬한 벽) — 근사: row별로 벽이 시작되는 col이 점점
    #    작아짐(건물이 아래로 갈수록 넓어지는 사다리꼴 형태를 반영)
    right_wall_col_by_row = {
        14: 24, 18: 23, 22: 22, 26: 21, 30: 20, 34: 19, 38: 18,
    }
    rows_sorted = sorted(right_wall_col_by_row)
    for i in range(len(rows_sorted) - 1):
        r0, r1 = rows_sorted[i], rows_sorted[i + 1]
        c0, c1 = right_wall_col_by_row[r0], right_wall_col_by_row[r1]
        for r in range(r0, r1):
            t = (r - r0) / (r1 - r0)
            wall_col = int(round(c0 + (c1 - c0) * t))
            g[r, wall_col:25] = WALL
    g[38:40, 18:25] = WALL

    # 8) 오른쪽 방 210/211/212와 복도를 가르는 벽 — col 13 부근
    g[14:39, 13] = WALL

    # 9) 210/211/212 내부 구획선(수평 경계, 참고용 — 벽까지는 확신 어려워 생략)
    #    room 하나로 취급(WALKABLE 관점에서 중요하지 않음)

    # 10) EXIT 2곳 — build_base_grid.py 자동 검출과 동일 위치(육안 대조 완료)
    g[14, 15] = EXIT
    g[35, 8] = EXIT

    return g


def save_visualization(grid, path="ground_truth_vis.jpg", cell_px=30):
    rows, cols = grid.shape
    colors = {WALL: (30, 30, 30), EXIT: (60, 180, 60), HALL: (200, 185, 155)}
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
    g = build_ground_truth()
    np.save("ground_truth_grid.npy", g)
    save_visualization(g)
    vals, counts = np.unique(g, return_counts=True)
    names = {HALL: "HALL", WALL: "WALL", EXIT: "EXIT"}
    print("cell counts:", {names[v]: c for v, c in zip(vals.tolist(), counts.tolist())})
    print("saved ground_truth_grid.npy, ground_truth_vis.jpg")
