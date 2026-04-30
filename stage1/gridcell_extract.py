import cv2
import numpy as np
import math

img = cv2.imread("image.jpg")

img = cv2.resize(img, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ranges = [
    ((0, 45, 60),   (10, 255, 255)),    # red low
    ((165, 45, 60), (179, 255, 255)),   # red high
    ((90, 35, 50),  (135, 255, 255)),   # blue
]

mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

for lower, upper in ranges:
    mask |= cv2.inRange(
        hsv,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8)
    )

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.dilate(mask, kernel)

rb_removed = img.copy()
rb_removed[mask != 0] = 255
enhanced = rb_removed.copy()

# 어두운 영역 마스크
gray_temp = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
dark_mask = gray_temp < 150

# 어두운 픽셀 검은색으로
enhanced[dark_mask] = 0

# 이후 enhanced를 gray로 변환
gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
black_mask = cv2.inRange(gray, 0, 130)
cv2.imwrite("result.jpg", black_mask)
cv2.imwrite("processed_image.jpg", black_mask)


num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    black_mask,
    connectivity=8
)

clean = np.zeros_like(black_mask)

min_area = 80      # 너무 작은 흰 객체 제거
min_width = 15     # 숫자처럼 작은 폭 제거
min_height = 15

for i in range(1, num_labels):  # 0은 배경
    x, y, w, h, area = stats[i]

    # 숫자/작은 노이즈 제거
    if area < min_area:
        continue

    # 너무 작은 객체 제거
    if w < min_width and h < min_height:
        continue

    clean[labels == i] = 255

cv2.imwrite("before.jpg", black_mask)
cv2.imwrite("clean.jpg", clean)


# 그리드맵 만들기
binary_img = clean.copy()

grid_size = 5  # 5x5픽셀 = 1셀
h, w = binary_img.shape

grid_h = int(np.ceil(h / grid_size))
grid_w = int(np.ceil(w / grid_size))

grid_map = np.zeros((grid_h, grid_w), dtype=np.uint8)

# 셀 안에서 흰색 비율이 이 값 이상이면 벽
white_threshold = 0.05   # 5%

for gy in range(grid_h):
    for gx in range(grid_w):
        y1 = gy * grid_size
        y2 = min((gy + 1) * grid_size, h)
        x1 = gx * grid_size
        x2 = min((gx + 1) * grid_size, w)

        cell = binary_img[y1:y2, x1:x2]

        white_ratio = np.count_nonzero(cell == 255) / cell.size

        if white_ratio > white_threshold:
            grid_map[gy, gx] = 1   # 벽
        else:
            grid_map[gy, gx] = 0   # 빈 공간

# grid_map을 이미지로 시각화
np.save("grid_map.npy", grid_map)

# ── 셀 단위 시각화 ──────────────────────────────────────────
CELL_PX   = 30                          # 출력 이미지에서 1셀 = 30px
WALL_CLR  = (30,  30,  30)              # 벽:  거의 검정
PASS_CLR  = (200, 185, 155)             # 통로: 베이지
LINE_CLR  = (120, 120, 120)             # 격자선: 회색

vis_h = grid_h * CELL_PX
vis_w = grid_w * CELL_PX
cell_vis = np.full((vis_h, vis_w, 3), 240, dtype=np.uint8)

for gy in range(grid_h):
    for gx in range(grid_w):
        y1c = gy * CELL_PX;  y2c = y1c + CELL_PX
        x1c = gx * CELL_PX;  x2c = x1c + CELL_PX
        cell_vis[y1c:y2c, x1c:x2c] = WALL_CLR if grid_map[gy, gx] == 1 else PASS_CLR

# 격자선
for gy in range(grid_h + 1):
    cv2.line(cell_vis, (0, gy * CELL_PX), (vis_w, gy * CELL_PX), LINE_CLR, 1)
for gx in range(grid_w + 1):
    cv2.line(cell_vis, (gx * CELL_PX, 0), (gx * CELL_PX, vis_h), LINE_CLR, 1)

# 셀 좌표 텍스트 (5셀마다)
for gy in range(0, grid_h, 5):
    for gx in range(0, grid_w, 5):
        cv2.putText(cell_vis, f"{gx},{gy}",
                    (gx * CELL_PX + 2, gy * CELL_PX + CELL_PX - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.6, (80, 80, 80), 1)

cv2.imwrite("binary_img.jpg",   binary_img)
cv2.imwrite("grid_cell_vis.jpg", cell_vis)
print(f"그리드 크기: {grid_w}cols × {grid_h}rows  |  "
      f"벽={int(grid_map.sum())}  통로={(grid_map==0).sum()}")