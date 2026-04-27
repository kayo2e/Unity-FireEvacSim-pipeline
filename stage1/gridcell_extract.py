"""
Stage 1 Pipeline: 피난안내도 → Grid Map
========================================
논문 「객체 탐지와 OpenCV를 활용한 피난안내도의 시각장애인을 위한
촉지도 제작 자동화 기술 개발」(손명훈·박재국, 2025) 의 전처리 파이프라인을
Stage 1 (피난안내도 → numpy Grid Map) 으로 재구현.

파이프라인 순서:
  Step 0. 입력 이미지 로드
  Step 1. 피난평면도 자동추출 (OpenCV Contours, 논문 §III-2)
  Step 2. 불필요 요소 제거  (HSV 마스킹, 논문 §III-4-1)
  Step 3. 텍스트 제거       (이진화 + 연결요소 분석, 논문 §III-4-2)
  Step 4. 벽체 세선화/일반화 (Zhang-Suen + HoughLinesP, 논문 §III-4-3)
  Step 5. Grid Map 변환      (30×20 셀, 벽선 기준 0/1 인코딩)

Grid 셀 값 (최소화):
  0 = 이동 가능  (복도, 방 내부, 화장실 등)
  1 = 이동 불가  (벽, 건물 외부, 엘리베이터)
  2 = 비상구     (목표 지점)
  3 = 계단       (이동 가능하지만 비용 높음 → 보상 패널티 설계용)
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
import json
import os
from typing import Optional

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
GRID_ROWS = 30
GRID_COLS = 20

# 셀 값 (4종으로 최소화)
CELL_PASSABLE  = 0   # 이동 가능 (복도, 방, 화장실 통합)
CELL_WALL      = 1   # 이동 불가 (벽, 건물 외부, 엘리베이터 통합)
CELL_EXIT      = 2   # 비상구 (목표)
CELL_STAIR     = 3   # 계단 (이동 가능, 비용 높음)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(SCRIPT_DIR, "debug_steps")
os.makedirs(DEBUG_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def save_debug(name: str, img: np.ndarray):
    """각 스텝의 중간 결과를 저장 (디버그용)."""
    path = os.path.join(DEBUG_DIR, name)
    if img.dtype == bool:
        img = (img * 255).astype(np.uint8)
    cv2.imwrite(path, img)
    print(f"  [debug] saved → {path}")


# ──────────────────────────────────────────────
# Step 1: 피난평면도 자동추출
# ──────────────────────────────────────────────
def extract_floor_plan(img_bgr: np.ndarray) -> np.ndarray:
    """
    논문 §III-2: OpenCV Contours 기법으로 피난안내도에서
    피난평면도 영역만 크롭해 반환.
    """
    print("[Step 1] 피난평면도 자동추출")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 이진화 (Otsu)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 모폴로지 클로징 - 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    save_debug("step1_binary.png", closed)

    # Contours 탐색
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 전체 이미지의 10% 이상 → 도면 영역
        if area > img_area * 0.10 and area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        print("  [!] Contours 자동추출 실패 → 전체 이미지 사용")
        return img_bgr.copy()

    x, y, w, h = cv2.boundingRect(best_cnt)
    cropped = img_bgr[y:y+h, x:x+w]
    save_debug("step1_floor_plan_crop.png", cropped)
    print(f"  → 추출 영역: x={x}, y={y}, w={w}, h={h}, area={best_area:.0f}")
    return cropped


# ──────────────────────────────────────────────
# Step 2: 불필요 요소 제거 (HSV 기반)
# ──────────────────────────────────────────────
def remove_unnecessary_elements(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    논문 §III-4-1: HSV 색 공간으로 변환 후
    벽체(어두운/회색 범위)를 보호 마스크로 설정하고
    비벽체 색상 픽셀을 반복 제거.

    Returns:
        cleaned  : 불필요 요소가 제거된 BGR 이미지
        wall_mask: 벽 보호 마스크 (uint8, 255=벽)
    """
    print("[Step 2] 불필요 요소 제거 (HSV)")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 벽체 보호 마스크: 어두운 범위(V<80) + 회색계열(S<40)
    dark_mask  = cv2.inRange(hsv, (0, 0, 0),   (180, 255, 80))
    gray_mask  = cv2.inRange(hsv, (0, 0, 80),  (180, 40,  220))
    wall_mask  = cv2.bitwise_or(dark_mask, gray_mask)

    # 벽체 연결 보강 (단절 방지)
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.dilate(wall_mask, kernel_d, iterations=1)

    save_debug("step2_wall_mask.png", wall_mask)

    result = img_bgr.copy()
    hsv_work = hsv.copy()

    # 4회 반복: 채도 임계 S를 40→28→22→16 으로 줄여가며 비벽체 컬러 픽셀 제거
    s_thresh = 40
    for i in range(4):
        color_mask = cv2.inRange(hsv_work, (0, s_thresh, 50), (180, 255, 255))
        # 벽 보호 마스크와 겹치지 않는 영역만 제거
        remove_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(wall_mask))
        # 팽창으로 주변 노이즈까지 제거
        remove_mask = cv2.dilate(remove_mask, kernel_d, iterations=1)
        result[remove_mask > 0] = [255, 255, 255]
        hsv_work = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        s_thresh = max(6, s_thresh - 6)

    save_debug("step2_color_removed.png", result)
    print("  → 비벽체 색상 픽셀 4회 반복 제거 완료")
    return result, wall_mask


# ──────────────────────────────────────────────
# Step 3: 텍스트 제거
# ──────────────────────────────────────────────
def remove_text(img_bgr: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """
    논문 §III-4-2: 이진화 기반 텍스트 검출 및 제거.
    연결요소(area<1000, width<100, height<50)를 텍스트로 판단.
    벽체와 겹치는 텍스트는 보존.
    """
    print("[Step 3] 텍스트 제거")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 전역 이진화
    _, binary_global = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # 적응형 이진화
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    # 결합
    text_mask = cv2.bitwise_and(binary_global, binary_adaptive)

    # 연결요소 분석으로 텍스트 후보 선별
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, connectivity=8)

    remove_img = np.zeros_like(text_mask)
    for lbl in range(1, num_labels):
        area   = stats[lbl, cv2.CC_STAT_AREA]
        width  = stats[lbl, cv2.CC_STAT_WIDTH]
        height = stats[lbl, cv2.CC_STAT_HEIGHT]
        # 논문 Eq.(2): area<1000, width<100, height<50
        if area < 1000 and width < 100 and height < 50:
            remove_img[labels == lbl] = 255

    # 팽창으로 인접 텍스트 픽셀 포함
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    remove_img = cv2.dilate(remove_img, kernel_5, iterations=2)

    # 벽체 영역은 보존 (비트 반전 교차)
    wall_inv = cv2.bitwise_not(wall_mask)
    remove_final = cv2.bitwise_and(remove_img, wall_inv)

    result = img_bgr.copy()
    result[remove_final > 0] = [255, 255, 255]

    save_debug("step3_text_removed.png", result)
    print(f"  → {num_labels-1}개 연결요소 중 텍스트 제거 완료")
    return result


def skeletonize_walls(img_bgr: np.ndarray, wall_prior: Optional[np.ndarray] = None) -> np.ndarray:
    """
    논문 §III-4-3: Zhang-Suen + Lee + HoughLinesP 세선화.
    Returns: 세선화된 1채널 이진 이미지 (255=벽선)
    """
    print("[Step 4] 세선화 및 일반화")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 어두운 선/테두리 강조
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 4
    )
    save_debug("step4_binary_raw.png", binary)

    # 수평/수직 구조만 우선 추출해 잡선 억제
    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 35), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    hv_mask = cv2.bitwise_or(horizontal, vertical)

    # 끊긴 벽선 연결
    hv_mask = cv2.morphologyEx(
        hv_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    # Step2의 벽 후보 마스크를 prior로 사용해 비벽 영역 잡선 제거
    if wall_prior is not None:
        prior = cv2.dilate(
            wall_prior,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        hv_mask = cv2.bitwise_and(hv_mask, prior)

    save_debug("step4_hv_mask.png", hv_mask)

    # Zhang-Suen 세선화로 선폭 축소
    skel_uint8 = (skeletonize(hv_mask.astype(bool)) * 255).astype(np.uint8)

    # 긴 축정렬 직선만 남겨 구조선 재구성
    line_canvas = np.zeros_like(skel_uint8)
    lines = cv2.HoughLinesP(
        skel_uint8, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=35, maxLineGap=8
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 3 or dy > dx * 3:
                cv2.line(line_canvas, (x1, y1), (x2, y2), 255, 1)

    # Hough 결과가 너무 적으면 원본 세선화 사용
    if np.sum(line_canvas > 0) > 500:
        skel_uint8 = line_canvas

    # 짧은 노이즈 연결요소 제거
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_uint8)
    final_skel = np.zeros_like(skel_uint8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 25:
            final_skel[labels == lbl] = 255

    save_debug("step4_skeleton.png", final_skel)
    print(f"  → 세선화 완료. 비zero 픽셀: {np.sum(final_skel>0)}")
    return final_skel


# ──────────────────────────────────────────────
# Step 5: Grid Map 변환
# ──────────────────────────────────────────────
def build_grid_map(
    wall_mask: np.ndarray,
    img_bgr:  np.ndarray,
    wall_prior: Optional[np.ndarray] = None,  # noqa: ARG001 (reserved for future use)
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> np.ndarray:
    """
    벽체 마스크 → rows×cols Grid Map (0/1).

    알고리즘 (벡터 기반):
    1. 건물 외곽 컨투어 채우기 → 내부/외부 마스크.
    2. HoughLinesP 로 구조 벽선만 추출 (노이즈·짧은선 제외).
    3. 구조 벽선을 두께=2 로 재래스터화 → vector_wall_mask.
    4. 셀별 pixel ratio 판정:
       - inside_ratio ≤ inside_thresh → 외부 → WALL
       - vec_ratio  >  wall_thresh    → 벽선 통과 → WALL
       - 나머지                        → PASSABLE
    5. wall_prior(Step2 HSV 마스크) 보조: 두꺼운 벽 픽셀 40%+ 이면 WALL 추가.
    """
    print("[Step 5] Grid Map 변환 (벡터 기반 · 픽셀비율)")
    h, w = wall_mask.shape[:2]
    cell_h = h / rows
    cell_w = w / cols

    grid = np.full((rows, cols), CELL_WALL, dtype=np.int32)

    # ── 1. 건물 내부 마스크 ──
    # bin_orig: 어두운 벽=255, 흰 복도/방/배경=0
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_orig = cv2.threshold(gray_orig, 240, 255, cv2.THRESH_BINARY_INV)
    # 큰 커널 MORPH_CLOSE: 흰 복도·방 구멍을 채워 건물 발자국을 하나로 합침
    fill_k = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    bin_filled = cv2.morphologyEx(bin_orig, cv2.MORPH_CLOSE, fill_k)
    contours, _ = cv2.findContours(bin_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    building_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(
            building_mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED
        )

    # ── 2. 벡터 벽선 시각화용 vwm 생성 (thickness=1, 판정에는 미사용) ──
    # wall_mask는 Step4에서 이미 HoughLinesP로 벡터화·필터된 세선화 마스크.
    # 여기서는 debug 이미지용으로만 HoughLinesP를 한 번 더 돌려 선분 형태로 저장.
    min_line_len = int(max(max(cell_h, cell_w), 30))
    lines = cv2.HoughLinesP(
        wall_mask, rho=1, theta=np.pi / 180,
        threshold=20, minLineLength=min_line_len, maxLineGap=8,
    )
    vwm = np.zeros((h, w), dtype=np.uint8)  # 시각화 전용
    n_segs = 0
    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv2.line(vwm, (x1, y1), (x2, y2), 255, 1)
            n_segs += 1
    save_debug("step5_vector_wall_mask.png", vwm)
    print(f"  → 구조 벽 선분 {n_segs}개 추출")

    # navigable 마스크 (시각화 전용)
    nav_mask = cv2.bitwise_and(
        building_mask,
        cv2.bitwise_not(cv2.dilate(vwm, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))),
    )
    save_debug("step5_navigable_mask.png", nav_mask)

    # ── 3. 셀별 픽셀 비율 판정 ──
    # 벽 판정은 wall_mask(Step4 Hough-필터 세선화, 1px thick)를 직접 사용.
    # 1px 선분이 셀 전폭을 가로지를 때 비율: 수평≈2.5%, 수직≈3.5%
    # wall_thresh=0.010 → full-crossing은 잡히고, 경계만 스치는 세그먼트는 통과
    inside_thresh = 0.15   # 건물 내부 판정 (building_mask 기준)
    wall_thresh   = 0.010  # Step4 벡터 세선화 기준 (1px full-crossing ≈2.5-3.5%)

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * cell_h);       y2 = int((r + 1) * cell_h)
            x1 = int(c * cell_w);       x2 = int((c + 1) * cell_w)
            if x2 <= x1 or y2 <= y1:
                continue

            n_px = (y2 - y1) * (x2 - x1)

            inside_ratio = float(np.sum(building_mask[y1:y2, x1:x2] > 0)) / n_px
            # ── 핵심: Step4 Hough-벡터화 마스크 직접 사용 (vwm 아님) ──
            wall_ratio   = float(np.sum(wall_mask[y1:y2, x1:x2]    > 0)) / n_px

            if inside_ratio > inside_thresh:
                grid[r, c] = CELL_PASSABLE  # 건물 내부 기본값
            if wall_ratio > wall_thresh:
                grid[r, c] = CELL_WALL      # 벽선 통과 → 벽 우선

            # (wall_prior는 이미지의 80%+ 커버로 too broad → secondary check 미사용)

    print(f"  → Grid {rows}×{cols} 생성 완료")
    unique, counts = np.unique(grid, return_counts=True)
    label_map = {0: "이동가능", 1: "이동불가"}
    for u, cnt in zip(unique, counts):
        print(f"     셀값 {u}({label_map.get(u, '?')}): {cnt}개")

    return grid


# ──────────────────────────────────────────────
# Grid 시각화
# ──────────────────────────────────────────────
CELL_COLORS = {
    CELL_WALL:     (40,  40,  40),   # 검정 (벽·외부·엘리베이터)
    CELL_PASSABLE: (255, 255, 255),  # 흰색 (이동 가능)
    CELL_EXIT:     (0,   200, 80),   # 초록 (비상구)
    CELL_STAIR:    (180, 120, 60),   # 갈색 (계단)
}

def visualize_grid(grid: np.ndarray, cell_size: int = 30) -> np.ndarray:
    rows, cols = grid.shape
    canvas = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    labels = {
        CELL_WALL:     "W",
        CELL_PASSABLE: "",
        CELL_EXIT:     "E",
        CELL_STAIR:    "ST",
    }

    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            color_rgb = CELL_COLORS.get(val, (128, 128, 128))
            color_bgr = color_rgb[::-1]
            y1, x1 = r * cell_size, c * cell_size
            y2, x2 = y1 + cell_size, x1 + cell_size
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), color_bgr, -1)
            cv2.rectangle(canvas, (x1, y1), (x2-1, y2-1), (160, 160, 160), 1)

            lbl = labels.get(val, str(val))
            if lbl and lbl != " ":
                font_scale = 0.28
                thickness  = 1
                text_color = (20, 20, 20) if val != CELL_WALL else (230, 230, 230)
                tw, th_val = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                tx = x1 + (cell_size - tw) // 2
                ty = y1 + (cell_size + th_val) // 2
                cv2.putText(canvas, lbl, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

    return canvas


# ──────────────────────────────────────────────
# 전체 파이프라인
# ──────────────────────────────────────────────
def run_pipeline(input_path: str, output_dir: str = SCRIPT_DIR) -> np.ndarray:
    print("=" * 60)
    print("Stage 1 Pipeline: 피난안내도 → Grid Map")
    print("=" * 60)

    # Step 0: 이미지 로드
    print(f"[Step 0] 이미지 로드: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {input_path}")
    print(f"  → 원본 shape: {img.shape}")

    # 처리 속도를 위해 최대 1200px로 리사이즈
    MAX_DIM = 1200
    h0, w0 = img.shape[:2]
    if max(h0, w0) > MAX_DIM:
        scale = MAX_DIM / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        print(f"  → 리사이즈 후 shape: {img.shape}")

    save_debug("step0_input.png", img)

    # Step 1: 피난평면도 추출
    floor_plan = extract_floor_plan(img)
    save_debug("step0_cropped.png", floor_plan)

    # Step 2: 불필요 요소 제거
    cleaned, wall_mask_hsv = remove_unnecessary_elements(floor_plan)

    # Step 3: 텍스트 제거
    text_removed = remove_text(cleaned, wall_mask_hsv)

    # Step 4: 벽체 세선화 마스크
    wall_mask = skeletonize_walls(floor_plan, wall_mask_hsv)
    save_debug("step4_selected_wall_mask.png", wall_mask)

    # Step 5: Grid Map (선 추출 기반 단순 변환)
    grid = build_grid_map(wall_mask, floor_plan, wall_mask_hsv)

    # 결과 저장
    grid_path = os.path.join(output_dir, "grid_map.npy")
    np.save(grid_path, grid)
    print(f"\n[결과] Grid Map 저장: {grid_path}")

    grid_json_path = os.path.join(output_dir, "grid_map.json")
    with open(grid_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "grid": grid.tolist(),
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "cell_legend": {
                "0": "passable",
                "1": "wall"
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"[결과] Grid JSON 저장: {grid_json_path}")

    # 시각화
    vis = visualize_grid(grid, cell_size=30)
    vis_path = os.path.join(output_dir, "grid_visualization.png")
    cv2.imwrite(vis_path, vis)
    print(f"[결과] 시각화 저장: {vis_path}")

    # 파이프라인 비교 이미지 (Step0 → Step4 → Grid 나란히)
    step0 = cv2.imread(os.path.join(DEBUG_DIR, "step0_input.png"))
    step1 = cv2.imread(os.path.join(DEBUG_DIR, "step1_floor_plan_crop.png"))
    step4 = cv2.imread(os.path.join(DEBUG_DIR, "step4_selected_wall_mask.png"))
    if step4 is None:
        step4 = cv2.imread(os.path.join(DEBUG_DIR, "step4_skeleton.png"))
    if step4 is not None and step4.ndim == 2:
        step4 = cv2.cvtColor(step4, cv2.COLOR_GRAY2BGR)
    elif step4 is None:
        step4 = np.zeros_like(step0)

    target_h = 400
    def resize_h(im, h):
        ratio = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1]*ratio), h))

    panels = [resize_h(step1, target_h),
              resize_h(step4, target_h),
              resize_h(vis, target_h)]
    combined = np.hstack(panels)
    combined_path = os.path.join(output_dir, "pipeline_overview.png")
    cv2.imwrite(combined_path, combined)
    print(f"[결과] 파이프라인 개요 저장: {combined_path}")

    print("\n" + "=" * 60)
    print("Pipeline 완료!")
    print("=" * 60)
    return grid


if __name__ == "__main__":
    input_image = os.path.join(SCRIPT_DIR, "fire_evacuation_diagram.jpg")
    grid = run_pipeline(input_image)
    print("\nGrid Map (30×20):")
    print(grid)