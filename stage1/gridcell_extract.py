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
  Step 4. 벽체 세선화/일반화 (Zhang-Suen + Lee + HoughLinesP, 논문 §III-4-3)
  Step 5. 피난시설 탐지      (YOLOv10 대신 컬러/형태 기반 규칙 탐지 - 모델 없이)
  Step 6. Grid Map 변환      (30×20 셀, 값 정의에 따라 인코딩)

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


# ──────────────────────────────────────────────
# Step 4: 세선화 및 선 굵기 일반화
# ──────────────────────────────────────────────
def skeletonize_walls(img_bgr: np.ndarray) -> np.ndarray:
    """
    논문 §III-4-3: Zhang-Suen + Lee + HoughLinesP 세선화.
    Returns: 세선화된 1채널 이진 이미지 (255=벽선)
    """
    print("[Step 4] 세선화 및 일반화")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 이진화
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 모폴로지 클로징 - 수평/수직 방향 단락 보완
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    closed_h = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kh)
    closed_v = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kv)
    combined = cv2.bitwise_or(closed_h, closed_v)

    # 형태학적 열림(Opening) - 작은 노이즈 사전 제거
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3)

    # Zhang-Suen 세선화 (skimage)
    bool_img = opened.astype(bool)
    skel_combined = skeletonize(bool_img)

    # 연결요소 필터링 (3픽셀 이상)
    skel_uint8 = (skel_combined * 255).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel_uint8)
    filtered = np.zeros_like(skel_uint8)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= 3:
            filtered[labels == lbl] = 255

    # HoughLinesP - 단절 직선 구간 보완 (논문 표 5 파라미터)
    lines = cv2.HoughLinesP(
        filtered, rho=1, theta=np.pi/180,
        threshold=10, minLineLength=5, maxLineGap=3
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(filtered, (x1, y1), (x2, y2), 255, 1)

    # 최종 노이즈 제거 (2픽셀 미만 고립점)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(filtered)
    final_skel = np.zeros_like(filtered)
    for lbl in range(1, num_labels2):
        if stats2[lbl, cv2.CC_STAT_AREA] >= 2:
            final_skel[labels2 == lbl] = 255

    save_debug("step4_skeleton.png", final_skel)
    print(f"  → 세선화 완료. 비zero 픽셀: {np.sum(final_skel>0)}")
    return final_skel


# ──────────────────────────────────────────────
# Step 5: 피난시설 탐지 (규칙 기반, YOLOv10 대체)
# ──────────────────────────────────────────────
def detect_facilities_rule_based(img_bgr: np.ndarray) -> dict:
    """
    YOLOv10 모델 없이 색상/형태 규칙으로 피난시설 탐지.
    - 비상구(EXIT): 초록색 계열 픽셀 클러스터
    - 소화기: 빨간색 계열 픽셀 클러스터 (특정 크기)
    - 엘리베이터: 파란색 계열 픽셀 클러스터
    - 화장실: 파란색 아이콘 (남녀 픽토그램)

    Returns:
        dict with keys: 'exits', 'extinguishers', 'elevators', 'toilets'
        각각 [(cx, cy, w, h), ...] 형태의 bbox 리스트
    """
    print("[Step 5] 피난시설 탐지 (규칙 기반)")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = img_bgr.shape[:2]
    results = {"exits": [], "elevators": [], "toilets": [], "stairs": []}

    def find_clusters(mask, min_area=30, max_area=8000):
        """연결요소로 마스크 클러스터 추출."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed)
        boxes = []
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                x = stats[lbl, cv2.CC_STAT_LEFT]
                y = stats[lbl, cv2.CC_STAT_TOP]
                bw = stats[lbl, cv2.CC_STAT_WIDTH]
                bh = stats[lbl, cv2.CC_STAT_HEIGHT]
                cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
                boxes.append((cx, cy, bw, bh))
        return boxes

    # 비상구: 초록색 (H: 40~90, S>80, V>80)
    exit_mask = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))
    results["exits"] = find_clusters(exit_mask, min_area=50)
    print(f"  → 비상구 {len(results['exits'])}개")

    # 엘리베이터: 파란색 중 작은 클러스터 (H: 90~130)
    blue_mask = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
    blue_boxes = find_clusters(blue_mask, min_area=40, max_area=6000)
    for box in blue_boxes:
        cx, cy, bw, bh = box
        if bw * bh > 1000:
            results["toilets"].append(box)   # 화장실: 파란색 대형 클러스터
        else:
            results["elevators"].append(box) # 엘리베이터: 파란색 소형 클러스터
    print(f"  → 엘리베이터 {len(results['elevators'])}개, 화장실 {len(results['toilets'])}개")

    # 계단: 흑백 줄무늬 패턴 (그레이 영역 내 수평선 반복 구조)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, stair_bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    stair_lines = cv2.morphologyEx(stair_bin, cv2.MORPH_OPEN, kh)
    stair_mask = cv2.dilate(stair_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10)))
    results["stairs"] = find_clusters(stair_mask, min_area=200, max_area=8000)
    print(f"  → 계단 {len(results['stairs'])}개")

    return results


# ──────────────────────────────────────────────
# Step 6: Grid Map 변환
# ──────────────────────────────────────────────
def build_grid_map(
    skeleton: np.ndarray,
    img_bgr:  np.ndarray,
    facilities: dict,
    rows: int = GRID_ROWS,
    cols: int = GRID_COLS,
) -> np.ndarray:
    """
    세선화된 벽체 이미지 + 시설 위치 → rows×cols Grid Map.

    알고리즘:
    1. 모든 셀을 1(이동 불가)로 초기화.
    2. 건물 외곽 내부 셀은 0(이동 가능)으로.
    3. 벽체 픽셀이 셀 면적의 thresh 이상이면 1(이동 불가) 유지.
    4. 시설 bbox 중심이 속한 셀에 해당 값 부여.
    """
    print("[Step 6] Grid Map 변환")
    h, w = skeleton.shape[:2]
    cell_h = h / rows
    cell_w = w / cols

    # 초기값 1(이동 불가) - 건물 외부도 동일하게 처리
    grid = np.full((rows, cols), CELL_WALL, dtype=np.int32)

    # ── 건물 내부 판정: 원본에서 밝은 흰색 배경이 아닌 영역 ──
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 건물 윤곽선으로 내부/외부 구분
    _, bin_orig = cv2.threshold(gray_orig, 240, 255, cv2.THRESH_BINARY_INV)
    # 가장 큰 건물 외곽 채우기
    contours, _ = cv2.findContours(bin_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    building_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(building_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # ── 셀별 분류 ──
    wall_thresh = 0.03   # 셀 픽셀의 3% 이상이 벽선이면 벽
    inside_thresh = 0.15 # 셀 픽셀의 15% 이상이 건물 내부면 내부

    for r in range(rows):
        for c in range(cols):
            y1 = int(r * cell_h);  y2 = int((r+1) * cell_h)
            x1 = int(c * cell_w);  x2 = int((c+1) * cell_w)

            cell_skel    = skeleton[y1:y2, x1:x2]
            cell_building= building_mask[y1:y2, x1:x2]
            cell_pixels  = cell_skel.size

            inside_ratio = np.sum(cell_building > 0) / (cell_pixels + 1e-6)
            wall_ratio   = np.sum(cell_skel > 0) / (cell_pixels + 1e-6)

            if inside_ratio > inside_thresh:
                grid[r, c] = CELL_PASSABLE   # 건물 내부 = 이동 가능
            if wall_ratio > wall_thresh:
                grid[r, c] = CELL_WALL        # 벽 우선

    # ── 시설 오버레이 ──
    def place_facility(boxes, cell_value):
        for cx, cy, bw, bh in boxes:
            c_idx = int(cx / cell_w)
            r_idx = int(cy / cell_h)
            c_idx = np.clip(c_idx, 0, cols-1)
            r_idx = np.clip(r_idx, 0, rows-1)
            grid[r_idx, c_idx] = cell_value

    place_facility(facilities["exits"],         CELL_EXIT)
    place_facility(facilities["elevators"],     CELL_WALL)      # 엘리베이터 → 이동 불가
    place_facility(facilities["toilets"],       CELL_PASSABLE)  # 화장실 → 이동 가능
    place_facility(facilities["stairs"],        CELL_STAIR)     # 계단 → 비용 높은 이동 가능

    print(f"  → Grid {rows}×{cols} 생성 완료")
    unique, counts = np.unique(grid, return_counts=True)
    label_map = {0: "이동가능", 1: "이동불가", 2: "비상구", 3: "계단"}
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

    # Step 2: 불필요 요소 제거
    cleaned, wall_mask = remove_unnecessary_elements(floor_plan)

    # Step 3: 텍스트 제거
    text_removed = remove_text(cleaned, wall_mask)

    # Step 4: 세선화
    skeleton = skeletonize_walls(text_removed)

    # Step 5: 시설 탐지 (floor_plan 원본 컬러 사용)
    facilities = detect_facilities_rule_based(floor_plan)

    # Step 6: Grid Map
    grid = build_grid_map(skeleton, floor_plan, facilities)

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
                "1": "wall",
                "2": "exit",
                "3": "stair"
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
    input_image = os.path.join(SCRIPT_DIR, "fire_evacuation_diagram.jpeg")
    grid = run_pipeline(input_image)
    print("\nGrid Map (30×20):")
    print(grid)