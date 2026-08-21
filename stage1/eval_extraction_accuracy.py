"""
eval_extraction_accuracy.py — Stage 1 추출 품질 평가 (Phase B/C/D)
========================================================================
docs/progress.md 2026-08-21 계획 참고. ground_truth_grid.npy(Phase A) 대비
base_grid.npy(Phase 0 개선 후 추출 결과)를 세 갈래로 평가한다.

실행:
    cd stage1
    python eval_extraction_accuracy.py
"""
import numpy as np
from collections import deque

HALL, WALL, EXIT = 0, 1, 2
WALKABLE = {HALL, EXIT}


def load():
    gt = np.load("ground_truth_grid.npy")
    pred = np.load("base_grid.npy")
    return gt, pred


def phase_b_standard_metrics(gt, pred):
    # Zeng et al. 2019 / CubiCasa5K 프로토콜: pixel accuracy, 클래스별 IoU, precision/recall
    gt_wall = gt == WALL
    pred_wall = pred == WALL
    tp = np.sum(gt_wall & pred_wall)
    fp = np.sum(~gt_wall & pred_wall)
    fn = np.sum(gt_wall & ~pred_wall)
    tn = np.sum(~gt_wall & ~pred_wall)

    pixel_acc = (tp + tn) / gt.size
    wall_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    wall_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    wall_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    hall_gt = gt != WALL
    hall_pred = pred != WALL
    inter = np.sum(hall_gt & hall_pred)
    union = np.sum(hall_gt | hall_pred)
    hall_iou = inter / union if union > 0 else float("nan")

    return {
        "pixel_accuracy": pixel_acc,
        "wall_iou": wall_iou,
        "wall_precision": wall_precision,
        "wall_recall": wall_recall,
        "hall_iou": hall_iou,
    }


def phase_c_lenient_match(gt, pred, radius=1):
    # de las Heras et al. 2014 프로토콜의 부분 매치 개념을 단순화 적용.
    # 정답 WALL 셀 기준 반경 내에 예측도 WALL이 있으면 매치로 인정.
    rows, cols = gt.shape
    gt_wall_cells = np.argwhere(gt == WALL)
    matched = 0
    for r, c in gt_wall_cells:
        r0, r1 = max(0, r - radius), min(rows, r + radius + 1)
        c0, c1 = max(0, c - radius), min(cols, c + radius + 1)
        if np.any(pred[r0:r1, c0:c1] == WALL):
            matched += 1
    lenient_recall = matched / len(gt_wall_cells) if len(gt_wall_cells) else float("nan")

    pred_wall_cells = np.argwhere(pred == WALL)
    matched_p = 0
    for r, c in pred_wall_cells:
        r0, r1 = max(0, r - radius), min(rows, r + radius + 1)
        c0, c1 = max(0, c - radius), min(cols, c + radius + 1)
        if np.any(gt[r0:r1, c0:c1] == WALL):
            matched_p += 1
    lenient_precision = matched_p / len(pred_wall_cells) if len(pred_wall_cells) else float("nan")

    return {"lenient_recall_r1": lenient_recall, "lenient_precision_r1": lenient_precision}


def _bfs_reachable(grid, starts):
    rows, cols = grid.shape
    dist = np.full((rows, cols), 9999)
    q = deque()
    for r, c in starts:
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and dist[nr, nc] == 9999 and grid[nr, nc] in WALKABLE:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def phase_d_downstream_metrics(gt, pred):
    # 연결성 보존율: 정답 기준 출구 도달 가능한 HALL 셀 중, 예측 그리드에서도
    # (같은 좌표가) 도달 가능한 비율. RL downstream 관점에서 픽셀 정확도보다
    # 직접적으로 의미 있는 지표(우리 고유 기여, 문헌에 없음).
    gt_exits = [tuple(rc) for rc in np.argwhere(gt == EXIT)]
    pred_exits = [tuple(rc) for rc in np.argwhere(pred == EXIT)]

    gt_dist = _bfs_reachable(gt, gt_exits)
    pred_dist = _bfs_reachable(pred, pred_exits)

    gt_reachable = {(r, c) for r, c in np.argwhere(gt != WALL) if gt_dist[r, c] != 9999 and gt[r, c] != EXIT}
    preserved = sum(1 for (r, c) in gt_reachable if pred_dist[r, c] != 9999)
    connectivity_preservation = preserved / len(gt_reachable) if gt_reachable else float("nan")

    # EXIT 검출 precision/recall (반경 2셀 이내 매치로 인정 — 셀 하나짜리 좌표라 엄격 매치는 과함)
    def _near(a, b_list, r=2):
        return any(abs(a[0] - b[0]) <= r and abs(a[1] - b[1]) <= r for b in b_list)

    exit_recall = sum(1 for e in gt_exits if _near(e, pred_exits)) / len(gt_exits) if gt_exits else float("nan")
    exit_precision = sum(1 for e in pred_exits if _near(e, gt_exits)) / len(pred_exits) if pred_exits else float("nan")

    return {
        "connectivity_preservation": connectivity_preservation,
        "gt_reachable_cells": len(gt_reachable),
        "exit_recall": exit_recall,
        "exit_precision": exit_precision,
    }


if __name__ == "__main__":
    gt, pred = load()
    print(f"grid shape gt={gt.shape} pred={pred.shape}")

    print("\n=== Phase B: 표준 지표 (Zeng 2019 / CubiCasa5K 프로토콜) ===")
    for k, v in phase_b_standard_metrics(gt, pred).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Phase C: 관대한 매칭 (de las Heras 2014 프로토콜, radius=1) ===")
    for k, v in phase_c_lenient_match(gt, pred, radius=1).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Phase D: 다운스트림 지표 (우리 고유 기여) ===")
    for k, v in phase_d_downstream_metrics(gt, pred).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
