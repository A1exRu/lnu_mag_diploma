import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from datasets.kitti_loader import load_kitti_pair
from main_stereo import (
    list_kitti_indices,
    safe_detect,
    load_gt_disparity_png16,
    valid_mask_from_disparity,
    depth_from_disp,
    roi_median_and_coverage,
    adaptive_min_valid
)
from models.detector import ObjectDetector

KITTI_ROOT = "datasets/kitti2015/training"
SCORE_THR = 0.5
GT_COVERAGE_THR = 0.20
CLICK_HALF = 30

click_point = None
current_custom_cls = 1

CLASS_NAMES = {
    1: "Person",
    3: "Car",
    10: "Traffic Light",
    11: "Sign"
}


def on_mouse(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        click_point = None


COCO_HEIGHT_HINTS: Dict[int, float] = {
    1: 1.70,
    2: 1.40,
    3: 1.50,
    4: 1.60,
    6: 2.30,
    8: 3.20,
    10: 0.80,
    11: 0.60,
}

MIN_BBOX_H_PX = 10


def draw_box_with_metrics_custom(img, x1, y1, x2, y2, z_pred, z_gt, absrel, cls_name=""):
    """Custom draw function with class name display."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    if np.isfinite(z_pred) and np.isfinite(z_gt) and np.isfinite(absrel):
        label = f"{cls_name} | pred={z_pred:.2f}m | gt={z_gt:.2f}m | rel={absrel:.3f}"
    elif np.isfinite(z_pred):
        label = f"{cls_name} | pred={z_pred:.2f}m | gt=n/a"
    else:
        label = f"{cls_name} | n/a"
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)


def pinhole_distance_from_bbox(
        box_xyxy: Tuple[int, int, int, int],
        cls_id: int,
        fx: float,
        height_hints: Dict[int, float],
        default_h: float = 1.70,
) -> Optional[float]:
    """
    Оцінка дистанції за pinhole-моделлю: Z ≈ fx * H_real / h_img.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    h_px = max(0, y2 - y1)
    if h_px < MIN_BBOX_H_PX or fx <= 0:
        return None
    H_real = float(height_hints.get(cls_id, default_h))
    Z = (fx * H_real) / float(h_px)
    if not np.isfinite(Z) or Z <= 0:
        return None
    return float(Z)


def process_index(detector, idx, custom_click=None, custom_cls=1, verbose=True):
    L, R, fx, B, gt_path = load_kitti_pair(KITTI_ROOT, idx)
    if verbose:
        print(
            f"\n=== IDX {idx} | fx={fx:.1f} px, gt={'yes' if gt_path else 'no'} | Custom Mode: {CLASS_NAMES.get(custom_cls, 'Unknown')} ===")

    depth_gt = None
    gt_mask = None
    if gt_path:
        d_gt = load_gt_disparity_png16(gt_path)
        if d_gt is not None:
            depth_gt = depth_from_disp(d_gt, fx, B)
            gt_mask = valid_mask_from_disparity(d_gt)

    t0 = time.time()
    dets = safe_detect(detector, L, score_thr=SCORE_THR)
    t_det = time.time() - t0
    if verbose:
        print(f"[time] detector={t_det * 1000:.1f}ms")

    vis = L.copy()
    H, W = L.shape[:2]

    all_dets = []
    for d in dets:
        ymin, xmin, ymax, xmax = d["box"]
        all_dets.append({
            "box": (int(xmin * W), int(ymin * H), int(xmax * W), int(ymax * H)),
            "cls": int(d.get("class_id", -1)),
            "is_custom": False
        })

    if custom_click is not None:
        cx, cy = custom_click
        x1, y1 = max(0, cx - CLICK_HALF), max(0, cy - CLICK_HALF)
        x2, y2 = min(W - 1, cx + CLICK_HALF), min(H - 1, cy + CLICK_HALF)
        all_dets.append({
            "box": (x1, y1, x2, y2),
            "cls": custom_cls,
            "is_custom": True
        })

    for i, d_info in enumerate(all_dets):
        x1, y1, x2, y2 = d_info["box"]
        cls_id = d_info["cls"]
        is_custom = d_info["is_custom"]

        cls_name = CLASS_NAMES.get(cls_id, f"id{cls_id}")

        z_pinhole = pinhole_distance_from_bbox((x1, y1, x2, y2), cls_id, fx, COCO_HEIGHT_HINTS)

        z_gt = float('nan')
        cov_gt = 0.0
        if depth_gt is not None:
            mvalid = adaptive_min_valid(x1, y1, x2, y2)
            z_gt, cov_gt = roi_median_and_coverage(depth_gt, gt_mask, x1, y1, x2, y2, min_valid=mvalid)

        absrel = float('nan')
        if z_pinhole is not None and np.isfinite(z_gt) and cov_gt >= GT_COVERAGE_THR:
            absrel = abs(z_pinhole - z_gt) / (z_gt + 1e-6)

        color = (255, 0, 255) if is_custom else (36, 255, 12)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        draw_box_with_metrics_custom(vis, x1, y1, x2, y2, z_pinhole if z_pinhole is not None else float('nan'), z_gt,
                                     absrel, cls_name)

        if verbose:
            prefix = "[custom]" if is_custom else f"[det#{i}]"
            zp_str = f"{z_pinhole:.2f}m" if z_pinhole is not None else "n/a"
            zg_str = f"{z_gt:.2f}m" if np.isfinite(z_gt) else "n/a"
            print(f"  {prefix} {cls_name:12s} | pinhole={zp_str:8s} | gt={zg_str:8s} | rel={absrel:.3f}")

    y_off = 30
    cv2.putText(vis, f"Mode: {CLASS_NAMES.get(custom_cls, '???')}", (W - 180, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    y_off += 25
    cv2.putText(vis, "1: Person | 2: Car | 3: Light | 4: Sign", (W - 350, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)

    return vis


def main():
    global click_point
    if not os.path.exists(KITTI_ROOT):
        print(f"Error: KITTI_ROOT not found at {KITTI_ROOT}")
        return

    indices = list_kitti_indices(KITTI_ROOT)
    if not indices:
        print("Error: No KITTI indices found.")
        return

    detector = ObjectDetector("models/ssd_mobilenet_v2")

    global current_custom_cls
    pos = 0
    win_name = "Mono Pinhole KITTI"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n[Controls] SPACE/→: next | SHIFT+SPACE/←: prev | ESC/q: quit")
    print("[Custom Class] 1: Person | 2: Car | 3: Traffic Light | 4: Sign")

    last_idx = None
    last_click = None
    last_cls = None

    while True:
        idx = indices[pos]

        verbose = (idx != last_idx) or (click_point != last_click) or (current_custom_cls != last_cls)

        vis = process_index(detector, idx, custom_click=click_point, custom_cls=current_custom_cls, verbose=verbose)

        last_idx = idx
        last_click = click_point
        last_cls = current_custom_cls

        cv2.imshow(win_name, vis)

        full_key = cv2.waitKey(30)
        key = full_key & 0xFF

        if key == 27 or key == ord('q'):
            break
        elif key == 32:
            pos = (pos + 1) % len(indices)
            click_point = None
        elif key == ord('p'):
            pos = (pos - 1) % len(indices)
            click_point = None
        elif key == ord('1'):
            current_custom_cls = 1
        elif key == ord('2'):
            current_custom_cls = 3
        elif key == ord('3'):
            current_custom_cls = 10
        elif key == ord('4'):
            current_custom_cls = 11

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
