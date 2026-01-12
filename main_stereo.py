import csv
import glob
import os
import time

import cv2
import numpy as np

from datasets.kitti_loader import load_kitti_pair
from models.depth_estimator_stereo import StereoDepthEstimator, StereoCalib
from models.detector import ObjectDetector
from utils.metrics import compute_depth_metrics

KITTI_ROOT = "datasets/kitti2015/training"
SCORE_THR = 0.5
GT_COVERAGE_THR = 0.20
MIN_MEDIAN_DISP = 2.0
CLICK_HALF = 30
CSV_PATH_PER_FRAME = "kitti_eval_frame.csv"
CSV_PATH_ALL = "kitti_eval_all.csv"

DEPTH_CLIP = (0.2, 200.0)


def load_gt_disparity_png16(path_png):
    """GT disparity PNG16 зі шкалою 256 → float disparity (px), nan там, де даних немає."""
    if path_png is None or not os.path.exists(path_png):
        return None
    disp_png = cv2.imread(path_png, cv2.IMREAD_UNCHANGED)
    if disp_png is None:
        return None
    d = disp_png.astype(np.float32) / 256.0
    d[d <= 0] = np.nan
    return d


def depth_from_disp(d, fx, B, clip=DEPTH_CLIP):
    """Z (м) = fx * B / d, з обрізанням для стабільного відображення."""
    Z = (fx * B) / d
    if clip:
        Z = np.clip(Z, clip[0], clip[1])
    return Z


def valid_mask_from_disparity(d):
    """Маска валідних GT-пікселів з disparity (True там, де GT є)."""
    if d is None:
        return None
    return np.isfinite(d) & (d > 0)


def adaptive_min_valid(x1, y1, x2, y2, cap=(20, 200), frac=0.05):
    """Адаптивний поріг валідних пікселів: 5% площі ROI, в межах cap."""
    area = max(1, (x2 - x1) * (y2 - y1))
    need = int(frac * area)
    return max(cap[0], min(cap[1], need))


def roi_median_and_coverage(M, mask, x1, y1, x2, y2, min_valid=80):
    """
    Повертає (медіана по валідних пікселях у ROI, покриття_mask_в_ROI у [0..1]).
    Якщо mask=None, покриття = частка валідних пікселів самої матриці M.
    """
    if M is None:
        return float('nan'), 0.0
    H, W = M.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return float('nan'), 0.0

    roi_M = M[y1:y2, x1:x2]
    if mask is not None:
        roi_mask = mask[y1:y2, x1:x2]
        valid_vals = roi_M[np.isfinite(roi_M) & roi_mask]
        coverage = float(np.count_nonzero(roi_mask)) / float(max(1, roi_mask.size))
    else:
        valid_vals = roi_M[np.isfinite(roi_M)]
        coverage = float(valid_vals.size) / float(max(1, roi_M.size))

    if valid_vals.size >= min_valid:
        return float(np.median(valid_vals)), coverage
    return float('nan'), coverage


def roi_median_disparity(disp, x1, y1, x2, y2, min_valid=80):
    """Медіана disparity у ROI (px)."""
    H, W = disp.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    roi = disp[y1:y2, x1:x2].astype(np.float32)
    roi = roi[np.isfinite(roi) & (roi > 0)]
    if roi.size >= min_valid:
        return float(np.median(roi))
    return float('nan')


def draw_box_with_metrics(img, x1, y1, x2, y2, z_pred, z_gt, absrel):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    if np.isfinite(z_pred) and np.isfinite(z_gt) and np.isfinite(absrel):
        label = f"pred={z_pred:.2f}m | gt={z_gt:.2f}m | rel={absrel:.3f}"
    elif np.isfinite(z_pred):
        label = f"pred={z_pred:.2f}m | gt=n/a"
    else:
        label = "n/a"
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)


def safe_detect(detector, frame_bgr, score_thr=SCORE_THR):
    """Підтримка обох версій ObjectDetector.detect."""
    try:
        return detector.detect(frame_bgr, score_thr=score_thr)
    except TypeError:
        dets = detector.detect(frame_bgr)
        if dets is None:
            return []
        out = []
        for d in dets:
            s = d.get("score", 1.0)
            if float(s) >= score_thr:
                out.append(d)
        return out


def list_kitti_indices(kitti_root):
    """Збирає всі базові імена файлів з image_2/*.png."""
    paths = sorted(glob.glob(os.path.join(kitti_root, "image_2", "*.png")))

    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def process_index(detector, idx):
    """
    Готує все для одного кадру:
      - читає L,R, fx, B, GT;
      - рахує disparity/depth (pred), depth_gt, gt_mask;
      - запускає детектор;
      - повертає всі візуалізації й таблицю результатів для CSV.
    """

    L, R, fx, B, gt_path = load_kitti_pair(KITTI_ROOT, idx)
    print(f"\n=== IDX {idx} | fx={fx:.1f} px, B={B:.3f} m, gt={'yes' if gt_path else 'no'} ===")

    stereo = StereoDepthEstimator(StereoCalib(fx=fx, baseline_m=B), use_uncalibrated_rectify=False)

    t0 = time.time()
    disp_pred = stereo.disparity(L, R)
    t_stereo = time.time() - t0

    d = disp_pred.astype(np.float32)
    d[d < 1.0] = np.nan
    depth_pred = depth_from_disp(d, fx, B)

    depth_gt = None
    gt_mask = None
    if gt_path:
        d_gt = load_gt_disparity_png16(gt_path)
        if d_gt is not None:
            depth_gt = depth_from_disp(d_gt, fx, B)
            gt_mask = valid_mask_from_disparity(d_gt)
        else:
            print("[warn] GT disparity not loaded")

    t0 = time.time()
    dets = safe_detect(detector, L, score_thr=SCORE_THR)
    t_det = time.time() - t0
    print(f"[time] stereo={t_stereo * 1000:.1f}ms, detector={t_det * 1000:.1f}ms")
    print(f"[det] {len(dets)} boxes with score >= {SCORE_THR}")

    disp_vis = np.nan_to_num(d)
    disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    out = L.copy()
    H, W = L.shape[:2]

    frame_metrics = {}
    if depth_gt is not None and gt_mask is not None:
        frame_metrics = compute_depth_metrics(depth_gt, depth_pred, mask=gt_mask)
        m = frame_metrics
        print(
            f"[metrics] AbsRel={m['abs_rel']:.3f} | RMSE={m['rmse']:.3f} | d1={m['delta1']:.3f} | SILog={m['silog']:.2f}")

    results_rows = []
    valid_absrel = []
    any_printed = False

    for i, det in enumerate(dets):
        ymin, xmin, ymax, xmax = det["box"]
        x1, y1 = int(xmin * W), int(ymin * H)
        x2, y2 = int(xmax * W), int(ymax * H)
        mvalid = adaptive_min_valid(x1, y1, x2, y2)

        d_med = roi_median_disparity(d, x1, y1, x2, y2, min_valid=mvalid)
        z_pred, _ = roi_median_and_coverage(depth_pred, None, x1, y1, x2, y2, min_valid=mvalid)
        z_gt, cov_gt = roi_median_and_coverage(depth_gt, gt_mask, x1, y1, x2, y2,
                                               min_valid=mvalid) if depth_gt is not None else (float('nan'), 0.0)

        absrel = float('nan')
        if np.isfinite(z_pred) and np.isfinite(z_gt) and cov_gt >= GT_COVERAGE_THR and np.isfinite(
                d_med) and d_med >= MIN_MEDIAN_DISP:
            absrel = abs(z_pred - z_gt) / (z_gt + 1e-6)
            valid_absrel.append(absrel)

        print(f"[eval] box#{i} id={det.get('class_id','?')} score={det.get('score',0):.2f} "
              f"-> pred={z_pred:.2f} m | gt={z_gt if np.isfinite(z_gt) else float('nan'):.2f} m "
              f"| covGT={cov_gt*100:.1f}% | d_med={d_med if np.isfinite(d_med) else float('nan'):.2f} px "
              f"| AbsRel={absrel if np.isfinite(absrel) else float('nan'):.3f}")

        draw_box_with_metrics(out, x1, y1, x2, y2, z_pred, z_gt, absrel)
        any_printed = True

        row = {
            "idx": idx,
            "box_id": i,
            "class_id": det.get("class_id", "?"),
            "score": f"{det.get('score', 0):.3f}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "covGT": f"{cov_gt:.3f}",
            "d_med": f"{d_med if np.isfinite(d_med) else float('nan'):.3f}",
            "z_pred": f"{z_pred if np.isfinite(z_pred) else float('nan'):.3f}",
            "z_gt": f"{z_gt if np.isfinite(z_gt) else float('nan'):.3f}",
            "absrel": f"{absrel if np.isfinite(absrel) else float('nan'):.3f}"
        }

        if frame_metrics:
            for k, v in frame_metrics.items():
                row[f"frame_{k}"] = f"{v:.4f}"
        results_rows.append(row)

    if len(valid_absrel) > 0:
        arr = np.array(valid_absrel, dtype=np.float32)
        print(f"[summary:{idx}] used={len(arr)} | AbsRel mean={arr.mean():.3f} | median={np.median(arr):.3f}")
    else:
        print(f"[summary:{idx}] no boxes passed coverage & disparity thresholds")

    return L, R, disp_color, out, results_rows, depth_pred, depth_gt, (d,), frame_metrics, (t_stereo, t_det)


def write_csv(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] saved {len(rows)} rows -> {path}")


def main():
    indices = list_kitti_indices(KITTI_ROOT)
    if not indices:
        raise RuntimeError("Не знайдено файлів у image_2/*.png")

    detector = ObjectDetector("models/ssd_mobilenet_v2")

    pos = 0
    all_rows = []
    click_img = None

    session_metrics = []
    session_times = []

    L, R, disp_color, out, rows, depth_pred, depth_gt, extras, f_metrics, f_times = process_index(detector,
                                                                                                  indices[pos])
    all_rows.extend(rows)
    if f_metrics: session_metrics.append(f_metrics)
    session_times.append(f_times)
    click_img = out.copy()

    def on_mouse(event, x, y, flags, param):
        nonlocal click_img
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x - CLICK_HALF, y - CLICK_HALF
            x2, y2 = x + CLICK_HALF, y + CLICK_HALF

            d = extras[0]
            mvalid = adaptive_min_valid(x1, y1, x2, y2, cap=(20, 200))
            z_pred, _ = roi_median_and_coverage(depth_pred, None, x1, y1, x2, y2, min_valid=mvalid)
            z_gt, cov_gt = roi_median_and_coverage(depth_gt, valid_mask_from_disparity(
                depth_gt if depth_gt is None else depth_pred) if False else valid_mask_from_disparity(
                load_gt_disparity_png16(
                    os.path.join(KITTI_ROOT, "disp_occ_0", f"{indices[pos]}.png"))) if os.path.exists(
                os.path.join(KITTI_ROOT, "disp_occ_0", f"{indices[pos]}.png")) else None, x1, y1, x2, y2,
                                                   min_valid=mvalid) if depth_gt is not None else (float('nan'), 0.0)
            d_med = roi_median_disparity(d, x1, y1, x2, y2, min_valid=mvalid)
            absrel = (abs(z_pred - z_gt) / (z_gt + 1e-6)) if np.isfinite(z_pred) and np.isfinite(
                z_gt) and cov_gt >= GT_COVERAGE_THR and np.isfinite(d_med) and d_med >= MIN_MEDIAN_DISP else float(
                'nan')
            click_img = out.copy()
            draw_box_with_metrics(click_img, x1, y1, x2, y2, z_pred, z_gt, absrel)
            print(
                f"[click:{indices[pos]}] ({x},{y}) -> pred={z_pred:.2f} m | gt={z_gt if np.isfinite(z_gt) else float('nan'):.2f} m "
                f"| covGT={cov_gt * 100:.1f}% | d_med={d_med if np.isfinite(d_med) else float('nan'):.2f} px "
                f"| AbsRel={absrel if np.isfinite(absrel) else float('nan'):.3f}")

    cv2.namedWindow("Left (pred vs GT)")
    cv2.setMouseCallback("Left (pred vs GT)", on_mouse)

    while True:
        cv2.imshow("Left (pred vs GT)", click_img)
        cv2.imshow("Right", R)
        cv2.imshow("Disparity (pred)", disp_color)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord('q'), ord('0')):
            break

        elif key == ord('w'):
            write_csv(rows, CSV_PATH_PER_FRAME)

        elif key == ord('W'):
            write_csv(all_rows, CSV_PATH_ALL)
            if session_metrics:
                print("\n=== SESSION SUMMARY METRICS ===")
                for k in session_metrics[0].keys():
                    vals = [m[k] for m in session_metrics if np.isfinite(m[k])]
                    if vals:
                        print(f"{k:10s}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}")
            if session_times:
                t_stereo = [t[0] for t in session_times]
                t_det = [t[1] for t in session_times]
                print(f"Time Stereo:   mean={np.mean(t_stereo) * 1000:.1f}ms, FPS={1.0 / np.mean(t_stereo):.1f}")
                print(f"Time Detector: mean={np.mean(t_det) * 1000:.1f}ms, FPS={1.0 / np.mean(t_det):.1f}")

        elif key == 32:
            pos = (pos + 1) % len(indices)

            L, R, disp_color, out, rows, depth_pred, depth_gt, extras, f_metrics, f_times = process_index(detector,
                                                                                                          indices[pos])
            all_rows.extend(rows)
            if f_metrics: session_metrics.append(f_metrics)
            session_times.append(f_times)
            click_img = out.copy()

            cv2.setMouseCallback("Left (pred vs GT)", on_mouse)
        elif key == ord('p'):
            pos = (pos - 1) % len(indices)

            L, R, disp_color, out, rows, depth_pred, depth_gt, extras, f_metrics, f_times = process_index(detector,
                                                                                                          indices[pos])
            all_rows.extend(rows)
            if f_metrics: session_metrics.append(f_metrics)
            session_times.append(f_times)
            click_img = out.copy()

            cv2.setMouseCallback("Left (pred vs GT)", on_mouse)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
