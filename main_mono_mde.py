import time

import cv2
import numpy as np

from datasets.kitti_loader import load_kitti_pair
from main_stereo import load_gt_disparity_png16, valid_mask_from_disparity, depth_from_disp, safe_detect, \
    adaptive_min_valid, roi_median_disparity, roi_median_and_coverage, draw_box_with_metrics, list_kitti_indices, \
    write_csv
from models.depth_estimator_mono import MonoDepthEstimator
from models.detector import ObjectDetector
from utils.metrics import compute_depth_metrics

KITTI_ROOT = "datasets/kitti2015/training"
SCORE_THR = 0.5
GT_COVERAGE_THR = 0.20
MIN_MEDIAN_DISP = 2.0
CLICK_HALF = 30
CSV_PATH_ALL = "mono_eval_all.csv"


def process_index(detector, estimator, idx, verbose=True):
    L, R, fx, B, gt_path = load_kitti_pair(KITTI_ROOT, idx)
    if verbose:
        print(f"\n=== IDX {idx} | fx={fx:.1f} px, B={B:.3f} m, gt={'yes' if gt_path else 'no'} ===")

    t0 = time.time()
    disp_pred = estimator.estimate(L)
    t_mono = time.time() - t0

    depth_gt = None
    gt_mask = None
    if gt_path:
        d_gt = load_gt_disparity_png16(gt_path)
        if d_gt is not None:
            depth_gt = depth_from_disp(d_gt, fx, B)
            gt_mask = valid_mask_from_disparity(d_gt)

    if depth_gt is not None and gt_mask is not None:
        valid_pred_disp = disp_pred[gt_mask]
        valid_gt_disp = (fx * B) / (depth_gt[gt_mask] + 1e-6)

        scale = np.median(valid_gt_disp) / (np.median(valid_pred_disp) + 1e-6)
        disp_aligned = disp_pred * scale
        depth_pred = (fx * B) / (disp_aligned + 1e-6)
    else:

        depth_pred = 100.0 / (disp_pred + 1e-6)
        disp_aligned = disp_pred

    t0 = time.time()
    dets = safe_detect(detector, L, score_thr=SCORE_THR)
    t_det = time.time() - t0
    if verbose:
        print(f"[time] mono_depth={t_mono * 1000:.1f}ms, detector={t_det * 1000:.1f}ms")

    disp_vis = cv2.normalize(disp_pred, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_TURBO)

    out = L.copy()
    H, W = L.shape[:2]

    frame_metrics = {}
    if depth_gt is not None and gt_mask is not None:
        frame_metrics = compute_depth_metrics(depth_gt, depth_pred, mask=gt_mask)
        m = frame_metrics
        if verbose:
            print(f"[metrics] AbsRel={m['abs_rel']:.3f} | RMSE={m['rmse']:.3f} | d1={m['delta1']:.3f}")

    results_rows = []
    valid_absrel = []

    for i, det in enumerate(dets):
        ymin, xmin, ymax, xmax = det["box"]
        x1, y1 = int(xmin * W), int(ymin * H)
        x2, y2 = int(xmax * W), int(ymax * H)
        mvalid = adaptive_min_valid(x1, y1, x2, y2)

        d_med = roi_median_disparity(disp_aligned if 'disp_aligned' in locals() else disp_pred, x1, y1, x2, y2,
                                     min_valid=mvalid)
        z_pred, _ = roi_median_and_coverage(depth_pred, None, x1, y1, x2, y2, min_valid=mvalid)
        z_gt, cov_gt = roi_median_and_coverage(depth_gt, gt_mask, x1, y1, x2, y2,
                                               min_valid=mvalid) if depth_gt is not None else (float('nan'), 0.0)

        absrel = float('nan')
        if np.isfinite(z_pred) and np.isfinite(z_gt) and cov_gt >= GT_COVERAGE_THR:
            absrel = abs(z_pred - z_gt) / (z_gt + 1e-6)
            valid_absrel.append(absrel)

        if verbose:
            print(f"[eval] box#{i} id={det.get('class_id', '?')} score={det.get('score', 0):.2f} "
                  f"-> pred={z_pred:.2f} m | gt={z_gt:.2f} m | covGT={cov_gt * 100:.1f}% "
                  f"| d_med={d_med:.2f} px | AbsRel={absrel:.3f}")

        draw_box_with_metrics(out, x1, y1, x2, y2, z_pred, z_gt, absrel)

        row = {
            "idx": idx, "box_id": i, "class_id": det.get("class_id", "?"),
            "score": f"{det.get('score', 0):.3f}",
            "z_pred": f"{z_pred:.3f}", "z_gt": f"{z_gt:.3f}", "absrel": f"{absrel:.3f}",
            "covGT": f"{cov_gt:.3f}", "d_med": f"{d_med:.3f}"
        }
        if frame_metrics:
            for k, v in frame_metrics.items(): row[f"frame_{k}"] = f"{v:.4f}"
        results_rows.append(row)

    return L, R, disp_color, out, results_rows, depth_pred, depth_gt, (disp_pred,), frame_metrics, (t_mono, t_det)


def main():
    indices = list_kitti_indices(KITTI_ROOT)
    detector = ObjectDetector("models/ssd_mobilenet_v2")
    estimator = MonoDepthEstimator(model_type="MiDaS_small")

    pos = 0
    all_rows = []
    session_metrics = []
    session_times = []

    print("[controls] SPACE/→: next | ←: prev | W: save and show summary | q: quit")

    last_idx = None

    while True:
        idx = indices[pos]
        verbose = (idx != last_idx)
        L, R, disp_color, out, rows, depth_pred, depth_gt, extras, f_metrics, f_times = process_index(detector,
                                                                                                      estimator, idx,
                                                                                                      verbose=verbose)

        last_idx = idx

        cv2.imshow("Mono Depth Eval (aligned scale)", out)
        cv2.imshow("Raw Disparity", disp_color)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), ord('0')):
            break
        elif key == 32:
            all_rows.extend(rows)
            if f_metrics: session_metrics.append(f_metrics)
            session_times.append(f_times)
            pos = (pos + 1) % len(indices)
        elif key == ord('p'):
            all_rows.extend(rows)
            if f_metrics: session_metrics.append(f_metrics)
            session_times.append(f_times)
            pos = (pos - 1) % len(indices)
        elif key == ord('W'):
            write_csv(all_rows, CSV_PATH_ALL)
            if session_metrics:
                print("\n=== SESSION SUMMARY ===")
                for k in session_metrics[0].keys():
                    vals = [m[k] for m in session_metrics if np.isfinite(m[k])]
                    if vals: print(f"{k:10s}: mean={np.mean(vals):.4f}")
            if session_times:
                tm = [t[0] for t in session_times];
                td = [t[1] for t in session_times]
                print(f"Time Mono: {np.mean(tm) * 1000:.1f}ms, Det: {np.mean(td) * 1000:.1f}ms")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
