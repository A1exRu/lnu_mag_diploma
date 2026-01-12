import csv

import numpy as np

from datasets.kitti_loader import load_kitti_pair
from main_mono_pinhole_kitti import pinhole_distance_from_bbox, COCO_HEIGHT_HINTS
from main_stereo import (
    list_kitti_indices,
    load_gt_disparity_png16,
    depth_from_disp,
    valid_mask_from_disparity,
    safe_detect,
    roi_median_and_coverage,
    adaptive_min_valid
)
from models.depth_estimator_mono import MonoDepthEstimator
from models.depth_estimator_stereo import StereoDepthEstimator, StereoCalib
from models.detector import ObjectDetector

KITTI_ROOT = "datasets/kitti2015/training"
NUM_SCENES = 1000
GT_COVERAGE_THR = 0.20


def compare_methods():
    print("🚀 Starting comparison analysis...")

    indices = list_kitti_indices(KITTI_ROOT)[:NUM_SCENES]
    if not indices:
        print("Error: No KITTI indices found.")
        return

    detector = ObjectDetector("models/ssd_mobilenet_v2")
    stereo_estimator = None
    mono_mde_estimator = MonoDepthEstimator(model_type="MiDaS_small")

    results = []

    for idx in indices:
        print(f"Processing {idx}...")
        L, R, fx, B, gt_path = load_kitti_pair(KITTI_ROOT, idx)
        H, W = L.shape[:2]

        depth_gt = None
        gt_mask = None
        if gt_path:
            d_gt = load_gt_disparity_png16(gt_path)
            if d_gt is not None:
                depth_gt = depth_from_disp(d_gt, fx, B)
                gt_mask = valid_mask_from_disparity(d_gt)

        stereo = StereoDepthEstimator(StereoCalib(fx=fx, baseline_m=B), use_uncalibrated_rectify=False)
        disp_stereo = stereo.disparity(L, R)
        d_s = disp_stereo.astype(np.float32)
        d_s[d_s < 1.0] = np.nan
        depth_stereo = depth_from_disp(d_s, fx, B)

        disp_mono = mono_mde_estimator.estimate(L)

        if depth_gt is not None and gt_mask is not None:
            valid_gt_disp = (fx * B) / (depth_gt[gt_mask] + 1e-6)
            valid_mono_disp = disp_mono[gt_mask]
            scale = np.median(valid_gt_disp) / (np.median(valid_mono_disp) + 1e-6)
            disp_mono_aligned = disp_mono * scale
            depth_mono = (fx * B) / (disp_mono_aligned + 1e-6)
        else:
            depth_mono = np.zeros_like(disp_mono) + np.nan

        dets = safe_detect(detector, L, score_thr=0.5)

        for i, det in enumerate(dets):
            ymin, xmin, ymax, xmax = det["box"]
            x1, y1 = int(xmin * W), int(ymin * H)
            x2, y2 = int(xmax * W), int(ymax * H)
            cls_id = int(det.get("class_id", -1))

            mvalid = adaptive_min_valid(x1, y1, x2, y2)

            z_gt, cov_gt = roi_median_and_coverage(depth_gt, gt_mask, x1, y1, x2, y2,
                                                   min_valid=mvalid) if depth_gt is not None else (float('nan'), 0.0)

            if not (np.isfinite(z_gt) and cov_gt >= GT_COVERAGE_THR):
                continue

            z_stereo, _ = roi_median_and_coverage(depth_stereo, None, x1, y1, x2, y2, min_valid=mvalid)

            z_mono_mde, _ = roi_median_and_coverage(depth_mono, None, x1, y1, x2, y2, min_valid=mvalid)

            z_pinhole = pinhole_distance_from_bbox((x1, y1, x2, y2), cls_id, fx, COCO_HEIGHT_HINTS)

            res = {
                "idx": idx,
                "cls": cls_id,
                "z_gt": z_gt,
                "z_stereo": z_stereo,
                "z_mono_mde": z_mono_mde,
                "z_pinhole": z_pinhole
            }

            if np.isfinite(z_stereo): res["err_stereo"] = abs(z_stereo - z_gt) / z_gt
            if np.isfinite(z_mono_mde): res["err_mono_mde"] = abs(z_mono_mde - z_gt) / z_gt
            if z_pinhole is not None: res["err_pinhole"] = abs(z_pinhole - z_gt) / z_gt

            results.append(res)

    print("\n" + "=" * 50)
    print(f"Analysis Summary (Scenes: {len(indices)}, Objects: {len(results)})")
    print("=" * 50)

    methods = ["stereo", "mono_mde", "pinhole"]
    for m in methods:
        errs = [r[f"err_{m}"] for r in results if f"err_{m}" in r]
        if errs:
            print(f"{m:10s} | AbsRel Mean: {np.mean(errs):.4f} | Median: {np.median(errs):.4f} | Count: {len(errs)}")
        else:
            print(f"{m:10s} | No data")

    csv_path = "comparison_results.csv"
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed results saved to {csv_path}")


if __name__ == "__main__":
    compare_methods()
