import cv2
import numpy as np

from datasets.kitti_loader import load_kitti_pair
from models.depth_estimator_stereo import StereoDepthEstimator, StereoCalib

KITTI_ROOT = "datasets/kitti2015/training"
IDX = "000000_10"


def median_distance_in_roi(Z, x1, y1, x2, y2, min_valid=50):
    h, w = Z.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return float('nan'), (x1, y1, x2, y2)
    roi = Z[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi)]
    if valid.size >= min_valid:
        return float(np.median(valid)), (x1, y1, x2, y2)
    return float('nan'), (x1, y1, x2, y2)


def main():
    L, R, fx, B, gt_path = load_kitti_pair(KITTI_ROOT, IDX)
    print(f"[kitti] fx={fx:.1f} px, baseline={B:.3f} m, gt={'yes' if gt_path else 'no'}")

    stereo = StereoDepthEstimator(StereoCalib(fx=fx, baseline_m=B), use_uncalibrated_rectify=False)

    disp = stereo.disparity(L, R)
    d = disp.astype(np.float32)
    d[d < 1.0] = np.nan
    Z = (fx * B) / d
    Z = np.clip(Z, 0.2, 200.0)

    d_vis = np.nan_to_num(d)
    d_vis = cv2.normalize(d_vis, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    disp_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_TURBO)

    out = L.copy()

    h, w = L.shape[:2]
    Z_med, (x1, y1, x2, y2) = median_distance_in_roi(Z, w // 2 - 50, h // 2 - 50, w // 2 + 50, h // 2 + 50)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    txt = f"{Z_med:.2f} m" if np.isfinite(Z_med) else "n/a"
    cv2.putText(out, txt, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    print(f"[kitti] center ROI -> {txt}")

    click_pt = [None, None]

    def on_mouse(event, x, y, flags, param):
        nonlocal out
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt[0], click_pt[1] = x, y
            out = L.copy()
            half = 30
            zc, (xx1, yy1, xx2, yy2) = median_distance_in_roi(Z, x - half, y - half, x + half, y + half, min_valid=25)
            cv2.rectangle(out, (xx1, yy1), (xx2, yy2), (0, 200, 0), 2)
            label = f"{zc:.2f} m" if np.isfinite(zc) else "n/a"
            cv2.putText(out, label, (xx1, max(0, yy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            print(f"[kitti] click @({x},{y}) -> {label}")

    cv2.namedWindow("Left + depth")
    cv2.setMouseCallback("Left + depth", on_mouse)

    while True:
        cv2.imshow("Left + depth", out)
        cv2.imshow("Right", R)
        cv2.imshow("Disparity", disp_color)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('q'):
            break

    if gt_path:
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is not None:
            gt = gt.astype(np.float32) / 256.0
            m = (gt > 0) & np.isfinite(d)
            if np.any(m):
                Z_est = (fx * B) / np.maximum(d, 1e-6)
                Z_gt = (fx * B) / np.maximum(gt, 1e-6)
                abs_rel = np.mean(np.abs(Z_est[m] - Z_gt[m]) / (Z_gt[m] + 1e-6))
                print(f"[kitti] AbsRel(depth) = {abs_rel:.3f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
