import cv2
import numpy as np

from models.depth_estimator_stereo import StereoDepthEstimator, StereoCalib
from models.detector import ObjectDetector
from vision.camera import DualCameraStream
from vision.visualization import Visualizer

CALIB_Z_TRUE_M = 2.00
CENTER_ROI = 80


def main():
    calib = StereoCalib(fx=500.0, baseline_m=0.38)
    cams = DualCameraStream(
        left_index=0,
        right_source="udp://@:5000?overrun_nonfatal=1&fifo_size=5000000&max_delay=0"
    )
    detector = ObjectDetector("models/ssd_mobilenet_v2")
    stereo = StereoDepthEstimator(calib, use_uncalibrated_rectify=True)
    vis = Visualizer()

    for _ in range(30):
        L, R = cams.read()
        if L is not None:
            break
    if L is not None:
        h, w = L.shape[:2]
        base_fx_640 = 500.0
        stereo.calib.fx = base_fx_640 * (w / 640.0)
        print(f"[calib] capture width={w}px -> fx={stereo.calib.fx:.1f}, baseline={stereo.calib.baseline_m} m")

    flip_right = False

    while True:
        L, R = cams.read()
        if L is None or R is None:
            continue
        if flip_right:
            R = cv2.flip(R, 1)

        disp = stereo.disparity(L, R)
        if disp is None:
            continue

        depth_m = stereo.depth_meters(disp, stereo)

        d_show = disp.copy().astype(np.float32)
        d_show[d_show < 1.0] = np.nan
        disp_norm = cv2.normalize(np.nan_to_num(d_show), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

        dets = detector.detect(L)
        left_vis = vis.draw(L.copy(), dets, depth_map=depth_m, depth_in_meters=True)

        cv2.imshow("Left + detections", left_vis)
        cv2.imshow("Right (raw)", R)
        cv2.imshow("Disparity", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("left.jpg", L)
            cv2.imwrite("right.jpg", R)
            cv2.imwrite("disparity.png", disp_color)
            print("saved: left.jpg, right.jpg, disparity.png")
        elif key == ord('f'):
            flip_right = not flip_right
            print("[info] flip_right =", flip_right)
        elif key == ord('c'):

            d_med = None
            if dets:
                h, w = L.shape[:2]
                det = max(dets, key=lambda d: (int(d["box"][3] * h) - int(d["box"][1] * h)) *
                                              (int(d["box"][2] * w) - int(d["box"][0] * w)))
                ymin, xmin, ymax, xmax = det["box"]
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                roi = disp[y1:y2, x1:x2].astype(np.float32)
                roi[roi < 1.0] = np.nan
                d_med = np.nanmedian(roi) if np.isfinite(np.nanmedian(roi)) else None
            if d_med is None or d_med <= 0:
                ch, cw = L.shape[:2]
                cx, cy = cw // 2, ch // 2
                hs = CENTER_ROI // 2
                roi = disp[cy - hs:cy + hs, cx - hs:cx + hs].astype(np.float32)
                roi[roi < 1.0] = np.nan
                d_med = np.nanmedian(roi)

            if np.isfinite(d_med) and d_med > 0:
                fx_new = (CALIB_Z_TRUE_M * d_med) / stereo.calib.baseline_m
                print(f"[calib] d_med={d_med:.2f} -> fx := {fx_new:.1f}")
                stereo.calib.fx = float(fx_new)
            else:
                print("[calib] unable to compute median disparity for calibration")

    cams.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
