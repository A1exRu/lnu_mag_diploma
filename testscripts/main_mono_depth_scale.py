import time

import cv2
import numpy as np

from models.depth_estimator_mono import MonoDepthEstimator

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
FPS = 30

REF_Z_TRUE_M = 2.00

CENTER_HALF = 40
CLICK_HALF = 30

DEPTH_VIZ_CLIP_M = (0.2, 30.0)


def _median_in_roi(M, x1, y1, x2, y2, min_valid=50):
    """Медіана по валідних значеннях матриці M у ROI."""
    H, W = M.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return float('nan')
    roi = M[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size >= min_valid:
        return float(np.median(valid))
    return float('nan')


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[warn] Cannot open camera {CAM_INDEX}. Trying default...")
        cap = cv2.VideoCapture(0)

    try:
        estimator = MonoDepthEstimator(model_type="MiDaS_small")
    except Exception as e:
        print(f"❌ Error initializing estimator: {e}")
        return

    print(
        "[controls] q/0 = quit | c = calibrate scale at center ROI | [ / ] = ref Z -/+ 0.1 m | click = distance in ROI")

    scale = 1000.0
    ref_Z = REF_Z_TRUE_M
    click_pt = [None, None]
    fps_ema = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pt[0], click_pt[1] = x, y

    win_name = "MonoDepth + Scale (MiDaS)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005);
            continue

        disp_rel = estimator.estimate(frame)

        eps = 1e-6
        depth_m = scale / (disp_rel + eps)

        H, W = frame.shape[:2]
        cx, cy = W // 2, H // 2
        x1, y1 = cx - CENTER_HALF, cy - CENTER_HALF
        x2, y2 = cx + CENTER_HALF, cy + CENTER_HALF

        center_disp = _median_in_roi(disp_rel, x1, y1, x2, y2, min_valid=100)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if click_pt[0] is not None:
            x, y = click_pt
            xx1, yy1 = x - CLICK_HALF, y - CLICK_HALF
            xx2, yy2 = x + CLICK_HALF, y + CLICK_HALF
            z_click = _median_in_roi(depth_m, xx1, yy1, xx2, yy2, min_valid=50)
            if np.isfinite(z_click):
                print(f"[click] ({x},{y}) -> {z_click:.2f} m")

                cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), (0, 200, 0), 2)
                cv2.putText(frame, f"{z_click:.2f}m", (xx1, max(0, yy1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            click_pt[0] = None

        disp_viz = cv2.normalize(disp_rel, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        disp_color = cv2.applyColorMap(disp_viz, cv2.COLORMAP_TURBO)

        dt = time.time() - t0
        fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / dt)) if fps_ema else (1.0 / dt)
        legend = f"MiDaS Small | scale={scale:.1f} | ref={ref_Z:.2f}m | fps={fps_ema:.1f}"
        cv2.putText(frame, legend, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 220, 255), 2)
        cv2.putText(frame, "c: calibrate | [/]: ref Z | click: measure", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(win_name, frame)
        cv2.imshow("Disparity (MiDaS)", disp_color)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('0')):
            break
        elif key == ord('c'):
            if np.isfinite(center_disp) and center_disp > 0:

                scale = float(ref_Z * center_disp)
                print(f"[calib] center_disp={center_disp:.2f} -> scale := {scale:.2f} (ref={ref_Z:.2f}m)")
            else:
                print("[calib] center_disp invalid")
        elif key == ord('['):
            ref_Z = max(0.1, ref_Z - 0.1)
        elif key == ord(']'):
            ref_Z = ref_Z + 0.1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
