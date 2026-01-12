import os
import time

import cv2
import numpy as np

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
FPS = 30

MARKER_LENGTH_M = 0.057

INTR_FILE = "camera_intrinsics.npz"
ASSUME_HFOV_DEG = 60.0

ARUCO_DICT = cv2.aruco.DICT_5X5_100

PREFERRED_ID = None


def approx_intrinsics(w, h, hfov_deg=60.0):
    fx = 0.5 * w / np.tan(np.deg2rad(hfov_deg) * 0.5)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, np.float32)
    return K, dist


def load_intrinsics_or_guess(w, h):
    if os.path.exists(INTR_FILE):
        data = np.load(INTR_FILE)
        K = data["K"].astype(np.float32)
        dist = data["dist"].astype(np.float32)
        print(f"[calib] loaded intrinsics from {INTR_FILE}")
        return K, dist
    else:
        K, dist = approx_intrinsics(w, h, ASSUME_HFOV_DEG)
        print(f"[calib] using approximate intrinsics (HFOV≈{ASSUME_HFOV_DEG}°): fx={K[0, 0]:.1f}, cx={K[0, 2]:.1f}")
        return K, dist


def pose_from_marker_corners(corners_xy, marker_length_m, K, dist):
    """
    corners_xy: np.ndarray shape (4,2) в ПОРЯДКУ ArUco:
      TL=(x0,y0), TR=(x1,y1), BR=(x2,y2), BL=(x3,y3)
    Рахуємо позу планарного квадрата через solvePnP.
    """
    s = float(marker_length_m)

    objp = np.array([
        [-s / 2, s / 2, 0.0],
        [s / 2, s / 2, 0.0],
        [s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float32)

    imgp = corners_xy.astype(np.float32)

    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
    if flag is None:
        flag = cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=flag)
    if not ok:

        if flag != cv2.SOLVEPNP_ITERATIVE:
            ok2, rvec2, tvec2 = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2:
                return rvec2, tvec2
        return None, None
    return rvec, tvec


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("No frames from camera")
    h, w = frame.shape[:2]
    K, dist = load_intrinsics_or_guess(w, h)

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)

    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        new_api = True
    except Exception:

        params = cv2.aruco.DetectorParameters_create()
        detector = None
        new_api = False
        print("[info] Falling back to old detectMarkers API")

    print("[info] Controls: q/0 = quit")
    print(f"[info] Marker length set to {MARKER_LENGTH_M * 100:.1f} cm")

    fps_ema = None
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        if new_api:
            corners_list, ids, _ = detector.detectMarkers(frame)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=params)

        out = frame.copy()

        if ids is not None and len(ids) > 0:

            if PREFERRED_ID is not None:
                filt = [i for i, mid in enumerate(ids.flatten()) if int(mid) == int(PREFERRED_ID)]
            else:
                filt = list(range(len(ids)))

            for idx in filt:
                corners = corners_list[idx].reshape(-1, 2)
                rvec, tvec = pose_from_marker_corners(corners, MARKER_LENGTH_M, K, dist)

                cv2.aruco.drawDetectedMarkers(out, [corners_list[idx]], ids[idx])
                if rvec is not None and tvec is not None:

                    t = tvec.reshape(3)
                    z_forward = float(t[2])
                    dist_3d = float(np.linalg.norm(t))

                    cv2.drawFrameAxes(out, K, dist, rvec, tvec, MARKER_LENGTH_M * 0.5)

                    x1, y1 = corners[:, 0].min(), corners[:, 1].min()
                    label = f"id={int(ids[idx])}  Z={z_forward:.2f} m  |  |t|={dist_3d:.2f} m"
                    cv2.putText(out, label, (int(x1), int(max(0, y1 - 8))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    print(f"[dist] id={int(ids[idx])} -> Z={z_forward:.3f} m | |t|={dist_3d:.3f} m")
                else:
                    print("[warn] solvePnP failed on detected marker")

        dt = time.time() - t0
        fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / dt)) if fps_ema else (1.0 / dt)
        cv2.putText(out, f"fps={fps_ema:.1f}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
        cv2.putText(out, "Controls: q/0 = quit", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Single-cam ArUco distance (PnP)", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('0')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
