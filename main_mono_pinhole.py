import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from models.detector import ObjectDetector
from vision.camera import CameraStream

INTR_FILE = "camera_intrinsics.npz"

ASSUME_HFOV_DEG = 60.0

COCO_HEIGHT_HINTS: Dict[int, float] = {
    1: 1.70,
    2: 1.40,
    3: 1.50,
    4: 1.60,
    6: 2.30,
    8: 3.20,
}

MIN_BBOX_H_PX = 20


def load_intrinsics_or_guess(w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Повертає (K, dist). Якщо немає файлу intrinsics — оцінюємо fx із припущення HFOV.
    """
    if os.path.exists(INTR_FILE):
        data = np.load(INTR_FILE)
        K = data["K"].astype(np.float32)
        dist = data.get("dist", np.zeros(5, np.float32)).astype(np.float32)
        print(f"[calib] loaded intrinsics from {INTR_FILE}: fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")
        return K, dist

    fx = 0.5 * w / np.tan(np.deg2rad(ASSUME_HFOV_DEG) * 0.5)
    fy = fx
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, np.float32)
    print(f"[calib] using approximate intrinsics (HFOV≈{ASSUME_HFOV_DEG}°): fx={fx:.1f}")
    return K, dist


def pinhole_distance_from_bbox(
        box_xyxy: Tuple[int, int, int, int],
        cls_id: int,
        fx: float,
        height_hints: Dict[int, float],
        default_h: float = 1.70,
) -> Optional[float]:
    """
    Оцінка дистанції за pinhole-моделлю: Z ≈ fx * H_real / h_img.
    Використовуємо приблизну реальну висоту об’єкта (за класом COCO).
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


def draw_box_with_label(img: np.ndarray, xyxy: Tuple[int, int, int, int], label: str, color=(36, 255, 12)):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y0 + th + 6), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    print("🚗 Starting mono distance (pinhole) — main_mono2.py")

    cam = CameraStream(0)

    frame = cam.read()
    if frame is None:

        t0 = time.time()
        while frame is None and (time.time() - t0) < 2.0:
            time.sleep(0.01)
            frame = cam.read()
    if frame is None:
        cam.release()
        raise RuntimeError(
            "[camera] no frames from camera (index=0). Ensure permissions and that the camera is not busy.")
    h, w = frame.shape[:2]
    K, _ = load_intrinsics_or_guess(w, h)
    fx = float(K[0, 0])

    detector = ObjectDetector("models/ssd_mobilenet_v2")

    fps_ema = None
    while True:
        t0 = time.time()
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(frame)

        vis = frame.copy()
        if detections:
            for d in detections:

                ymin, xmin, ymax, xmax = d["box"]
                x1 = int(xmin * w)
                y1 = int(ymin * h)
                x2 = int(xmax * w)
                y2 = int(ymax * h)
                cls_id = int(d.get("class_id", -1))
                score = float(d.get("score", 0.0))

                Z = pinhole_distance_from_bbox((x1, y1, x2, y2), cls_id, fx, COCO_HEIGHT_HINTS)
                if Z is not None:
                    label = f"id={cls_id} {score:.2f} | {Z:.1f} m"
                else:
                    label = f"id={cls_id} {score:.2f}"
                draw_box_with_label(vis, (x1, y1, x2, y2), label)
        else:
            cv2.putText(vis, "No detections", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        dt = time.time() - t0
        fps_ema = (0.9 * fps_ema + 0.1 * (1.0 / dt)) if fps_ema else (1.0 / dt)
        cv2.putText(vis, f"fps={fps_ema:.1f}", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)

        cv2.imshow("Mono pinhole distance (main_mono2)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
