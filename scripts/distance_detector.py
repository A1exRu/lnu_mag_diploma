"""
Thesis demo: Object detection (YOLOv8) + distance estimation with OpenCV.

Features
- Detects objects with YOLOv8 (Ultralytics).
- Two distance-estimation modes:
  A) Stereo (preferred on KITTI): Z = f * B / disparity (block matching).
  B) Monocular fallback: pinhole approximation using known real-world object sizes.
- Draws bounding boxes with class, confidence, and distance overlay.
- Works on single images, folders, or video/webcam streams.

Usage
------


























"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None


@dataclass
class MonoSizeHint:
    """Approximate real-world object dimensions (meters) for monocular fallback.
    Values are conservative averages and can be tuned for your dataset.
    """
    width_m: float
    height_m: float


DEFAULT_SIZE_HINTS: Dict[str, MonoSizeHint] = {

    "person": MonoSizeHint(width_m=0.55, height_m=1.70),
    "car": MonoSizeHint(width_m=1.80, height_m=1.45),
    "truck": MonoSizeHint(width_m=2.50, height_m=3.00),
    "bus": MonoSizeHint(width_m=2.50, height_m=3.20),
    "bicycle": MonoSizeHint(width_m=0.60, height_m=1.30),
    "motorcycle": MonoSizeHint(width_m=0.80, height_m=1.20),
}


def parse_kitti_calib(calib_path: str) -> Dict[str, np.ndarray]:
    """Parses KITTI calib_cam_to_cam.txt style files.

    Returns a dict with entries like 'P_rect_02', 'P_rect_03' as 3x4 matrices.
    """
    mats: Dict[str, np.ndarray] = {}
    if not calib_path:
        return mats
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            val = val.strip()
            try:
                arr = np.fromstring(val, sep=" ")
            except ValueError:
                continue
            if arr.size in (9, 12):
                rows = 3
                cols = arr.size // 3
                mats[key.strip()] = arr.reshape(rows, cols)
    return mats


def fx_from_P(P: np.ndarray) -> float:
    """Extract focal length fx from 3x4 projection matrix P."""
    if P is None or P.shape != (3, 4):
        return np.nan
    return float(P[0, 0])


def compute_disparity_sgbm(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute disparity using Semi-Global Block Matching (OpenCV SGBM)."""

    gl = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gr = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    min_disp = 0
    num_disp = 128
    block_size = 5

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(gl, gr).astype(np.float32) / 16.0
    return disp


def z_from_disparity(disparity: float, fx: float, baseline_m: float, eps: float = 1e-6) -> float:
    """Z = f * B / d, meters. Returns np.inf if disparity is ~0."""
    if disparity <= eps:
        return float("inf")
    return (fx * baseline_m) / disparity


def estimate_mono_distance(
        bbox_xyxy: Tuple[int, int, int, int],
        fx: Optional[float],
        size_hints: Dict[str, MonoSizeHint],
        class_name: str,
) -> Optional[float]:
    """Pinhole-geometry heuristic using known real height.

    Z ≈ (H_real * f) / h_pixels.
    If fx is unknown, use image width as a crude proxy (will be poor but non-crashing).
    """
    (x1, y1, x2, y2) = bbox_xyxy
    h_px = max(1, y2 - y1)
    hint = size_hints.get(class_name)
    if hint is None:
        return None
    f = fx if (fx and fx > 0) else None
    if f is None:
        f = 1000.0
    Z = (hint.height_m * f) / float(h_px)
    return float(Z)


def put_label(img: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_det(
        img: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        cls_name: str,
        conf: float,
        distance_m: Optional[float],
) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
    label = f"{cls_name} {conf:.2f}"
    if distance_m is not None and np.isfinite(distance_m):
        label += f" | {distance_m:.1f} m"
    put_label(img, label, (x1, max(15, y1 - 5)))


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt", device: Optional[str] = None):
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. pip install ultralytics")
        self.model = YOLO(model_path)
        if device:

            self.device = device
        else:
            self.device = None

    def infer(self, image_bgr: np.ndarray, conf: float = 0.25):

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=image_rgb, verbose=False, conf=conf, device=self.device)
        return results[0]


@dataclass
class Calib:
    fx: Optional[float] = None
    baseline_m: Optional[float] = None


def load_calibration(calib_path: Optional[str], baseline_m: Optional[float]) -> Calib:
    fx = None
    if calib_path and os.path.isfile(calib_path):
        mats = parse_kitti_calib(calib_path)

        P = mats.get("P_rect_02", None)
        if P is None:
            P = mats.get("P2", None)
        fx = fx_from_P(P) if P is not None else None
    return Calib(fx=fx, baseline_m=baseline_m)


def central_disparity_for_bbox(disp: np.ndarray, xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = max(0, x1);
    y1 = max(0, y1)
    x2 = min(disp.shape[1] - 1, x2);
    y2 = min(disp.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    cx1 = x1 + (x2 - x1) // 4
    cx2 = x2 - (x2 - x1) // 4
    cy1 = y1 + (y2 - y1) // 4
    cy2 = y2 - (y2 - y1) // 4
    roi = disp[cy1:cy2, cx1:cx2]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def process_pair(
        left_bgr: np.ndarray,
        right_bgr: Optional[np.ndarray],
        det: Detector,
        calib: Calib,
        size_hints: Dict[str, MonoSizeHint] = DEFAULT_SIZE_HINTS,
        conf_thr: float = 0.25,
) -> np.ndarray:
    """Runs detection and overlays distance estimations.

    If right image and (calib.fx & calib.baseline_m) are provided -> stereo distances.
    Otherwise -> monocular fallback per class size hints.
    """

    r = det.infer(left_bgr, conf=conf_thr)

    class_names = det.model.names
    boxes = r.boxes

    disparity = None
    can_stereo = right_bgr is not None and calib.fx and calib.baseline_m
    if can_stereo:
        disparity = compute_disparity_sgbm(left_bgr, right_bgr)

    vis = left_bgr.copy()
    if boxes is None or boxes.shape[0] == 0:
        put_label(vis, "No detections", (10, 30))
        return vis

    for i in range(len(boxes)):
        b = boxes[i]

        xyxy = tuple(map(int, b.xyxy[0].tolist()))
        conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
        cls_id = int(b.cls[0]) if hasattr(b, 'cls') else 0
        cls_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else str(cls_id)

        distance_m: Optional[float] = None
        if can_stereo and disparity is not None:
            d = central_disparity_for_bbox(disparity, xyxy)
            distance_m = z_from_disparity(d, float(calib.fx), float(calib.baseline_m))
        else:
            distance_m = estimate_mono_distance(xyxy, calib.fx, size_hints, cls_name)

        draw_det(vis, xyxy, cls_name, conf, distance_m)

    if disparity is not None:
        disp_norm = np.clip((disparity - np.nanmin(disparity)) / (np.nanmax(disparity) - np.nanmin(disparity) + 1e-9),
                            0, 1)
        disp_small = (disp_norm * 255).astype(np.uint8)
        disp_small = cv2.applyColorMap(disp_small, cv2.COLORMAP_INFERNO)
        h, w = vis.shape[:2]
        thumb_w = max(160, w // 5)
        thumb_h = int(thumb_w * h / w)
        disp_small = cv2.resize(disp_small, (thumb_w, thumb_h))
        vis[0:thumb_h, w - thumb_w:w] = disp_small
        put_label(vis, "disparity", (w - thumb_w + 8, 20))

    put_label(vis, "Mode: Stereo" if can_stereo else "Mode: Monocular (heuristic)", (10, 20))
    return vis


def imread_rgb_or_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="YOLOv8 + OpenCV distance estimation (KITTI-ready)")
    g = ap.add_argument_group("Inputs")
    g.add_argument("--left_path", type=str, default=None, help="Left image path OR single image path")
    g.add_argument("--right_path", type=str, default=None, help="Right image path (stereo)")
    g.add_argument("--left_dir", type=str, default=None, help="Directory of left images")
    g.add_argument("--right_dir", type=str, default=None, help="Directory of right images (paired by filename)")
    g.add_argument("--video", type=str, default=None, help="Video path or webcam index (e.g. 0)")

    m = ap.add_argument_group("Model & calibration")
    m.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model path/name")
    m.add_argument("--device", type=str, default=None, help="cuda:0 / cpu")
    m.add_argument("--calib_path", type=str, default=None, help="KITTI calib_cam_to_cam.txt")
    m.add_argument("--baseline_m", type=float, default=None, help="Stereo baseline in meters (e.g., 0.537)")
    m.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")

    o = ap.add_argument_group("Output")
    o.add_argument("--save", type=str, default=None, help="Save a single image result here")
    o.add_argument("--out_dir", type=str, default=None, help="Save multiple results to this folder")
    o.add_argument("--show", action="store_true", help="Show window")

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    det = Detector(model_path=args.model, device=args.device)
    calib = load_calibration(args.calib_path, args.baseline_m)

    if args.video is not None:

        src = args.video
        cap = cv2.VideoCapture(0 if src.isdigit() else src)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                vis = process_pair(frame, None, det, calib, conf_thr=args.conf)
                if args.show:
                    cv2.imshow("Det+Distance", vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            if args.save:
                save_image(args.save, vis)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return

    if args.left_path and os.path.isfile(args.left_path):
        left = imread_rgb_or_bgr(args.left_path)
        right = imread_rgb_or_bgr(args.right_path) if args.right_path and os.path.isfile(args.right_path) else None
        vis = process_pair(left, right, det, calib, conf_thr=args.conf)
        if args.show:
            cv2.imshow("Det+Distance", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.save:
            save_image(args.save, vis)
        else:

            base, ext = os.path.splitext(args.left_path)
            save_image(base + "_vis" + ext, vis)
        return

    if args.left_dir and os.path.isdir(args.left_dir):
        right_dir = args.right_dir if (args.right_dir and os.path.isdir(args.right_dir)) else None
        out_dir = args.out_dir or os.path.join("runs", "vis")
        os.makedirs(out_dir, exist_ok=True)
        names = sorted([n for n in os.listdir(args.left_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))])
        for name in tqdm(names, desc="Processing"):
            lp = os.path.join(args.left_dir, name)
            rp = os.path.join(right_dir, name) if right_dir else None
            left = imread_rgb_or_bgr(lp)
            right = imread_rgb_or_bgr(rp) if (rp and os.path.isfile(rp)) else None
            vis = process_pair(left, right, det, calib, conf_thr=args.conf)
            save_image(os.path.join(out_dir, name), vis)
        print(f"Saved {len(names)} results to {out_dir}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
