import os

import cv2
import numpy as np


def _read_P(lines, name):
    keys = [name]
    if name == "P2": keys.append("P_rect_02")
    if name == "P3": keys.append("P_rect_03")
    for k in keys:
        for line in lines:
            if line.startswith(k + ":"):
                arr = np.fromstring(line.split(":")[1], sep=' ')
                return arr.reshape(3, 4)
    return None


def load_kitti_pair(root, idx="000000_10"):
    """
    root: шлях до 'datasets/kitti2015/training'
    idx : ім'я файлу без розширення, напр. '000000_10'
    Повертає: L, R (BGR), fx (px), baseline (м), шлях до GT disparity (або None)
    """
    L = cv2.imread(os.path.join(root, "image_2", f"{idx}.png"))
    R = cv2.imread(os.path.join(root, "image_3", f"{idx}.png"))
    if L is None or R is None:
        raise FileNotFoundError("Не знайдено image_2/image_3 для індексу " + idx)

    calib_candidates = [
        os.path.join(root, "calib_cam_to_cam", f"{idx.split('_')[0]}.txt"),
        os.path.join(root, "calib_cam_to_cam.txt"),
        os.path.join(os.path.dirname(root), "calib_cam_to_cam.txt"),
    ]
    calib_path = next((p for p in calib_candidates if os.path.exists(p)), None)
    if calib_path is None:
        raise FileNotFoundError("Не знайдено файл калібрування (calib/*.txt або calib_cam_to_cam.txt)")

    with open(calib_path, "r") as f:
        lines = f.readlines()
    P2 = _read_P(lines, "P2")
    P3 = _read_P(lines, "P3")
    if P2 is None or P3 is None:
        raise RuntimeError("У калібруванні відсутні P2/P3")

    fx = float(P2[0, 0])
    baseline = float(-P3[0, 3] / fx)

    gt_path = os.path.join(root, "disp_occ_0", f"{idx}.png")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(root, "disp_noc_0", f"{idx}.png")
    if not os.path.exists(gt_path):
        gt_path = None

    return L, R, fx, baseline, gt_path
