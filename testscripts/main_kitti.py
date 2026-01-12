import cv2

from datasets.kitti_loader import load_kitti_pair
from models.depth_estimator_stereo import StereoDepthEstimator, StereoCalib
from vision.visualization import Visualizer

KITTI_ROOT = "datasets/kitti2015/training"
IDX = "000000_10"


def main():
    L, R, fx, B = load_kitti_pair(KITTI_ROOT, IDX)
    print(f"[kitti] fx={fx:.1f}, baseline={B:.3f} m")

    calib = StereoCalib(fx=fx, baseline_m=B)
    stereo = StereoDepthEstimator(calib)

    disp = stereo.disparity(L, R)
    depth = stereo.depth_meters(disp, stereo)

    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)

    vis = Visualizer()
    dets = []
    out = vis.draw(L.copy(), dets, depth_map=depth, depth_in_meters=True)

    cv2.imshow("Left (meters overlay)", out)
    cv2.imshow("Right", R)
    cv2.imshow("Disparity", disp_color)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
