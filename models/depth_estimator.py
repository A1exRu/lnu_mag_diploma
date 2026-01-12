import cv2


class DepthEstimator:
    def __init__(self):
        print("🧮 Using stereo depth estimator (simulated).")

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_map = cv2.GaussianBlur(gray, (11, 11), 0)
        return depth_map
