from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoCalib:
    fx: float = None
    baseline_m: float = None
    rectify_maps: tuple = None


class StereoDepthEstimator:
    def __init__(self, calib: StereoCalib = StereoCalib(), use_uncalibrated_rectify=True):
        self.calib = calib
        self.use_uncalibrated_rectify = use_uncalibrated_rectify and (calib.rectify_maps is None)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 8,
            blockSize=5,
            P1=8 * 3 * 5 * 5,
            P2=32 * 3 * 5 * 5,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.H1 = None
        self.H2 = None

    def _rectify_with_maps(self, L, R):
        l1, l2, r1, r2 = self.calib.rectify_maps
        Lr = cv2.remap(L, l1, l2, cv2.INTER_LINEAR)
        Rr = cv2.remap(R, r1, r2, cv2.INTER_LINEAR)
        return Lr, Rr

    def _rectify_uncalibrated(self, L, R):
        """Ректифікація без інтрінсиків: ORB → F → stereoRectifyUncalibrated → warpPerspective."""
        h, w = L.shape[:2]
        if self.H1 is None or self.H2 is None:

            orb = cv2.ORB_create(1500)
            k1, d1 = orb.detectAndCompute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), None)
            k2, d2 = orb.detectAndCompute(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), None)
            if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
                return L, R

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            m = bf.match(d1, d2)
            if len(m) < 20:
                return L, R
            m = sorted(m, key=lambda x: x.distance)[:400]
            pts1 = np.float32([k1[x.queryIdx].pt for x in m])
            pts2 = np.float32([k2[x.trainIdx].pt for x in m])

            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None:
                return L, R
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            if len(pts1) < 15:
                return L, R

            ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w, h))
            if not ok:
                return L, R
            self.H1, self.H2 = H1, H2

        Lr = cv2.warpPerspective(L, self.H1, (w, h))
        Rr = cv2.warpPerspective(R, self.H2, (w, h))
        return Lr, Rr

    def _rectify(self, L, R):
        if self.calib.rectify_maps is not None:
            return self._rectify_with_maps(L, R)
        if self.use_uncalibrated_rectify:
            return self._rectify_uncalibrated(L, R)
        return L, R

    def disparity(self, left_bgr, right_bgr):

        if left_bgr.shape[:2] != right_bgr.shape[:2]:
            right_bgr = cv2.resize(right_bgr, (left_bgr.shape[1], left_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        L, R = self._rectify(left_bgr, right_bgr)

        Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        disp = self.matcher.compute(Lg, Rg).astype(np.float32) / 16.0
        disp[disp < 0] = 0
        return disp

    def depth_meters(self, disparity, stereo):
        if not (self.calib.fx and self.calib.baseline_m):
            return None
        d = disparity.copy().astype(np.float32)
        d[d < 1.0] = np.nan
        Z = (stereo.calib.fx * stereo.calib.baseline_m) / d
        return np.clip(Z, 0.2, 50.0)
