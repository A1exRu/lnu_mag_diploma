import time

import numpy as np


class FPSMeter:
    def __init__(self):
        self.last = time.time()

    def measure(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        return 1.0 / dt if dt > 0 else 0.0


def compute_depth_metrics(gt, pred, mask=None):
    """
    Обчислює стандартні метрики глибини: AbsRel, SqRel, RMSE, RMSElog, SILog, delta thresholds.
    gt, pred: numpy arrays
    mask: boolean mask of valid pixels
    """
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    gt = gt[valid]
    pred = pred[valid]

    if len(gt) == 0:
        return {
            "abs_rel": float('nan'), "sq_rel": float('nan'),
            "rmse": float('nan'), "rmse_log": float('nan'),
            "silog": float('nan'),
            "delta1": float('nan'), "delta2": float('nan'), "delta3": float('nan')
        }

    thresh = np.maximum((gt / pred), (pred / gt))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err_log = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err_log ** 2) - np.mean(err_log) ** 2) * 100

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "silog": silog,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3
    }
