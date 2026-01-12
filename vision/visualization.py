import cv2
import numpy as np

class Visualizer:
    def draw(self, frame, detections, depth_map=None, depth_in_meters=False):
        """
        depth_map: або disparity, або метри — залежно від прапорця depth_in_meters.
        Для стабільності використовуємо МЕДІАНУ по ROI.
        """
        h, w, _ = frame.shape

        def median_in_roi(Z, x1, y1, x2, y2, min_valid=100):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            roi = Z[y1:y2, x1:x2]
            valid = roi[np.isfinite(roi)]
            if valid.size >= min_valid:
                return float(np.median(valid))
            return float('nan')

        for det in detections:
            ymin, xmin, ymax, xmax = det["box"]
            x1, y1 = int(xmin*w), int(ymin*h)
            x2, y2 = int(xmax*w), int(ymax*h)
            x1, y1 = max(0,x1), max(0,y1)
            x2, y2 = min(w-1,x2), min(h-1,y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"ID:{det.get('class_id','?')} {det.get('score',0):.2f}"

            if depth_map is not None and isinstance(depth_map, np.ndarray):
                val = median_in_roi(depth_map, x1, y1, x2, y2)
                if np.isfinite(val):
                    if depth_in_meters:
                        label += f" | {val:.2f} m"
                    else:
                        label += f" | d~{val:.1f}"

            cv2.putText(frame, label, (x1, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        return frame
