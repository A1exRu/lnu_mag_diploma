import threading
import time
from collections import deque

import cv2


class CameraStream:
    def __init__(self, index=0, width=640, height=480, fps=30, backend=None):
        """
        index: int (local cam index) or URL (e.g., 'udp://@:5000?...')
        backend: e.g., cv2.CAP_FFMPEG for UDP/FFmpeg inputs
        """
        self.src = index
        self.backend = backend if backend is not None else 0
        self.cap = cv2.VideoCapture(self.src, self.backend)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            cv2_accel = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
            if cv2_accel is not None:
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2_accel)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source: {self.src}")

        self._queue = deque(maxlen=1)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):

        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            self._queue.append(frame)

    def read(self):
        """Return the latest frame (or None if nothing yet)."""
        return self._queue[-1] if self._queue else None

    def release(self):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=0.3)
        if self.cap:
            self.cap.release()


class DualCameraStream:
    """
    left: local camera index (e.g., 0)
    right: other local (1) or the low-latency UDP URL from ffmpeg sender
    """

    def __init__(
            self,
            left_index=0,
            right_source="udp://@:5000?pkt_size=1316&fifo_size=0&overrun_nonfatal=1",
            width=640,
            height=480,
            fps=30
    ):

        self.left = CameraStream(left_index, width, height, fps)

        self.right = CameraStream(
            right_source, width, height, fps, backend=cv2.CAP_FFMPEG
        )

    def read(self):
        return self.left.read(), self.right.read()

    def release(self):
        if self.left:  self.left.release()
        if self.right: self.right.release()
