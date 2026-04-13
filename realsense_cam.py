"""
RealSense D435i capture thread for data collection.

Captures at native resolution (default 1280x720) for full FOV.
Recording saves raw-resolution JPEGs to pkl — no resize in the hot loop.
Resize to output resolution (width x height) only happens on get_frame()
for live preview / inference.

Usage:
    cam = RealsenseCam(name="top_cam", width=640, height=480, fps=30)
    cam.start()
    cam.start_recording()
    ...
    cam.stop_recording()
    frames = cam.get_recorded_frames()   # raw 1280x720 JPEGs
    cam.stop()
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealsenseCam:

    def __init__(
        self,
        name: str = "top_cam",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 95,
        capture_width: int = 1280,
        capture_height: int = 720,
    ):
        if rs is None:
            raise ImportError("pyrealsense2 not installed; run: pip install pyrealsense2")

        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._cap_w = capture_width
        self._cap_h = capture_height

        self._pipeline: rs.pipeline | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        self._latest_raw: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        self._recording = False
        self._rec_buf: list[dict] = []
        self._rec_lock = threading.Lock()

        # Clock alignment: exponential moving average of (pc_ns - hw_ns).
        # Adapts continuously to eliminate drift; skips first few frames
        # to avoid pipeline warmup jitter.
        self._offset_ns: float | None = None
        self._offset_alpha: float = 0.02  # EMA smoothing factor
        self._frame_count: int = 0
        self._warmup_frames: int = 5

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        if self._running:
            return

        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self._cap_w, self._cap_h, rs.format.bgr8, self.fps)

        self._pipeline = rs.pipeline()
        self._pipeline.start(cfg)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"RealsenseCam [{self.name}] started "
              f"(capture {self._cap_w}x{self._cap_h} @ {self.fps} Hz, "
              f"output {self.width}x{self.height})")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        print(f"RealsenseCam [{self.name}] stopped")

    # ------------------------------------------------------------------
    # Recording control
    # ------------------------------------------------------------------

    def start_recording(self):
        with self._rec_lock:
            self._rec_buf = []
            self._recording = True

    def stop_recording(self):
        self._recording = False

    def get_recorded_frames(self) -> list[dict]:
        with self._rec_lock:
            return list(self._rec_buf)

    # ------------------------------------------------------------------
    # Internal capture loop — no resize, minimal work
    # ------------------------------------------------------------------

    def _loop(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            raw = np.asanyarray(color_frame.get_data())

            hw_ns = int(color_frame.get_timestamp() * 1_000_000)
            pc_now = time.time_ns()
            self._frame_count += 1

            sample_offset = pc_now - hw_ns
            if self._frame_count <= self._warmup_frames:
                # skip warmup frames for calibration (pipeline startup jitter)
                self._offset_ns = sample_offset
                ts = pc_now
            else:
                # EMA: continuous drift correction
                self._offset_ns += self._offset_alpha * (sample_offset - self._offset_ns)
                ts = hw_ns + int(self._offset_ns)

            with self._frame_lock:
                self._latest_raw = raw

            if self._recording:
                ok, jpeg = cv2.imencode(".jpg", raw, encode_params)
                if ok:
                    with self._rec_lock:
                        self._rec_buf.append({"ts": ts, "data": jpeg.tobytes()})

    # ------------------------------------------------------------------
    # Live access (resize on read, not on capture)
    # ------------------------------------------------------------------

    @property
    def latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            raw = self._latest_raw
        if raw is None:
            return None
        if (raw.shape[1], raw.shape[0]) != (self.width, self.height):
            return cv2.resize(raw, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return raw

    def get_frame(self) -> np.ndarray | None:
        return self.latest_frame
