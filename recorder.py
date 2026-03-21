import time
import threading


class ArmRecorder:
    """Records left/right arm states at a fixed frequency, each with its own timestamp."""

    def __init__(self, arm_left, arm_right, freq=100):
        self.arm_left = arm_left
        self.arm_right = arm_right

        self.freq = freq
        self.dt = 1.0 / freq

        self.running = False
        self.thread = None

        self.data = []
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        with self.lock:
            self.data = []

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"ArmRecorder started ({self.freq} Hz)")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        print("ArmRecorder stopped")

    def _loop(self):
        while self.running:
            t0 = time.time()
            ts = time.time_ns()

            left = self.arm_left.get_state()
            right = self.arm_right.get_state()

            with self.lock:
                self.data.append({"ts": ts, "left": left, "right": right})

            elapsed = time.time() - t0
            sleep = self.dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def get_data(self):
        with self.lock:
            return list(self.data)

    def is_running(self):
        return self.running
