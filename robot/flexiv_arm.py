# robot/flexiv_arm.py

import threading
import time

from .flexiv import FlexivRobot


class FlexivArm:
    """
    Thin wrapper for a single Flexiv arm.

    Teach mode keep-alive is handled here to match the C++ ControlTask():
        if (teach_active) {
            robot_->SendCartesianMotionForce(st.tcp_pose);
        }
    """

    def __init__(self, robot_sn: str, teach_freq: float = 100.0):
        print(f"Connecting to robot [{robot_sn}] ...")
        self.robot = FlexivRobot(robot_sn)
        print("Robot ready.")

        self.teach_freq = teach_freq
        self.teach_dt = 1.0 / teach_freq

        self._teach_active = False
        self._teach_thread = None
        self._teach_lock = threading.Lock()

    # =========================
    # Teach mode
    # =========================

    def _teach_loop(self) -> None:
        """
        Periodically send current tcp_pose to maintain teach mode behavior.
        """
        while True:
            with self._teach_lock:
                if not self._teach_active:
                    break

            try:
                pose = self.robot.get_tcp_pose()
                self.robot.send_cartesian_motion_force(pose)
            except Exception as e:
                print(f"Teach keep-alive loop error: {e}")
                with self._teach_lock:
                    self._teach_active = False
                break

            time.sleep(self.teach_dt)

    def enable_teach(self) -> None:
        """
        Enable kinesthetic teaching mode.
        """
        with self._teach_lock:
            if self._teach_active:
                print("Teach mode already enabled.")
                return

        self.robot.enable_teach()

        with self._teach_lock:
            self._teach_active = True
            self._teach_thread = threading.Thread(target=self._teach_loop, daemon=True)
            self._teach_thread.start()

    def disable_teach(self) -> None:
        """
        Disable kinesthetic teaching mode.
        """
        with self._teach_lock:
            if not self._teach_active:
                self.robot.disable_teach()
                return
            self._teach_active = False

        if self._teach_thread is not None:
            self._teach_thread.join()
            self._teach_thread = None

        self.robot.disable_teach()

    def teach_active(self) -> bool:
        with self._teach_lock:
            return self._teach_active

    # =========================
    # State API
    # =========================

    def get_state(self):
        return self.robot.get_state()

    def get_full_state(self):
        return self.robot.get_full_state()

    # =========================
    # Utilities
    # =========================

    def stop(self) -> None:
        self.robot.stop()

    def raw(self) -> FlexivRobot:
        return self.robot