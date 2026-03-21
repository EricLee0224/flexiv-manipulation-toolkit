import threading


class Cyber2PCObserver:

    def __init__(self):
        self._running = False
        self._thread = None

    def start(self):

        if self._running:
            return

        self._running = True

        def _run():
            import asyncio
            import receiver_gripper_cam

            asyncio.run(receiver_gripper_cam.main())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

        print("✅ Cyber2PCObserver started (receiver in-process)")

    def stop(self):
        self._running = False
        print("🛑 Cyber2PCObserver stopped")

    # --- live buffer access (for monitoring / preview) ---

    def get_cam(self):
        from receiver_gripper_cam import shared_buffer
        return shared_buffer["cam"]

    def get_gripper(self):
        from receiver_gripper_cam import shared_buffer
        return shared_buffer["gripper"]

    def get_latest(self):
        from receiver_gripper_cam import shared_buffer
        return shared_buffer

    # --- recording control ---

    def start_recording(self):
        from receiver_gripper_cam import start_recording
        start_recording()

    def stop_recording(self):
        from receiver_gripper_cam import flush_and_stop_recording
        flush_and_stop_recording()

    def get_recorded_cam(self):
        from receiver_gripper_cam import get_recorded_cam
        return get_recorded_cam()

    def get_recorded_gripper(self):
        from receiver_gripper_cam import get_recorded_gripper
        return get_recorded_gripper()
