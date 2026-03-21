# dataset_writer.py

import h5py
import numpy as np
import cv2


class DatasetWriter:

    def __init__(self, file_path):
        self.file_path = file_path

    # --------------------------------------------------
    # 🔥 图像编码（稳定版）
    # --------------------------------------------------

    def encode_image(self, img):
        """
        numpy (H, W, 3) → JPEG bytes (uint8 1D)
        """

        if img is None:
            return np.empty((0,), dtype=np.uint8)

        # ⚠️ 保证是uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        success, buf = cv2.imencode(".jpg", img)

        if not success:
            return np.empty((0,), dtype=np.uint8)

        # ⚠️ flatten（非常关键）
        return buf.reshape(-1).astype(np.uint8)

    # --------------------------------------------------

    def write(self, data):

        if len(data) == 0:
            print("No data to write.")
            return

        n = len(data)

        timestamps = np.zeros((n,), dtype=np.int64)

        left_q = np.zeros((n, 7), dtype=np.float32)
        right_q = np.zeros((n, 7), dtype=np.float32)

        gripper_left = np.zeros((n,), dtype=np.float32)
        gripper_right = np.zeros((n,), dtype=np.float32)

        with h5py.File(self.file_path, "w") as f:

            # --------------------
            # camera（vlen uint8）
            # --------------------

            cam_group = f.create_group("camera")

            vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))

            cam_left_cam0 = cam_group.create_dataset("left_cam0", (n,), dtype=vlen_dtype)
            cam_left_cam1 = cam_group.create_dataset("left_cam1", (n,), dtype=vlen_dtype)
            cam_right_cam0 = cam_group.create_dataset("right_cam0", (n,), dtype=vlen_dtype)
            cam_right_cam1 = cam_group.create_dataset("right_cam1", (n,), dtype=vlen_dtype)

            # --------------------
            # write frames
            # --------------------

            for i, frame in enumerate(data):

                timestamps[i] = frame["t"]

                # --------------------
                # robot
                # --------------------
                left_q[i] = frame["left"]["q"]
                right_q[i] = frame["right"]["q"]

                gripper_left[i] = frame["gripper_left"] if frame["gripper_left"] is not None else 0.0
                gripper_right[i] = frame["gripper_right"] if frame["gripper_right"] is not None else 0.0

                # --------------------
                # camera（关键）
                # --------------------
                cam = frame["camera"]

                cam_left_cam0[i] = self.encode_image(cam.get("left_cam0"))
                cam_left_cam1[i] = self.encode_image(cam.get("left_cam1"))
                cam_right_cam0[i] = self.encode_image(cam.get("right_cam0"))
                cam_right_cam1[i] = self.encode_image(cam.get("right_cam1"))

            # --------------------
            # robot states
            # --------------------

            f.create_dataset("timestamp", data=timestamps)

            g_left = f.create_group("left_arm")
            g_left.create_dataset("q", data=left_q)
            g_left.create_dataset("gripper", data=gripper_left)

            g_right = f.create_group("right_arm")
            g_right.create_dataset("q", data=right_q)
            g_right.create_dataset("gripper", data=gripper_right)

        print(f"Dataset saved to {self.file_path}")