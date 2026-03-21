# validate_hdf5.py

import os
import sys
import numpy as np
import h5py
import cv2


# --------------------------------------------------
# 打印结构
# --------------------------------------------------

def print_tree(group, prefix=""):
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            print(f"[GROUP]   {path}")
            print_tree(item, path)
        elif isinstance(item, h5py.Dataset):
            print(f"[DATASET] {path} | shape={item.shape} dtype={item.dtype}")


# --------------------------------------------------
# 主验证
# --------------------------------------------------

def validate_file(file_path: str):

    if not os.path.exists(file_path):
        print(f"ERROR: file does not exist: {file_path}")
        return 1

    print("=" * 60)
    print(f"Validating HDF5 file: {file_path}")
    print("=" * 60)

    with h5py.File(file_path, "r") as f:

        # --------------------------------------------------
        # 结构
        # --------------------------------------------------
        print("\n--- HDF5 Tree ---")
        print_tree(f)

        required = [
            "timestamp",
            "left_arm/q",
            "left_arm/gripper",
            "right_arm/q",
            "right_arm/gripper",
            "camera/left_cam0",
            "camera/left_cam1",
            "camera/right_cam0",
            "camera/right_cam1",
        ]

        print("\n--- Required Dataset Check ---")
        missing = []
        for path in required:
            if path not in f:
                print(f"[MISSING] {path}")
                missing.append(path)
            else:
                print(f"[OK]      {path}")

        if missing:
            print("\n❌ Missing required datasets")
            return 1

        # --------------------------------------------------
        # 读取
        # --------------------------------------------------
        ts = f["timestamp"][:]

        left_q = f["left_arm/q"][:]
        right_q = f["right_arm/q"][:]

        gl = f["left_arm/gripper"][:]
        gr = f["right_arm/gripper"][:]

        cam0 = f["camera/left_cam0"]
        cam1 = f["camera/left_cam1"]
        cam2 = f["camera/right_cam0"]
        cam3 = f["camera/right_cam1"]

        n = len(ts)

        print("\n--- Shape Check ---")
        print("frames:", n)
        print("left_q :", left_q.shape)
        print("right_q:", right_q.shape)

        if left_q.shape != (n, 7) or right_q.shape != (n, 7):
            print("❌ robot shape mismatch")
            return 1
        else:
            print("✅ robot shape OK")

        # --------------------------------------------------
        # timestamp
        # --------------------------------------------------
        print("\n--- Timestamp Check ---")

        if n < 2:
            print("❌ too few frames")
            return 1

        dt = np.diff(ts)

        if np.any(dt <= 0):
            print("❌ timestamp not strictly increasing")
            return 1
        else:
            print("✅ timestamp monotonic")

        mean_dt = np.mean(dt)
        freq = 1e9 / mean_dt

        print(f"Mean dt : {mean_dt:.2f} ns")
        print(f"Freq    : {freq:.2f} Hz")

        # --------------------------------------------------
        # gripper
        # --------------------------------------------------
        print("\n--- Gripper Check ---")

        print("left  min/max:", gl.min(), gl.max())
        print("right min/max:", gr.min(), gr.max())

        if np.all(gl == 0) and np.all(gr == 0):
            print("⚠️ gripper never moved (可能正常)")
        else:
            print("✅ gripper has signal")

        # --------------------------------------------------
        # camera decode（最关键）
        # --------------------------------------------------
        print("\n--- Camera Decode Check ---")

        def check_cam(ds, name):

            valid = 0

            for i in range(min(10, n)):
                buf = ds[i]

                if len(buf) == 0:
                    continue

                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                if img is not None:
                    valid += 1

            print(f"{name}: valid {valid}/10")

            if valid == 0:
                print(f"❌ {name} decode failed")
                return False
            return True

        ok = True
        ok &= check_cam(cam0, "left_cam0")
        ok &= check_cam(cam1, "left_cam1")
        ok &= check_cam(cam2, "right_cam0")
        ok &= check_cam(cam3, "right_cam1")

        if not ok:
            print("\n❌ camera decode failed")
            return 1
        else:
            print("\n✅ camera decode OK")

        # --------------------------------------------------
        # camera：是否“冻屏”（多帧 JPEG 完全一致 → 多半是采集/解码，不是时间轴对齐）
        # --------------------------------------------------
        print("\n--- Camera Motion / Uniqueness Check ---")

        def jpeg_bytes_hash(ds, idx):
            buf = ds[idx]
            if len(buf) == 0:
                return None
            return hash(buf.tobytes())

        sample_idx = [0, 1, min(50, n - 1), n // 2, n - 1]
        sample_idx = sorted(set(i for i in sample_idx if 0 <= i < n))

        hashes = [jpeg_bytes_hash(cam0, i) for i in sample_idx]
        non_none = [h for h in hashes if h is not None]
        unique = len(set(non_none))

        print(f"left_cam0 sampled indices: {sample_idx}")
        print(f"unique non-empty JPEG hashes among samples: {unique}/{len(non_none)}")

        if n >= 3:
            same_as_first = sum(
                1
                for i in range(min(30, n))
                if jpeg_bytes_hash(cam0, i) == jpeg_bytes_hash(cam0, 0)
                and jpeg_bytes_hash(cam0, i) is not None
            )
            print(
                f"first {min(30, n)} frames: {same_as_first} share the same hash as frame 0"
            )
            if same_as_first == min(30, n) and jpeg_bytes_hash(cam0, 0) is not None:
                print(
                    "⚠️ 图像字节在多帧上完全相同 → 播放会像定格；请查 WebSocket/H265 解码/shared_buffer 是否在更新"
                )
            elif unique >= 2:
                print("✅ JPEG 内容在采样帧之间有差异， HDF5 里视频在换帧")

        # --------------------------------------------------
        # sample
        # --------------------------------------------------
        print("\n--- Sample Frame ---")

        i = 0
        print("timestamp:", ts[i])
        print("left_q   :", left_q[i])
        print("right_q  :", right_q[i])
        print("gripper  :", gl[i], gr[i])

        buf = cam0[i]
        if len(buf) > 0:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            print("img shape:", img.shape)

    print("\n🎉 Validation PASSED")
    return 0


# --------------------------------------------------

def main():
    if len(sys.argv) < 2:
        path = "demo.h5"
    else:
        path = sys.argv[1]

    raise SystemExit(validate_file(path))


if __name__ == "__main__":
    main()