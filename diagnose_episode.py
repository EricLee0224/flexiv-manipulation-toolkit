"""
Stage 3 诊断：验证对齐后的 episode.hdf5 质量。

报告：
- 帧数、时长、实际帧率
- 时间戳间隔分布
- 相机帧 JPEG 解码成功率
- 相机帧唯一性（是否冻屏/重复帧）
- 可选：生成带帧号的诊断视频

用法:
    python diagnose_episode.py dataset/test/episode_001/episode.hdf5
    python diagnose_episode.py dataset/test/episode_001/episode.hdf5 --video
"""

import argparse
import hashlib
import sys

import cv2
import h5py
import numpy as np

CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]


def main():
    parser = argparse.ArgumentParser(description="Stage 3: diagnose aligned episode.hdf5")
    parser.add_argument("hdf5", type=str, help="Path to episode.hdf5")
    parser.add_argument("--video", action="store_true", help="Generate diagnostic video")
    parser.add_argument("--output", type=str, default=None, help="Video output path (default: beside the hdf5 file)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Episode Diagnosis: {args.hdf5}")
    print("=" * 60)

    with h5py.File(args.hdf5, "r") as f:
        ts = f["timestamp"][:]
        n = len(ts)

        if n == 0:
            print("EMPTY episode (0 frames)")
            sys.exit(1)

        duration = (ts[-1] - ts[0]) / 1e9
        avg_hz = (n - 1) / duration if duration > 0 and n > 1 else 0

        print(f"\n--- Timeline ---")
        print(f"  frames: {n}")
        print(f"  duration: {duration:.1f}s")
        print(f"  average Hz: {avg_hz:.1f}")

        if n >= 2:
            dt = np.diff(ts) / 1e6
            print(f"  dt: median {np.median(dt):.1f}ms, p95 {np.percentile(dt, 95):.1f}ms, max {np.max(dt):.1f}ms")

            gaps = np.where(dt > 100)[0]
            if gaps.size > 0:
                print(f"  gaps >100ms: {len(gaps)}x")
                for g in gaps[:5]:
                    t_sec = (ts[g] - ts[0]) / 1e9
                    print(f"    frame {g} (t={t_sec:.1f}s): {dt[g]:.0f}ms")

        # --- robot data ---
        print(f"\n--- Robot ---")
        for arm in ["left_arm", "right_arm"]:
            if arm in f:
                q = f[f"{arm}/q"]
                print(f"  {arm}/q: shape {q.shape}")
                if "tcp_pose" in f[arm]:
                    print(f"  {arm}/tcp_pose: shape {f[arm]['tcp_pose'].shape}")
                if "gripper" in f[arm]:
                    g = f[f"{arm}/gripper"][:]
                    print(f"  {arm}/gripper: min={g.min():.4f}, max={g.max():.4f}", end="")
                    if np.all(g == 0):
                        print(" (all zero)")
                    else:
                        print()

        # --- cameras ---
        print(f"\n--- Cameras ---")

        cam_group = f["camera"]

        for name in CAM_NAMES:
            ds = cam_group[name]

            # decode check
            sample_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
            sample_idx = sorted(set(i for i in sample_idx if 0 <= i < n))
            decode_ok = 0
            sizes = []
            for i in sample_idx:
                buf = ds[i]
                sizes.append(len(buf))
                if len(buf) > 0:
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if img is not None:
                        decode_ok += 1

            avg_kb = np.mean(sizes) / 1024 if sizes else 0
            print(f"  [{name}] decode {decode_ok}/{len(sample_idx)}, avg JPEG {avg_kb:.0f} KB", end="")

            # uniqueness
            spread = np.linspace(0, n - 1, min(30, n), dtype=int)
            hashes = []
            for i in spread:
                buf = ds[i]
                if len(buf) > 0:
                    hashes.append(hashlib.md5(buf.tobytes()).hexdigest())

            unique = len(set(hashes))
            total = len(hashes)
            dup_pct = (1 - unique / total) * 100 if total > 0 else 0
            print(f", {unique}/{total} unique", end="")
            if dup_pct > 50:
                print(f" ⚠️  {dup_pct:.0f}% duplicates")
            else:
                print(" ✓")

            # consecutive duplicate analysis
            if n > 10:
                check_n = min(n, 300)
                prev_h = None
                max_run = 0
                cur_run = 0
                for i in range(check_n):
                    buf = ds[i]
                    h = hashlib.md5(buf.tobytes()).hexdigest() if len(buf) > 0 else "empty"
                    if h == prev_h:
                        cur_run += 1
                        max_run = max(max_run, cur_run)
                    else:
                        cur_run = 1
                    prev_h = h

                if max_run > 3:
                    print(f"    longest consecutive dup run: {max_run} frames")

        # --- diagnostic video ---
        if args.video:
            print(f"\n--- Generating diagnostic video ---")
            import os
            if args.output:
                out_path = args.output
            else:
                out_path = os.path.join(os.path.dirname(args.hdf5), "diag_episode.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, 30, (1280, 960))

            # pre-read all camera data
            cam_data = {}
            for cam_name in CAM_NAMES:
                print(f"  loading {cam_name} ...")
                cam_data[cam_name] = [cam_group[cam_name][i] for i in range(n)]

            def decode(buf):
                if len(buf) == 0:
                    return np.zeros((480, 640, 3), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

            for i in range(n):
                imgs = [decode(cam_data[name][i]) for name in CAM_NAMES]

                for j, img in enumerate(imgs):
                    h, w = img.shape[:2]
                    if (w, h) != (640, 480):
                        imgs[j] = cv2.resize(img, (640, 480))

                # add frame number + timestamp overlay
                dt_ms = (ts[i] - ts[0]) / 1e6
                for j, img in enumerate(imgs):
                    cv2.putText(img, f"#{i} {dt_ms:.0f}ms", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, CAM_NAMES[j], (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                top = np.hstack((imgs[0], imgs[1]))
                bottom = np.hstack((imgs[2], imgs[3]))
                grid = np.vstack((top, bottom))
                writer.write(grid)

                if i % 100 == 0:
                    print(f"  frame {i}/{n}")

            writer.release()
            print(f"  → {out_path}")

    print()


if __name__ == "__main__":
    main()
