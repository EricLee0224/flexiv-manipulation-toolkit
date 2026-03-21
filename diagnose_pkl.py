"""
Stage 2 诊断：验证 pkl 录制质量。

分析一个 episode 目录下的所有 pkl 文件，报告：
- 每个源的帧数、时长、实际帧率
- 时间戳间隔分布（median / p95 / max）
- 大间隔（gap）的位置
- 相机帧是否在变化（不是冻屏）
- JPEG 能否正常解码

用法:
    python diagnose_pkl.py dataset/test/episode_001
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_timestamps(ts, label):
    n = len(ts)
    if n == 0:
        print(f"  [{label}] EMPTY — no data")
        return

    duration = (ts[-1] - ts[0]) / 1e9
    avg_hz = (n - 1) / duration if duration > 0 and n > 1 else 0

    print(f"  [{label}] {n} samples, {duration:.1f}s, avg {avg_hz:.1f} Hz")

    if n < 2:
        return

    dt = np.diff(ts) / 1e6  # ms
    print(f"    dt: median {np.median(dt):.1f}ms, p95 {np.percentile(dt, 95):.1f}ms, max {np.max(dt):.1f}ms")

    gaps = np.where(dt > 100)[0]
    if gaps.size > 0:
        print(f"    gaps >100ms: {len(gaps)}x")
        for g in gaps[:8]:
            t_sec = (ts[g] - ts[0]) / 1e9
            print(f"      frame {g} (t={t_sec:.1f}s): {dt[g]:.0f}ms")
    else:
        print(f"    no gaps >100ms")


def analyze_camera_pkl(path, name):
    data = load_pkl(path)
    ts = data["timestamps"]
    frames = data["frames"]

    analyze_timestamps(ts, name)

    n = len(frames)
    if n == 0:
        return

    # decode check: first, middle, last
    sample_indices = [0, n // 2, n - 1]
    decode_ok = 0
    for i in sample_indices:
        buf = np.frombuffer(frames[i], dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is not None:
            decode_ok += 1

    print(f"    JPEG decode: {decode_ok}/{len(sample_indices)} sample frames OK")

    # uniqueness: hash a spread of frames
    sample_spread = np.linspace(0, n - 1, min(20, n), dtype=int)
    hashes = set()
    for i in sample_spread:
        h = hashlib.md5(frames[i]).hexdigest()
        hashes.add(h)

    print(f"    uniqueness: {len(hashes)} distinct images among {len(sample_spread)} samples", end="")
    if len(hashes) <= 2:
        print(" ⚠️  POSSIBLE FREEZE")
    else:
        print(" ✓")

    # consecutive duplicate runs
    max_run = 1
    cur_run = 1
    prev_h = hashlib.md5(frames[0]).hexdigest()
    for i in range(1, min(n, 200)):
        h = hashlib.md5(frames[i]).hexdigest()
        if h == prev_h:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
        prev_h = h

    if max_run > 3:
        print(f"    longest consecutive duplicate run: {max_run} frames ⚠️")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: diagnose pkl recording quality")
    parser.add_argument("episode_dir", type=str, help="Path to episode directory")
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    if not ep.is_dir():
        print(f"Not a directory: {ep}")
        sys.exit(1)

    print("=" * 60)
    print(f"PKL Diagnosis: {ep}")
    print("=" * 60)

    # --- arm ---
    arm_path = ep / "arm.pkl"
    if arm_path.exists():
        arm = load_pkl(arm_path)
        print("\n--- Arm ---")
        analyze_timestamps(arm["timestamps"], "arm")
    else:
        print("\n--- Arm: MISSING ---")

    # --- gripper ---
    gripper_path = ep / "gripper.pkl"
    if gripper_path.exists():
        gripper = load_pkl(gripper_path)
        print("\n--- Gripper ---")
        for side in ("left", "right"):
            ts = gripper.get(f"{side}_timestamps", np.empty(0))
            analyze_timestamps(ts, f"gripper_{side}")
    else:
        print("\n--- Gripper: MISSING ---")

    # --- cameras ---
    print("\n--- Cameras ---")
    for name in CAM_NAMES:
        path = ep / f"{name}.pkl"
        if path.exists():
            analyze_camera_pkl(path, name)
        else:
            print(f"  [{name}] MISSING")
        print()

    # --- cross-source time range ---
    print("--- Time Range Overlap ---")
    ranges = {}

    if arm_path.exists():
        ts = load_pkl(arm_path)["timestamps"]
        if len(ts) > 0:
            ranges["arm"] = (ts[0], ts[-1])

    for name in CAM_NAMES:
        path = ep / f"{name}.pkl"
        if path.exists():
            ts = load_pkl(path)["timestamps"]
            if len(ts) > 0:
                ranges[name] = (ts[0], ts[-1])

    if len(ranges) >= 2:
        t0 = max(r[0] for r in ranges.values())
        t1 = min(r[1] for r in ranges.values())
        overlap = (t1 - t0) / 1e9

        for name, (start, end) in ranges.items():
            dur = (end - start) / 1e9
            print(f"  {name}: {dur:.1f}s")

        if overlap > 0:
            print(f"  overlap: {overlap:.1f}s ✓")
        else:
            print(f"  overlap: NONE ⚠️  sources don't overlap in time!")
    else:
        print("  not enough sources to compute overlap")

    print()


if __name__ == "__main__":
    main()
