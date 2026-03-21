"""
Offline alignment: read per-source .pkl files from an episode directory,
use left_cam0 (~30 Hz) as the reference timeline, align all other sources
by nearest-neighbor timestamp matching, and write a single episode.hdf5.

Usage:
    python build_dataset.py dataset/pick_cup/episode_000
    python build_dataset.py dataset/pick_cup          # batch: all episodes under a task
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np


CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]
REF_CAM = "left_cam0"


# =====================================================================
# I/O helpers
# =====================================================================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_nearest_indices(ref_ts, target_ts):
    """For each timestamp in ref_ts, find the index of the closest timestamp in target_ts."""
    idx = np.searchsorted(target_ts, ref_ts)
    idx = np.clip(idx, 1, len(target_ts) - 1)

    left = target_ts[idx - 1]
    right = target_ts[idx]
    idx -= (ref_ts - left) <= (right - ref_ts)
    return idx


# =====================================================================
# Core
# =====================================================================

def build_episode(episode_dir: Path):
    episode_dir = Path(episode_dir)

    arm_path = episode_dir / "arm.pkl"
    gripper_path = episode_dir / "gripper.pkl"
    cam_paths = {name: episode_dir / f"{name}.pkl" for name in CAM_NAMES}

    missing = []
    if not arm_path.exists():
        missing.append("arm.pkl")
    for name, p in cam_paths.items():
        if not p.exists():
            missing.append(f"{name}.pkl")
    if missing:
        print(f"  SKIP {episode_dir.name}: missing {missing}")
        return False

    arm = load_pkl(arm_path)
    gripper = load_pkl(gripper_path) if gripper_path.exists() else None
    cams = {name: load_pkl(p) for name, p in cam_paths.items()}

    # --- reference timeline: left_cam0 timestamps (~30 Hz) ---
    ref_ts = cams[REF_CAM]["timestamps"]
    n = len(ref_ts)

    if n == 0:
        print(f"  SKIP {episode_dir.name}: reference camera has 0 frames")
        return False

    duration_s = (ref_ts[-1] - ref_ts[0]) / 1e9
    actual_freq = (n - 1) / duration_s if duration_s > 0 else 0
    print(f"  Reference: {REF_CAM}, {n} frames, {duration_s:.1f} s, ~{actual_freq:.1f} Hz")

    # --- align arm (60 Hz → 30 Hz by nearest neighbor) ---
    arm_idx = find_nearest_indices(ref_ts, arm["timestamps"])

    left_q = arm["left_q"][arm_idx]
    right_q = arm["right_q"][arm_idx]
    left_tcp_pose = arm["left_tcp_pose"][arm_idx]
    right_tcp_pose = arm["right_tcp_pose"][arm_idx]

    # --- align gripper ---
    gripper_left = np.zeros(n, dtype=np.float64)
    gripper_right = np.zeros(n, dtype=np.float64)

    if gripper is not None:
        for side, arr in [("left", gripper_left), ("right", gripper_right)]:
            ts_key = f"{side}_timestamps"
            pos_key = f"{side}_pos"
            if len(gripper[ts_key]) > 0:
                idx = find_nearest_indices(ref_ts, gripper[ts_key])
                arr[:] = gripper[pos_key][idx]

    # --- align other cameras to ref camera ---
    cam_indices = {}
    for name in CAM_NAMES:
        if name == REF_CAM:
            cam_indices[name] = np.arange(n)
        else:
            cam_indices[name] = find_nearest_indices(ref_ts, cams[name]["timestamps"])

    # --- report alignment quality ---
    dt_arm = np.abs(ref_ts - arm["timestamps"][arm_idx])
    print(f"    arm: median {np.median(dt_arm)/1e6:.1f} ms, max {np.max(dt_arm)/1e6:.1f} ms")

    for name in CAM_NAMES:
        if name == REF_CAM:
            continue
        dt = np.abs(ref_ts - cams[name]["timestamps"][cam_indices[name]])
        print(f"    {name}: median {np.median(dt)/1e6:.1f} ms, max {np.max(dt)/1e6:.1f} ms")

    if gripper is not None:
        for side in ("left", "right"):
            ts_key = f"{side}_timestamps"
            if len(gripper[ts_key]) > 0:
                idx = find_nearest_indices(ref_ts, gripper[ts_key])
                dt = np.abs(ref_ts - gripper[ts_key][idx])
                print(f"    gripper_{side}: median {np.median(dt)/1e6:.1f} ms, max {np.max(dt)/1e6:.1f} ms")

    # --- write HDF5 ---
    out_path = episode_dir / "episode.hdf5"

    with h5py.File(out_path, "w") as f:
        f.create_dataset("timestamp", data=ref_ts)

        g_left = f.create_group("left_arm")
        g_left.create_dataset("q", data=left_q.astype(np.float32))
        g_left.create_dataset("tcp_pose", data=left_tcp_pose.astype(np.float32))
        g_left.create_dataset("gripper", data=gripper_left.astype(np.float32))

        g_right = f.create_group("right_arm")
        g_right.create_dataset("q", data=right_q.astype(np.float32))
        g_right.create_dataset("tcp_pose", data=right_tcp_pose.astype(np.float32))
        g_right.create_dataset("gripper", data=gripper_right.astype(np.float32))

        vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        cam_group = f.create_group("camera")

        for name in CAM_NAMES:
            ds = cam_group.create_dataset(name, (n,), dtype=vlen_dtype)
            frames = cams[name]["frames"]
            indices = cam_indices[name]

            for i in range(n):
                jpeg_bytes = frames[indices[i]]
                ds[i] = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    print(f"  → {out_path} ({n} frames)")
    return True


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Build aligned HDF5 from per-source .pkl files")
    parser.add_argument("path", type=str, help="Episode directory or task directory (batch mode)")
    args = parser.parse_args()

    target = Path(args.path)

    if (target / "arm.pkl").exists():
        print(f"Building: {target}")
        ok = build_episode(target)
        sys.exit(0 if ok else 1)
    else:
        episodes = sorted(target.glob("episode_*"))
        episodes = [p for p in episodes if p.is_dir() and (p / "arm.pkl").exists()]

        if not episodes:
            print(f"No episodes found in {target}")
            sys.exit(1)

        print(f"Found {len(episodes)} episodes in {target}")
        ok_count = 0
        for ep in episodes:
            print(f"\nBuilding: {ep.name}")
            if build_episode(ep):
                ok_count += 1

        print(f"\nDone: {ok_count}/{len(episodes)} episodes built successfully.")
        sys.exit(0 if ok_count == len(episodes) else 1)


if __name__ == "__main__":
    main()
