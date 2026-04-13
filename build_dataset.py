"""
Offline alignment: read per-source .pkl files from an episode directory,
use left_cam0 (~30 Hz) as the reference timeline, align all other sources,
and write a single episode.hdf5.

Gripper values are aligned via linear interpolation (not nearest-neighbor)
for smooth transitions. An optional --gripper_offset_ms shifts gripper
timestamps to compensate for Orin-side feedback transport delay.

Usage:
    python build_dataset.py dataset/pick_cup/episode_000
    python build_dataset.py dataset/pick_cup                          # batch
    python build_dataset.py dataset/pick_cup --gripper_offset_ms -10  # shift gripper earlier by 10ms
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1", "top_cam"]
REF_CAM = "left_cam0"
TARGET_SIZE = (640, 480)
RESIZE_CAMS = {"top_cam"}


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

def build_episode(episode_dir: Path, gripper_offset_ns: int = 0):
    episode_dir = Path(episode_dir)

    arm_path = episode_dir / "arm.pkl"
    gripper_path = episode_dir / "gripper.pkl"
    cam_paths = {name: episode_dir / f"{name}.pkl" for name in CAM_NAMES}

    # top_cam is optional (USB RealSense, not present in old episodes)
    OPTIONAL_CAMS = {"top_cam"}

    missing = []
    if not arm_path.exists():
        missing.append("arm.pkl")
    for name, p in cam_paths.items():
        if not p.exists() and name not in OPTIONAL_CAMS:
            missing.append(f"{name}.pkl")
    if missing:
        print(f"  SKIP {episode_dir.name}: missing {missing}")
        return False

    arm = load_pkl(arm_path)
    gripper = load_pkl(gripper_path) if gripper_path.exists() else None

    available_cams = [name for name in CAM_NAMES if cam_paths[name].exists()]
    cams = {name: load_pkl(cam_paths[name]) for name in available_cams}

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

    # --- align gripper (linear interpolation) ---
    gripper_left = np.zeros(n, dtype=np.float64)
    gripper_right = np.zeros(n, dtype=np.float64)

    if gripper is not None:
        for side, arr in [("left", gripper_left), ("right", gripper_right)]:
            ts_key = f"{side}_timestamps"
            pos_key = f"{side}_pos"
            if len(gripper[ts_key]) > 0:
                g_ts = gripper[ts_key].astype(np.float64) + gripper_offset_ns
                arr[:] = np.interp(ref_ts.astype(np.float64),
                                   g_ts, gripper[pos_key].astype(np.float64))

    # --- align other cameras to ref camera ---
    cam_indices = {}
    for name in available_cams:
        if name == REF_CAM:
            cam_indices[name] = np.arange(n)
        else:
            cam_indices[name] = find_nearest_indices(ref_ts, cams[name]["timestamps"])

    # --- report alignment quality ---
    dt_arm = np.abs(ref_ts - arm["timestamps"][arm_idx])
    print(f"    arm: median {np.median(dt_arm)/1e6:.1f} ms, max {np.max(dt_arm)/1e6:.1f} ms")

    for name in available_cams:
        if name == REF_CAM:
            continue
        dt = np.abs(ref_ts - cams[name]["timestamps"][cam_indices[name]])
        print(f"    {name}: median {np.median(dt)/1e6:.1f} ms, max {np.max(dt)/1e6:.1f} ms")

    if gripper is not None:
        for side in ("left", "right"):
            ts_key = f"{side}_timestamps"
            if len(gripper[ts_key]) > 0:
                g_ts = gripper[ts_key].astype(np.float64) + gripper_offset_ns
                idx = find_nearest_indices(ref_ts, g_ts.astype(np.int64))
                signed = (g_ts[idx] - ref_ts) / 1e6
                dt_abs = np.abs(signed)
                # effective rate (excluding duplicate timestamps)
                g_dt = np.diff(gripper[ts_key])
                n_unique = np.sum(g_dt > 0) + 1
                g_dur = (gripper[ts_key][-1] - gripper[ts_key][0]) / 1e9
                eff_hz = (n_unique - 1) / g_dur if g_dur > 0 and n_unique > 1 else 0
                max_gap_ms = np.max(g_dt[g_dt > 0]) / 1e6 if np.any(g_dt > 0) else 0
                print(f"    gripper_{side}: abs median {np.median(dt_abs):.1f} ms, max {np.max(dt_abs):.1f} ms | "
                      f"signed median {np.median(signed):+.1f} ms | "
                      f"{eff_hz:.0f} Hz eff ({len(gripper[ts_key])} raw), max gap {max_gap_ms:.0f} ms"
                      + (f" | offset {gripper_offset_ns/1e6:+.0f} ms" if gripper_offset_ns != 0 else ""))

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

        for name in available_cams:
            ds = cam_group.create_dataset(name, (n,), dtype=vlen_dtype)
            frames = cams[name]["frames"]
            indices = cam_indices[name]
            need_resize = name in RESIZE_CAMS

            for i in range(n):
                jpeg_bytes = frames[indices[i]]
                if need_resize:
                    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None and (img.shape[1], img.shape[0]) != TARGET_SIZE:
                        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        jpeg_bytes = buf.tobytes()
                ds[i] = np.frombuffer(jpeg_bytes, dtype=np.uint8)

    print(f"  → {out_path} ({n} frames)")
    return True


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Build aligned HDF5 from per-source .pkl files")
    parser.add_argument("path", type=str, help="Episode directory or task directory (batch mode)")
    parser.add_argument("--gripper_offset_ms", type=float, default=0.0,
                        help="Shift gripper timestamps by this many ms before alignment. "
                             "Negative = gripper was measured earlier than its stamp (compensate feedback delay).")
    args = parser.parse_args()

    target = Path(args.path)
    offset_ns = int(args.gripper_offset_ms * 1_000_000)

    if (target / "arm.pkl").exists():
        print(f"Building: {target}")
        ok = build_episode(target, gripper_offset_ns=offset_ns)
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
            if build_episode(ep, gripper_offset_ns=offset_ns):
                ok_count += 1

        print(f"\nDone: {ok_count}/{len(episodes)} episodes built successfully.")
        sys.exit(0 if ok_count == len(episodes) else 1)


if __name__ == "__main__":
    main()
