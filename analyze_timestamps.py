"""
Multi-sensor timestamp alignment analysis.

Three levels of analysis:
  1. PC receive timestamps: pairwise offsets between raw pkl sources
  2. Physical sync: cross-correlation between arm velocity and camera motion
  3. Post-alignment: residual error in the final episode.hdf5

Usage:
    python analyze_timestamps.py dataset/test/episode_010
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

CAM_NAMES = ["left_cam0", "left_cam1", "right_cam0", "right_cam1"]


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_nearest_indices(ref_ts, target_ts):
    idx = np.searchsorted(target_ts, ref_ts)
    idx = np.clip(idx, 1, len(target_ts) - 1)
    left = target_ts[idx - 1]
    right = target_ts[idx]
    idx -= (ref_ts - left) <= (right - ref_ts)
    return idx


# =====================================================================
# Part 1: PC receive timestamp offsets (raw pkl)
# =====================================================================

def analyze_pc_timestamps(ep: Path):
    print("=" * 60)
    print("Part 1: PC Receive Timestamp Offsets (raw pkl)")
    print("=" * 60)

    sources = {}

    arm_path = ep / "arm.pkl"
    if arm_path.exists():
        arm = load_pkl(arm_path)
        sources["arm"] = np.array(arm["timestamps"], dtype=np.int64)

    for name in CAM_NAMES:
        p = ep / f"{name}.pkl"
        if p.exists():
            cam = load_pkl(p)
            sources[name] = np.array(cam["timestamps"], dtype=np.int64)

    gripper_path = ep / "gripper.pkl"
    if gripper_path.exists():
        g = load_pkl(gripper_path)
        for side in ("left", "right"):
            ts = np.array(g[f"{side}_timestamps"], dtype=np.int64)
            if len(ts) > 0:
                sources[f"gripper_{side}"] = ts

    names = list(sources.keys())
    print(f"\n  Sources: {names}")
    for name, ts in sources.items():
        dur = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0
        hz = (len(ts) - 1) / dur if dur > 0 else 0
        print(f"    {name}: {len(ts)} samples, {dur:.1f}s, {hz:.1f} Hz")

    # Pairwise: for each pair, find nearest-neighbor offsets
    ref_name = "left_cam0"
    if ref_name not in sources:
        print("  WARNING: left_cam0 not found, skipping pairwise analysis")
        return

    ref_ts = sources[ref_name]
    print(f"\n  Pairwise offsets (reference: {ref_name}):\n")
    print(f"  {'Source':<20s} {'median':>10s} {'p95':>10s} {'max':>10s} {'mean':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, ts in sources.items():
        if name == ref_name:
            continue
        idx = find_nearest_indices(ref_ts, ts)
        offsets_ms = np.abs(ref_ts - ts[idx]) / 1e6
        print(f"  {name:<20s} {np.median(offsets_ms):>9.1f}ms "
              f"{np.percentile(offsets_ms, 95):>9.1f}ms "
              f"{np.max(offsets_ms):>9.1f}ms "
              f"{np.mean(offsets_ms):>9.1f}ms")

    # Camera-to-camera: for each ref timestamp, check spread across cameras
    cam_sources = {n: sources[n] for n in CAM_NAMES if n in sources}
    if len(cam_sources) >= 2:
        print(f"\n  Camera-to-camera spread (per reference frame):")
        cam_matched_ts = {}
        for name, ts in cam_sources.items():
            idx = find_nearest_indices(ref_ts, ts)
            cam_matched_ts[name] = ts[idx]

        all_cam_ts = np.stack([cam_matched_ts[n] for n in cam_matched_ts], axis=0)
        spread_ms = (all_cam_ts.max(axis=0) - all_cam_ts.min(axis=0)) / 1e6
        print(f"    spread: median {np.median(spread_ms):.1f}ms, "
              f"p95 {np.percentile(spread_ms, 95):.1f}ms, "
              f"max {np.max(spread_ms):.1f}ms")


# =====================================================================
# Part 2: Physical sync via cross-correlation
# =====================================================================

def analyze_physical_sync(ep: Path):
    print("\n" + "=" * 60)
    print("Part 2: Physical Sync Error (cross-correlation)")
    print("=" * 60)

    arm_path = ep / "arm.pkl"
    if not arm_path.exists():
        print("  arm.pkl not found, skipping")
        return

    arm = load_pkl(arm_path)
    arm_ts = np.array(arm["timestamps"], dtype=np.int64)

    # Compute arm angular velocity magnitude (sum of |dq| across joints)
    # Use left_dq and right_dq if available, else finite-difference left_q
    arm_vel = np.zeros(len(arm_ts))
    for side in ("left", "right"):
        dq_key = f"{side}_dq"
        q_key = f"{side}_q"
        if dq_key in arm and len(arm[dq_key]) > 0:
            dq = np.array(arm[dq_key])
            arm_vel += np.sqrt(np.sum(dq ** 2, axis=1))
        elif q_key in arm and len(arm[q_key]) > 1:
            q = np.array(arm[q_key])
            dt = np.diff(arm_ts) / 1e9
            dq = np.diff(q, axis=0) / dt[:, None]
            dq = np.vstack([dq, dq[-1:]])
            arm_vel += np.sqrt(np.sum(dq ** 2, axis=1))

    for cam_name in CAM_NAMES:
        cam_path = ep / f"{cam_name}.pkl"
        if not cam_path.exists():
            continue

        cam = load_pkl(cam_path)
        cam_ts = np.array(cam["timestamps"], dtype=np.int64)
        frames = cam["frames"]
        n_cam = len(frames)

        if n_cam < 10:
            print(f"  [{cam_name}] too few frames ({n_cam}), skipping")
            continue

        # Compute frame-to-frame pixel difference magnitude
        cam_motion = np.zeros(n_cam)
        prev_gray = None
        for i in range(n_cam):
            buf = np.frombuffer(frames[i], dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_small = cv2.resize(img, (160, 120))
            if prev_gray is not None:
                cam_motion[i] = np.mean(np.abs(img_small.astype(float) - prev_gray.astype(float)))
            prev_gray = img_small

        # Resample both signals to a common uniform timeline (100 Hz)
        t0 = max(arm_ts[0], cam_ts[0])
        t1 = min(arm_ts[-1], cam_ts[-1])
        if t1 <= t0:
            print(f"  [{cam_name}] no time overlap with arm, skipping")
            continue

        uniform_ts = np.arange(t0, t1, int(1e9 / 100))  # 100 Hz
        n_uniform = len(uniform_ts)
        if n_uniform < 20:
            print(f"  [{cam_name}] overlap too short, skipping")
            continue

        arm_idx = find_nearest_indices(uniform_ts, arm_ts)
        arm_signal = arm_vel[arm_idx]

        cam_idx = find_nearest_indices(uniform_ts, cam_ts)
        cam_signal = cam_motion[cam_idx]

        # Normalize
        arm_signal = arm_signal - np.mean(arm_signal)
        cam_signal = cam_signal - np.mean(cam_signal)

        arm_std = np.std(arm_signal)
        cam_std = np.std(cam_signal)
        if arm_std < 1e-8 or cam_std < 1e-8:
            print(f"  [{cam_name}] signal too flat for cross-correlation, skipping")
            continue

        arm_signal /= arm_std
        cam_signal /= cam_std

        # Cross-correlation over ±500ms window
        max_lag_samples = 50  # ±500ms at 100Hz
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        corr = np.zeros(len(lags))

        for j, lag in enumerate(lags):
            if lag >= 0:
                a = arm_signal[lag:]
                c = cam_signal[:n_uniform - lag]
            else:
                a = arm_signal[:n_uniform + lag]
                c = cam_signal[-lag:]
            if len(a) > 0:
                corr[j] = np.mean(a * c)

        best_idx = np.argmax(corr)
        best_lag = lags[best_idx]
        best_lag_ms = best_lag * 10  # 100Hz → 10ms per sample
        peak_corr = corr[best_idx]

        print(f"  [{cam_name}] peak correlation: {peak_corr:.3f} at lag {best_lag_ms:+d}ms "
              f"(camera {'leads' if best_lag_ms < 0 else 'lags'} arm by {abs(best_lag_ms)}ms)")


# =====================================================================
# Part 3: Post-alignment residual error (episode.hdf5)
# =====================================================================

def analyze_aligned_episode(ep: Path):
    print("\n" + "=" * 60)
    print("Part 3: Post-Alignment Residual Error (episode.hdf5)")
    print("=" * 60)

    hdf5_path = ep / "episode.hdf5"
    if not hdf5_path.exists():
        print("  episode.hdf5 not found, skipping")
        return

    # Load raw pkl timestamps for comparison
    raw_ts = {}
    arm_path = ep / "arm.pkl"
    if arm_path.exists():
        raw_ts["arm"] = np.array(load_pkl(arm_path)["timestamps"], dtype=np.int64)

    for name in CAM_NAMES:
        p = ep / f"{name}.pkl"
        if p.exists():
            raw_ts[name] = np.array(load_pkl(p)["timestamps"], dtype=np.int64)

    gripper_path = ep / "gripper.pkl"
    if gripper_path.exists():
        g = load_pkl(gripper_path)
        for side in ("left", "right"):
            ts = np.array(g[f"{side}_timestamps"], dtype=np.int64)
            if len(ts) > 0:
                raw_ts[f"gripper_{side}"] = ts

    with h5py.File(hdf5_path, "r") as f:
        ref_ts = f["timestamp"][:]
        n = len(ref_ts)

        print(f"\n  Episode: {n} frames, {(ref_ts[-1]-ref_ts[0])/1e9:.1f}s")

        # Alignment residuals: how far is each aligned sample from its
        # nearest raw source timestamp?
        print(f"\n  Nearest-neighbor alignment residuals:\n")
        print(f"  {'Source':<20s} {'median':>10s} {'p95':>10s} {'max':>10s} {'mean':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for name, ts in raw_ts.items():
            idx = find_nearest_indices(ref_ts, ts)
            residuals_ms = np.abs(ref_ts - ts[idx]) / 1e6
            print(f"  {name:<20s} {np.median(residuals_ms):>9.1f}ms "
                  f"{np.percentile(residuals_ms, 95):>9.1f}ms "
                  f"{np.max(residuals_ms):>9.1f}ms "
                  f"{np.mean(residuals_ms):>9.1f}ms")

        # Timeline regularity
        if n >= 2:
            dt = np.diff(ref_ts) / 1e6
            print(f"\n  Timeline regularity (frame-to-frame dt):")
            print(f"    median {np.median(dt):.1f}ms, p95 {np.percentile(dt, 95):.1f}ms, "
                  f"max {np.max(dt):.1f}ms, std {np.std(dt):.1f}ms")

            target_dt = 1000.0 / 30  # 33.3ms for 30Hz target
            jitter = dt - target_dt
            print(f"\n  Jitter relative to 30Hz target ({target_dt:.1f}ms):")
            print(f"    median {np.median(jitter):+.1f}ms, p95 {np.percentile(np.abs(jitter), 95):.1f}ms, "
                  f"max {np.max(np.abs(jitter)):.1f}ms")

        # Camera content: do different cameras show temporally consistent content?
        # Check frame-to-frame motion correlation across cameras
        print(f"\n  Cross-camera frame motion consistency:")
        cam_group = f["camera"]
        cam_motions = {}

        for name in CAM_NAMES:
            if name not in cam_group:
                continue
            ds = cam_group[name]
            motion = np.zeros(n)
            prev_gray = None
            sample_every = max(1, n // 200)
            for i in range(0, n, sample_every):
                buf = ds[i]
                if len(buf) == 0:
                    continue
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_small = cv2.resize(img, (80, 60))
                if prev_gray is not None:
                    motion[i] = np.mean(np.abs(img_small.astype(float) - prev_gray.astype(float)))
                prev_gray = img_small
            cam_motions[name] = motion

        if len(cam_motions) >= 2:
            names_list = list(cam_motions.keys())
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    a = cam_motions[names_list[i]]
                    b = cam_motions[names_list[j]]
                    mask = (a > 0) & (b > 0)
                    if np.sum(mask) > 10:
                        r = np.corrcoef(a[mask], b[mask])[0, 1]
                        print(f"    {names_list[i]} vs {names_list[j]}: "
                              f"motion correlation r={r:.3f}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-sensor timestamp alignment analysis")
    parser.add_argument("episode_dir", type=str, help="Path to episode directory")
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    if not ep.is_dir():
        print(f"Not a directory: {ep}")
        sys.exit(1)

    analyze_pc_timestamps(ep)
    analyze_physical_sync(ep)
    analyze_aligned_episode(ep)
    print()


if __name__ == "__main__":
    main()
