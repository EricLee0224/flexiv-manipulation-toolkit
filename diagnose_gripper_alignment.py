"""
Diagnose gripper-camera timestamp alignment for a raw episode directory.

Reads the raw .pkl files (before build_dataset), plots the gripper signal
alongside camera frame timestamps, and shows the effect of different
--gripper_offset_ms values to help pick the right compensation.

Outputs a PNG with:
  - Row 1: raw gripper signal (100 Hz) with camera frame ticks
  - Row 2: zoomed view of the fastest gripper transition region
  - Row 3: aligned gripper values at camera timestamps for different offsets

Usage:
    python diagnose_gripper_alignment.py dataset/pick_cup/episode_000
    python diagnose_gripper_alignment.py dataset/pick_cup/episode_000 --offsets 0 -10 -20 -30
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Diagnose gripper-camera alignment")
    parser.add_argument("episode_dir", type=str)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0, -5, -10, -15, -20],
                        help="Offset values (ms) to compare (default: 0 -5 -10 -15 -20)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--side", type=str, choices=["left", "right"], default="left")
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    cam_path = ep / "left_cam0.pkl"
    gripper_path = ep / "gripper.pkl"

    for p in [cam_path, gripper_path]:
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    cam = load_pkl(cam_path)
    gripper = load_pkl(gripper_path)

    side = args.side
    cam_ts = cam["timestamps"].astype(np.float64)
    g_ts = gripper[f"{side}_timestamps"].astype(np.float64)
    g_pos = gripper[f"{side}_pos"].astype(np.float64)

    t0 = cam_ts[0]
    cam_sec = (cam_ts - t0) / 1e9
    g_sec = (g_ts - t0) / 1e9

    n_cam = len(cam_ts)
    n_grip = len(g_ts)
    g_dt = np.diff(g_ts)
    n_unique = np.sum(g_dt > 0) + 1
    g_dur = (g_ts[-1] - g_ts[0]) / 1e9
    eff_hz = (n_unique - 1) / g_dur if g_dur > 0 and n_unique > 1 else 0

    print(f"Camera:  {n_cam} frames, {cam_sec[-1]:.1f}s")
    print(f"Gripper ({side}): {n_grip} samples ({n_unique} unique ts), {g_dur:.1f}s, {eff_hz:.0f} Hz eff")

    # find the fastest transition region (largest |dg/dt| over a 0.5s window)
    dg = np.abs(np.diff(g_pos))
    window = max(1, int(eff_hz * 0.5))
    if len(dg) > window:
        cumsum = np.cumsum(dg)
        windowed = cumsum[window:] - cumsum[:-window]
        peak_end = np.argmax(windowed) + window
        peak_start = peak_end - window
    else:
        peak_start, peak_end = 0, len(g_pos) - 1

    zoom_margin = int(eff_hz * 0.3)
    z_start = max(0, peak_start - zoom_margin)
    z_end = min(len(g_pos), peak_end + zoom_margin)
    zoom_t_lo = g_sec[z_start]
    zoom_t_hi = g_sec[min(z_end, len(g_sec) - 1)]

    # interpolate at camera timestamps for each offset
    interp_results = {}
    for offset_ms in args.offsets:
        offset_ns = offset_ms * 1e6
        shifted = g_ts + offset_ns
        vals = np.interp(cam_ts, shifted, g_pos)
        interp_results[offset_ms] = vals

    # --- plot ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    fig.suptitle(f"Gripper Alignment Diagnosis — {ep.name} ({side})", fontsize=13, fontweight="bold")

    # Row 1: full timeline
    ax = axes[0]
    ax.plot(g_sec, g_pos, color="#3498db", lw=0.6, alpha=0.8, label=f"gripper raw ({eff_hz:.0f} Hz)")
    for i in range(n_cam):
        ax.axvline(cam_sec[i], color="#e74c3c", alpha=0.08, lw=0.3)
    ax.axvspan(zoom_t_lo, zoom_t_hi, alpha=0.12, color="orange", label="zoom region")
    ax.set_ylabel("Gripper pos")
    ax.set_xlabel("Time (s)")
    ax.set_title("Full Timeline (red ticks = camera frames)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Row 2: zoomed transition
    ax = axes[1]
    z_mask = (g_sec >= zoom_t_lo) & (g_sec <= zoom_t_hi)
    ax.plot(g_sec[z_mask], g_pos[z_mask], "o-", color="#3498db", markersize=2, lw=1.0, label="gripper raw")
    cam_zoom_mask = (cam_sec >= zoom_t_lo) & (cam_sec <= zoom_t_hi)
    cam_zoom_sec = cam_sec[cam_zoom_mask]
    for t in cam_zoom_sec:
        ax.axvline(t, color="#e74c3c", alpha=0.3, lw=1)
    # show nearest-neighbor vs interpolated
    nn_vals = g_pos[np.searchsorted(g_sec, cam_zoom_sec).clip(0, len(g_pos) - 1)]
    interp0 = interp_results[0][cam_zoom_mask] if 0 in interp_results else None
    ax.plot(cam_zoom_sec, nn_vals, "s", color="#e74c3c", markersize=5, alpha=0.7, label="nearest-neighbor")
    if interp0 is not None:
        ax.plot(cam_zoom_sec, interp0, "^", color="#2ecc71", markersize=5, alpha=0.7, label="interpolated (0ms)")
    ax.set_ylabel("Gripper pos")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Zoom: Fastest Transition ({zoom_t_lo:.2f}–{zoom_t_hi:.2f}s)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Row 3: compare offsets at camera timestamps
    ax = axes[2]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(args.offsets)))
    for (offset_ms, vals), c in zip(interp_results.items(), colors):
        label = f"{offset_ms:+.0f} ms" if offset_ms != 0 else "0 ms (no offset)"
        ax.plot(cam_sec, vals, lw=0.8, color=c, alpha=0.9, label=label)
    ax.set_ylabel("Gripper pos")
    ax.set_xlabel("Time (s)")
    ax.set_title("Interpolated Gripper at Camera Timestamps — Offset Comparison")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = args.output or str(ep / f"gripper_alignment_{side}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
