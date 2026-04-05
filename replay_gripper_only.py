#!/usr/bin/env python3
"""
Replay only gripper positions from episode.hdf5 via MQTT — no Flexiv arms.

Use this to isolate gripper/MQTT/Orin issues when full replay works for arms
but grippers do not move.

Usage:
    python replay_gripper_only.py dataset/test0328/episode_001
    python replay_gripper_only.py dataset/test0328/episode_001 --freq 30 --speed-scale 0.5
    python replay_gripper_only.py dataset/test0328/episode_001 --dry-run --sample 15
    python replay_gripper_only.py dataset/test0328/episode_001 --mqtt-host 192.168.20.2
    python replay_gripper_only.py dataset/test0328/episode_001 --no-print-stored
    python replay_gripper_only.py dataset/test0328/episode_001 --print-stored-rows 0

Orin: mosquitto + gripper_2 / gripper_4 mqtt_forwarding DAGs should be running.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from replay_demo import (
    DEFAULT_MQTT_HOST,
    DEFAULT_MQTT_PORT,
    compute_dt,
    init_grippers,
    resolve_hdf5_path,
    _gripper_send,
    _print_progress,
    LEFT_GRIPPER_NAME,
    RIGHT_GRIPPER_NAME,
)


def _import_gripper_controller():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "gripper"))
    from gripper_ctrl import GripperController

    return GripperController


def load_gripper_episode(file_path: Path) -> dict:
    print(f"Loading: {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {
            "timestamp": f["timestamp"][:],
            "left_gripper": f["left_arm/gripper"][:],
            "right_gripper": f["right_arm/gripper"][:],
        }
    n = len(data["timestamp"])
    dur = (data["timestamp"][-1] - data["timestamp"][0]) / 1e9 if n >= 2 else 0
    print(f"  {n} frames, {dur:.1f}s (gripper channels only)")
    return data


def print_stored_gripper_state(data: dict, *, sample: int = 16) -> None:
    """Print gripper scalars stored in episode.hdf5 (aligned to timestamp)."""
    ts = data["timestamp"]
    lg = np.asarray(data["left_gripper"], dtype=np.float64)
    rg = np.asarray(data["right_gripper"], dtype=np.float64)
    n = len(ts)
    t0 = int(ts[0]) if n else 0
    ts_rel = (ts.astype(np.float64) - t0) / 1e9

    print("")
    print("Stored robot gripper state (HDF5: left_arm/gripper, right_arm/gripper)")
    print("-" * 72)
    for name, arr in (("left_arm/gripper", lg), ("right_arm/gripper", rg)):
        print(
            f"  {name}:  min={arr.min():.6f}  max={arr.max():.6f}  "
            f"mean={arr.mean():.6f}  std={arr.std():.6f}"
        )
    if n >= 2:
        dt_ms = np.median(np.diff(ts).astype(np.float64)) / 1e6
        print(f"  timestamp:  n={n}  span={ts_rel[-1]:.3f}s  median_dt={dt_ms:.2f} ms")

    k = max(0, min(sample, n))
    if k > 0:
        idx = np.linspace(0, n - 1, k, dtype=int) if k > 1 else np.array([0], dtype=int)
        print(f"  sample ({k} rows):  idx  t_rel(s)  left_gripper  right_gripper")
        for i in idx:
            print(
                f"    {i:5d}  {ts_rel[i]:8.3f}  {lg[i]:12.6f}  {rg[i]:12.6f}"
            )
    print("-" * 72)
    print("")


def dry_run(data: dict, dt: float, sample: int):
    n = len(data["timestamp"])
    ts = data["timestamp"]
    ts_sec = (ts - ts[0]) / 1e9
    k = min(sample, n)
    idx = np.linspace(0, n - 1, k, dtype=int) if k > 1 else np.array([0], dtype=int)

    print("")
    print("=" * 72)
    print(f"DRY RUN — {n} frames, record timeline {ts_sec[-1]:.1f}s")
    print(f"  dt={dt*1000:.1f}ms ({1.0/dt:.1f} Hz effective)")
    print(f"  LEFT  MQTT {LEFT_GRIPPER_NAME}")
    print(f"  RIGHT MQTT {RIGHT_GRIPPER_NAME}")
    print(f"  showing {len(idx)} sampled frames")
    print("=" * 72)

    for i in idx:
        print(
            f"  frame {i:4d}  t={ts_sec[i]:6.2f}s  "
            f"L={float(data['left_gripper'][i]):.4f}  "
            f"R={float(data['right_gripper'][i]):.4f}"
        )

    lg = data["left_gripper"]
    rg = data["right_gripper"]
    print("\n  left_gripper  min/max:", float(lg.min()), float(lg.max()))
    print("  right_gripper min/max:", float(rg.min()), float(rg.max()))
    print("")


def main():
    ap = argparse.ArgumentParser(description="Replay gripper only from episode.hdf5 (MQTT).")
    ap.add_argument("episode", help="Episode directory or episode.hdf5 path")
    ap.add_argument("--freq", type=float, default=None, help="Command rate Hz (default: from data median, else 30)")
    ap.add_argument("--speed-scale", type=float, default=1.0, help=">1 faster wall-clock, <1 slower")
    ap.add_argument("--mqtt-host", type=str, default=DEFAULT_MQTT_HOST)
    ap.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample", type=int, default=12, help="Dry-run: number of sampled rows to print")
    ap.add_argument(
        "--verbose-step",
        type=int,
        default=0,
        help="If >0, print every N-th frame during live replay (debug)",
    )
    ap.add_argument(
        "--print-stored-rows",
        type=int,
        default=16,
        help="After load: print this many evenly-spaced stored gripper rows (0 = stats only)",
    )
    ap.add_argument(
        "--no-print-stored",
        action="store_true",
        help="Do not print stored gripper summary from HDF5",
    )
    args = ap.parse_args()

    hdf5_path = resolve_hdf5_path(args.episode)
    data = load_gripper_episode(hdf5_path)
    if not args.no_print_stored:
        print_stored_gripper_state(data, sample=max(0, args.print_stored_rows))

    if args.freq is not None:
        dt = 1.0 / args.freq / args.speed_scale
    else:
        dt = compute_dt(data["timestamp"], 30.0) / args.speed_scale

    print(f"Replay dt: {dt*1000:.1f} ms ({1.0/dt:.1f} Hz)")

    if args.dry_run:
        dry_run(data, dt, args.sample)
        return

    GripperController = _import_gripper_controller()
    left_g, right_g = init_grippers(
        GripperController,
        args.mqtt_host,
        args.mqtt_port,
    )

    input("\nPress ENTER to start gripper-only replay (no arm motion) ...")

    n = len(data["left_gripper"])
    replay_start = time.time()
    fails = 0
    vstep = args.verbose_step

    try:
        print(f"Replaying {n} gripper frames ...")
        for i in range(n):
            t0 = time.time()

            if vstep > 0 and i % vstep == 0:
                print(
                    f"  [{i}/{n}] L={float(data['left_gripper'][i]):.4f} "
                    f"R={float(data['right_gripper'][i]):.4f}"
                )

            fl, fr = _gripper_send(left_g, right_g, i, data)
            fails += fl + fr

            elapsed = time.time() - t0
            rem = dt - elapsed
            if rem > 0:
                time.sleep(rem)

            _print_progress(i, n, replay_start, dt)

        if fails:
            print(f"WARN: {fails} MQTT publishes failed (not connected or publish error).")
        else:
            print("All gripper MQTT publishes returned OK (rc=0).")
        print("Gripper-only replay finished.")
    finally:
        for label, g in [("left", left_g), ("right", right_g)]:
            try:
                g.stop()
            except Exception as e:
                print(f"Warning: stop {label} gripper: {e}")


if __name__ == "__main__":
    main()
