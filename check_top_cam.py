"""
Quick visualization of top_cam.pkl as MP4.

Usage:
    python check_top_cam.py dataset/0412_place_tube_100/episode_000
    python check_top_cam.py dataset/0412_place_tube_100/episode_000 --output /tmp/top.mp4
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize top_cam.pkl as MP4")
    parser.add_argument("episode_dir", type=str, help="Episode directory containing top_cam.pkl")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: <episode_dir>/top_cam.mp4)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ep = Path(args.episode_dir)
    pkl_path = ep / "top_cam.pkl"
    if not pkl_path.exists():
        print(f"top_cam.pkl not found in {ep}")
        sys.exit(1)

    print(f"Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    timestamps = data["timestamps"]
    frames = data["frames"]
    n = len(frames)

    if n == 0:
        print("No frames in top_cam.pkl")
        sys.exit(1)

    duration = (timestamps[-1] - timestamps[0]) / 1e9
    actual_hz = (n - 1) / duration if duration > 0 and n > 1 else 0
    print(f"  {n} frames, {duration:.1f}s, ~{actual_hz:.1f} Hz")

    first = cv2.imdecode(np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    print(f"  resolution: {w}x{h}")

    out_path = args.output or str(ep / "top_cam.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    for i in range(n):
        img = cv2.imdecode(np.frombuffer(frames[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)

        t_sec = (timestamps[i] - timestamps[0]) / 1e9
        cv2.putText(img, f"#{i} t={t_sec:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(img)

        if i % 200 == 0:
            print(f"  {i}/{n} ({100 * i / n:.0f}%)")

    writer.release()
    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
