"""
Stage 1 诊断：验证 Orin → PC 的 WebSocket + H.265 链路。

单独运行，不需要机器人。启动后等待 Orin 连接，分两层统计：
  1. 数据到达率（只做 protobuf parse，不解码）— 看网络/传输是否健康
  2. 解码输出率（H.265 decode）— 看解码能否跟上

用法:
    python diagnose_receiver.py              # 运行 10 秒
    python diagnose_receiver.py --duration 30
    python diagnose_receiver.py --save-frames
"""

import argparse
import asyncio
import os
import sys
import time

import av
import cv2
import numpy as np
import websockets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_PATH = os.path.join(CURRENT_DIR, "proto_py")
if PROTO_PATH not in sys.path:
    sys.path.insert(0, PROTO_PATH)

from modules.tools.cyber2pc.proto import Cyber2pcBatch_pb2

HOST = "0.0.0.0"
PORT = 9002

CAM_CHANNELS = {
    "/sensor_camera_senyun/bl_fisheye/compressed": "left_cam0",
    "/sensor_camera_senyun/cl_fisheye/compressed": "left_cam1",
    "/sensor_camera_senyun/br_fisheye/compressed": "right_cam0",
    "/sensor_camera_senyun/cr_fisheye/compressed": "right_cam1",
}

TARGET_SIZE = (640, 480)


class ChannelStats:
    def __init__(self, name):
        self.name = name
        self.decoder = av.CodecContext.create("hevc", "r")

        # layer 1: raw data arrival
        self.data_arrivals = []

        # layer 2: decoded frames
        self.frames_decoded = 0
        self.decode_times_ms = []
        self.frame_timestamps = []
        self.first_frame_time = None
        self.start_time = time.time()

    def feed(self, data):
        now = time.time()
        self.data_arrivals.append(now)

        t0 = time.monotonic()
        try:
            packets = self.decoder.parse(data)
            frames = []
            for pkt in packets:
                frames.extend(self.decoder.decode(pkt))

            if not frames:
                return None

            elapsed_ms = (time.monotonic() - t0) * 1000

            self.frames_decoded += 1
            self.decode_times_ms.append(elapsed_ms)
            self.frame_timestamps.append(now)

            if self.first_frame_time is None:
                self.first_frame_time = now
                warmup = now - self.start_time
                print(f"  [{self.name}] first frame after {warmup:.2f}s warmup")

            return frames[-1].to_ndarray(format="bgr24")

        except Exception as e:
            print(f"  [{self.name}] decode error: {e}")
            return None

    def report(self):
        elapsed = time.time() - self.start_time

        lines = [f"  [{self.name}]"]

        # --- layer 1: data arrival ---
        n_arrived = len(self.data_arrivals)
        arrival_fps = n_arrived / elapsed if elapsed > 0 else 0
        lines.append(f"    data chunks arrived: {n_arrived} ({arrival_fps:.1f}/s)")

        if len(self.data_arrivals) >= 2:
            ifi = np.diff(self.data_arrivals) * 1000
            lines.append(f"    arrival interval: median {np.median(ifi):.1f}ms, p95 {np.percentile(ifi, 95):.1f}ms, max {np.max(ifi):.1f}ms")
            arrival_gaps = int(np.sum(ifi > 100))
            if arrival_gaps > 0:
                lines.append(f"    arrival gaps >100ms: {arrival_gaps}x  ← network/transport issue")

        # --- layer 2: decoded frames ---
        avg_fps = self.frames_decoded / elapsed if elapsed > 0 else 0
        lines.append(f"    frames decoded: {self.frames_decoded} ({avg_fps:.1f} fps)")

        if self.first_frame_time:
            lines.append(f"    warmup: {self.first_frame_time - self.start_time:.2f}s")

        if self.decode_times_ms:
            dt = np.array(self.decode_times_ms)
            lines.append(f"    decode time: median {np.median(dt):.1f}ms, p95 {np.percentile(dt, 95):.1f}ms, max {np.max(dt):.1f}ms")

        if len(self.frame_timestamps) >= 2:
            ifi = np.diff(self.frame_timestamps) * 1000
            lines.append(f"    decode interval: median {np.median(ifi):.1f}ms, p95 {np.percentile(ifi, 95):.1f}ms, max {np.max(ifi):.1f}ms")
            gaps = int(np.sum(ifi > 100))
            if gaps > 0:
                lines.append(f"    decode gaps >100ms: {gaps}x")

        return "\n".join(lines)


async def run_diagnostic(duration, save_frames):
    channels = {}
    batch_count = 0
    batch_timestamps = []
    t_start = time.time()
    saved_count = {name: 0 for name in CAM_CHANNELS.values()}

    async def handler(websocket):
        nonlocal batch_count
        print(f"Client connected from {websocket.remote_address}")

        last_report = time.time()

        async for data in websocket:
            if time.time() - t_start > duration:
                break

            if not isinstance(data, (bytes, bytearray)):
                continue

            batch = Cyber2pcBatch_pb2.Cyber2pcBatch()
            try:
                batch.ParseFromString(data)
            except Exception:
                continue

            batch_count += 1
            batch_timestamps.append(time.time())

            for item in batch.camera_image_data:
                ch = item.channel_name
                if ch not in CAM_CHANNELS:
                    continue

                name = CAM_CHANNELS[ch]
                if name not in channels:
                    channels[name] = ChannelStats(name)

                img = channels[name].feed(item.image.data)

                if save_frames and img is not None and saved_count[name] < 3:
                    img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                    path = f"/tmp/diag_{name}_{saved_count[name]}.jpg"
                    cv2.imwrite(path, img_resized)
                    saved_count[name] += 1

            now = time.time()
            if now - last_report > 2.0:
                elapsed = now - t_start
                print(f"\n--- {elapsed:.0f}s / {duration}s, {batch_count} batches ---")
                for name in CAM_CHANNELS.values():
                    if name in channels:
                        print(channels[name].report())
                last_report = now

        print("\n" + "=" * 60)
        print(f"FINAL REPORT ({duration}s)")
        print("=" * 60)

        # batch-level stats
        if len(batch_timestamps) >= 2:
            bi = np.diff(batch_timestamps) * 1000
            print(f"\n  WebSocket batches: {batch_count} ({batch_count/duration:.0f}/s)")
            print(f"    batch interval: median {np.median(bi):.1f}ms, p95 {np.percentile(bi, 95):.1f}ms, max {np.max(bi):.1f}ms")

        if not channels:
            print("\n  No camera data received at all!")
            print("  Check: is the Orin sending to this PC's IP on port 9002?")
        else:
            for name in CAM_CHANNELS.values():
                if name in channels:
                    print(channels[name].report())
                else:
                    print(f"  [{name}] NO DATA RECEIVED")

        if save_frames:
            print(f"\nSample frames saved to /tmp/diag_*.jpg")

    server = await websockets.serve(handler, HOST, PORT, max_size=None, ping_interval=None)
    print(f"Listening on ws://{HOST}:{PORT} for {duration}s ...")
    print("Waiting for Orin to connect ...\n")

    await asyncio.sleep(duration + 5)
    server.close()


def main():
    parser = argparse.ArgumentParser(description="Stage 1: diagnose Orin → PC WebSocket + H.265 link")
    parser.add_argument("--duration", type=float, default=10, help="Test duration in seconds (default: 10)")
    parser.add_argument("--save-frames", action="store_true", help="Save first few decoded frames to /tmp/")
    args = parser.parse_args()

    asyncio.run(run_diagnostic(args.duration, args.save_frames))


if __name__ == "__main__":
    main()
