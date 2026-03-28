import asyncio
import time
import cv2
import numpy as np
import websockets
import av
import sys
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# =========================
# Path setup
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_PATH = os.path.join(CURRENT_DIR, "proto_py")

if PROTO_PATH not in sys.path:
    sys.path.insert(0, PROTO_PATH)

from modules.tools.cyber2pc.proto import Cyber2pcBatch_pb2


HOST = "0.0.0.0"
PORT = 9003

# =========================
# Channels
# =========================
CAM_CHANNELS = {
    "/sensor_camera_senyun/bl_fisheye/compressed": "left_cam0",
    "/sensor_camera_senyun/cl_fisheye/compressed": "left_cam1",
    "/sensor_camera_senyun/br_fisheye/compressed": "right_cam0",
    "/sensor_camera_senyun/cr_fisheye/compressed": "right_cam1",
}

GRIPPER_CHANNELS = {
    "/tars/motor/gripper2_feedback": "right_gripper",
    "/tars/motor/gripper4_feedback": "left_gripper",
}

# =========================
# Shared buffer (for live preview / monitoring)
# =========================
shared_buffer = {
    "cam": {
        "left_cam0": None,
        "left_cam1": None,
        "right_cam0": None,
        "right_cam1": None,
        "ts": None,
    },
    "gripper": {
        "left": deque(maxlen=200),
        "right": deque(maxlen=200),
    }
}

# =========================
# Recording buffers — each source stores its own timestamped stream
# =========================
_recording = False
_last_frame_added_ts = 0

_cam_buffers = {
    "left_cam0": [],
    "left_cam1": [],
    "right_cam0": [],
    "right_cam1": [],
}

_gripper_buffers = {
    "left": [],
    "right": [],
}


def start_recording():
    global _recording, _cam_buffers, _gripper_buffers, _last_seq
    _cam_buffers = {k: [] for k in CAM_CHANNELS.values()}
    _gripper_buffers = {"left": [], "right": []}
    _last_seq.clear()
    _recording = True


def stop_recording():
    global _recording
    _recording = False


def flush_and_stop_recording(timeout=3.0, grace=2.0):
    """Keep recording until the decode pipeline has caught up, then stop.

    Phase 1: wait until a camera frame with recv_ts >= cutoff has been
    written to the buffer, proving all PC-side in-flight decodes are done.
    Phase 2: grace period to let Orin-side encoding + network latency
    deliver the final frames that were captured before the stop command.
    """
    global _recording
    cutoff = time.time_ns()
    counts_before = {k: len(v) for k, v in _cam_buffers.items()}
    start = time.time()

    while time.time() - start < timeout:
        if _last_frame_added_ts >= cutoff:
            break
        time.sleep(0.05)

    phase1 = time.time() - start
    print(f"[FLUSH] PC pipeline caught up in {phase1:.2f}s, "
          f"waiting {grace}s for upstream latency ...")
    time.sleep(grace)

    elapsed = time.time() - start
    counts_after = {k: len(v) for k, v in _cam_buffers.items()}
    extra = {k: counts_after[k] - counts_before.get(k, 0) for k in counts_after}
    print(f"[FLUSH] done in {elapsed:.2f}s, extra frames captured: {extra}")

    _recording = False


def get_recorded_cam():
    return dict(_cam_buffers)


def get_recorded_gripper():
    return dict(_gripper_buffers)


# =========================
# Config
# =========================
VERBOSE = False
TARGET_SIZE = (640, 480)

# =========================
# Per-channel decoder (each channel gets its own codec + thread)
# =========================
_decode_stats = {}
_decoders = {}
_decode_pool = ThreadPoolExecutor(max_workers=4)
_last_seq = {}  # per-channel seq for dedup
_orin_pc_offset = None  # estimated once: pc_ns - orin_ns


def get_decode_stats():
    return dict(_decode_stats)


def _calibrate_clock(channel, orin_ts, recv_ts):
    """Estimate Orin-to-PC clock offset from the first decoded frame."""
    global _orin_pc_offset
    _orin_pc_offset = recv_ts - orin_ts
    offset_ms = _orin_pc_offset / 1e6
    short = channel.split("/")[-2] if "/" in channel else channel
    print(f"[CLOCK] Orin→PC offset calibrated via {short}: {offset_ms:+.1f} ms")


def _decode_one(channel, data, recording):
    """Decode + resize + optional JPEG encode. Runs in a thread pool worker.

    Returns (img, jpeg_bytes) where jpeg_bytes is None if not recording.
    """
    if channel not in _decoders:
        _decoders[channel] = av.CodecContext.create("hevc", "r")

    if channel not in _decode_stats:
        _decode_stats[channel] = {
            "frames_out": 0,
            "errors": 0,
            "first_frame_at": None,
            "start_time": time.time(),
            "seq_dup": 0,
            "recv_total": 0,
            "recv_bytes_total": 0,
            "packets_total": 0,
            "empty_decode": 0,
        }

    stats = _decode_stats[channel]
    codec = _decoders[channel]

    try:
        packets = codec.parse(data)
        frames = []

        for packet in packets:
            frames.extend(codec.decode(packet))

        stats["recv_total"] += 1
        stats["recv_bytes_total"] += len(data)
        stats["packets_total"] += len(packets)
        if not frames:
            stats["empty_decode"] += 1
            return None, None

        img = frames[-1].to_ndarray(format="bgr24")
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        jpeg_bytes = None
        if recording:
            _, buf = cv2.imencode(".jpg", img)
            jpeg_bytes = buf.tobytes()

        stats["frames_out"] += 1
        if stats["first_frame_at"] is None:
            warmup = time.time() - stats["start_time"]
            stats["first_frame_at"] = time.time()
            short = channel.split("/")[-2] if "/" in channel else channel
            print(f"[DECODE] {short}: first frame after {warmup:.1f}s warmup")

        return img, jpeg_bytes

    except Exception as e:
        stats["errors"] += 1
        if VERBOSE:
            print(f"❌ decode error [{channel}]:", e)
        return None, None


async def handler(websocket):

    global _last_frame_added_ts
    print("✅ Client connected")

    loop = asyncio.get_event_loop()

    async for data in websocket:

        if not isinstance(data, (bytes, bytearray)):
            continue

        batch = Cyber2pcBatch_pb2.Cyber2pcBatch()

        try:
            batch.ParseFromString(data)
        except Exception as e:
            if VERBOSE:
                print("❌ Parse failed:", e)
            continue

        recv_ts = time.time_ns()

        # =========================
        # CAMERA — seq-based dedup + parallel decode
        # =========================
        cam_items = []
        for item in batch.camera_image_data:
            ch = item.channel_name
            if ch not in CAM_CHANNELS:
                continue

            hdr = item.image.header
            seq = hdr.seq
            prev_seq = _last_seq.get(ch)
            if prev_seq is not None and seq == prev_seq:
                if ch in _decode_stats:
                    _decode_stats[ch]["seq_dup"] += 1
                continue
            _last_seq[ch] = seq

            orin_ts = int(hdr.stamp.sec) * 1_000_000_000 + int(hdr.stamp.nsec)
            cam_items.append((ch, CAM_CHANNELS[ch], item.image.data, orin_ts))

        if cam_items:
            rec = _recording
            try:
                futures = [
                    loop.run_in_executor(_decode_pool, _decode_one, ch, img_data, rec)
                    for ch, name, img_data, orin_ts in cam_items
                ]
                results = await asyncio.gather(*futures)
            except RuntimeError:
                continue

            cam_updated = False
            for (ch, name, _, raw_orin_ts), (img, jpeg_bytes) in zip(cam_items, results):
                if img is not None:
                    if _orin_pc_offset is None:
                        _calibrate_clock(ch, raw_orin_ts, recv_ts)
                    aligned_ts = raw_orin_ts + _orin_pc_offset
                    shared_buffer["cam"][name] = img
                    cam_updated = True

                    if rec and jpeg_bytes is not None:
                        _cam_buffers[name].append({"ts": aligned_ts, "data": jpeg_bytes})
                        _last_frame_added_ts = recv_ts

            if cam_updated:
                shared_buffer["cam"]["ts"] = recv_ts

        # =========================
        # GRIPPER
        # =========================
        for item in batch.body_feedback_data:

            ch = item.channel_name

            if ch not in GRIPPER_CHANNELS:
                continue

            name = GRIPPER_CHANNELS[ch]

            for joint in item.feedback.joints:

                if not joint.measurements:
                    continue

                m = joint.measurements[0]
                pos = getattr(m, "position", None)

                if pos is None:
                    continue

                ts = time.time_ns()

                side = "left" if "left" in name else "right"

                shared_buffer["gripper"][side].append({
                    "ts": ts,
                    "pos": pos
                })

                if _recording:
                    _gripper_buffers[side].append({"ts": ts, "pos": pos})


async def main():

    server = await websockets.serve(
        handler,
        HOST,
        PORT,
        max_size=None,
        ping_interval=None,
    )

    print(f"🚀 Listening on ws://{HOST}:{PORT}")

    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
