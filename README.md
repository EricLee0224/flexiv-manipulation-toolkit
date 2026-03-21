# Flexiv Dual-Arm Data Collection

Kinesthetic demonstration data collection system for dual Flexiv Rizon4s arms with multi-camera observation from an NVIDIA Orin edge computer.

The system records synchronized data from **7 sources**: left/right arm joint states (60 Hz), left/right gripper positions (~30 Hz), and 4 fisheye camera streams (30 Hz H.265 via WebSocket). All sources are independently timestamped and offline-aligned into a single HDF5 episode file.

## Architecture

```
┌────────────────────────────────┐       WebSocket (protobuf)       ┌──────────────────────────────┐
│         NVIDIA Orin            │ ────────────────────────────────▶ │          PC (this repo)       │
│                                │        H.265 + gripper           │                               │
│  4× fisheye cameras           │                                   │  receiver_gripper_cam.py      │
│  2× gripper feedback          │                                   │    ├─ H.265 decode (PyAV)     │
│  Cyber2pcBatch protobuf       │                                   │    ├─ JPEG encode + buffer    │
│  batch_cache_ architecture    │                                   │    └─ gripper state buffer    │
└────────────────────────────────┘                                   │                               │
                                                                     │  recorder.py (ArmRecorder)    │
┌────────────────────────────────┐       Flexiv RDK (ethernet)      │    └─ arm state @ 60 Hz      │
│   Flexiv Rizon4s × 2          │ ◀────────────────────────────────▶│                               │
│   Left:  Rizon4s-063201       │       joint/TCP state + control   │  data_collection.py           │
│   Right: Rizon4s-063236       │                                   │    └─ keyboard UI + save      │
└────────────────────────────────┘                                   └──────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
conda activate flexiv_env

# Core dependencies
pip install numpy h5py opencv-python websockets av protobuf

# Visualization & analysis (optional)
pip install matplotlib scipy
```

The Flexiv RDK Python bindings (`flexivrdk`) must be installed following Flexiv's documentation. The MQTT-based gripper controller requires `paho-mqtt`.

### 1. Collect an Episode

```bash
python data_collection.py
```

Interactive keyboard commands:

| Key | Action |
|-----|--------|
| `e` | Enable teach mode (both arms become compliant) |
| `s` | Disable teach mode |
| `r` | Start recording all sources |
| `t` | Stop recording, flush camera pipeline, save episode |
| `q` | Quit |

Episodes are saved to `dataset/<task_name>/episode_XXX/` as independent `.pkl` files per source.

### 2. Build Aligned Dataset

```bash
# Single episode
python build_dataset.py dataset/<task>/episode_000

# All episodes under a task
python build_dataset.py dataset/<task>
```

Uses `left_cam0` as the 30 Hz reference timeline. All other sources are aligned via nearest-neighbor timestamp matching. Outputs `episode.hdf5`.

### 3. Visualize

```bash
python visualize_episode.py dataset/<task>/episode_000
```

Generates a 1920×720 real-time video with synchronized:
- 2×2 camera grid (left panel)
- 3D end-effector trajectory with orientation frames (right-top)
- Euler angle curves + gripper state with time cursor (right-bottom)

### 4. Replay a Demonstration

```bash
# Joint replay at half speed (safe first test)
python replay_demo.py --mode joint --speed-scale 0.5

# Cartesian replay
python replay_demo.py --mode cartesian --file dataset/<task>/episode_000/episode.hdf5
```

## Project Structure

```
flexiv_data_collection/
│
├── data_collection.py          # Main collection script (keyboard UI)
├── config.py                   # Robot serial numbers, arm IDs, frequencies
├── recorder.py                 # ArmRecorder: threaded arm state sampling
├── receiver_gripper_cam.py     # WebSocket server: H.265 decode + gripper rx
├── ee/
│   └── cyber2pc_observer.py    # Thread wrapper for receiver_gripper_cam
├── robot/
│   ├── flexiv.py               # Flexiv RDK v1.8 wrapper
│   └── flexiv_arm.py           # Teach-mode arm with keep-alive loop
├── gripper/
│   ├── gripper_ctrl.py         # MQTT gripper controller (TsWbd protobuf)
│   └── main.py                 # Gripper demo
│
├── build_dataset.py            # Offline alignment → episode.hdf5
├── dataset_utils.py            # Path helpers, episode numbering
├── dataset_writer.py           # HDF5 writer (legacy)
│
├── visualize_episode.py        # Full-element video: cameras + 3D + curves
├── diagnose_episode.py         # HDF5 quality check + diagnostic video
├── diagnose_receiver.py        # WebSocket reception diagnostics
├── diagnose_pkl.py             # Raw pkl quality check
├── analyze_timestamps.py       # Multi-sensor timestamp alignment analysis
├── validate_hdf5.py            # HDF5 structure validation
├── check_video.py              # Simple camera playback from HDF5
│
├── replay_demo.py              # Replay recorded demo (joint or Cartesian)
├── go_home.py                  # Move single robot to home position
├── yuv_camera_visualization.py # YUV camera viewer (Cyber RT, Orin-side)
│
├── proto_py/                   # Generated protobuf Python modules
│   └── modules/tools/cyber2pc/proto/
│       └── Cyber2pcBatch_pb2.py
├── protos/                     # Protobuf source definitions
│   └── modules/
│       ├── tools/cyber2pc/proto/
│       │   └── Cyber2pcBatch.proto
│       └── ts_common_msgs/
│           └── proto_custom_msgs/collect/
│               └── tscompressedimage.proto
│
└── dataset/                    # Collected data (not checked in)
    └── <task_name>/
        └── episode_XXX/
            ├── arm.pkl
            ├── gripper.pkl
            ├── left_cam0.pkl
            ├── left_cam1.pkl
            ├── right_cam0.pkl
            ├── right_cam1.pkl
            └── episode.hdf5
```

## Data Format

### Raw Per-Source `.pkl` Files

Each source is recorded independently with its own `time.time_ns()` timestamps.

**`arm.pkl`** — 60 Hz arm state

| Field | Shape | Description |
|-------|-------|-------------|
| `timestamps` | `(N,)` int64 | PC nanosecond timestamps |
| `left_q` / `right_q` | `(N, 7)` float64 | Joint positions (rad) |
| `left_dq` / `right_dq` | `(N, 7)` float64 | Joint velocities (rad/s) |
| `left_tcp_pose` / `right_tcp_pose` | `(N, 7)` float64 | EE pose `[x, y, z, qw, qx, qy, qz]` |
| `left_tcp_vel` / `right_tcp_vel` | `(N, 6)` float64 | EE velocity |

**`<camera>.pkl`** — ~30 Hz camera frames

| Field | Shape | Description |
|-------|-------|-------------|
| `timestamps` | `(N,)` int64 | PC receive timestamps (ns) |
| `frames` | list of `bytes` | JPEG-encoded 640×480 images |

**`gripper.pkl`** — ~30 Hz gripper state

| Field | Shape | Description |
|-------|-------|-------------|
| `left_timestamps` / `right_timestamps` | `(N,)` int64 | PC receive timestamps (ns) |
| `left_pos` / `right_pos` | `(N,)` float64 | Gripper position |

### Aligned `episode.hdf5`

All sources aligned to `left_cam0`'s timeline via nearest-neighbor matching.

```
episode.hdf5
├── timestamp          (N,)    int64     Reference timeline (ns)
├── left_arm/
│   ├── q              (N, 7)  float32   Joint positions
│   ├── tcp_pose       (N, 7)  float32   [x, y, z, qw, qx, qy, qz]
│   └── gripper        (N,)    float32   Gripper position
├── right_arm/
│   ├── q              (N, 7)  float32
│   ├── tcp_pose       (N, 7)  float32
│   └── gripper        (N,)    float32
└── camera/
    ├── left_cam0      (N,)    vlen<uint8>   JPEG bytes
    ├── left_cam1      (N,)    vlen<uint8>
    ├── right_cam0     (N,)    vlen<uint8>
    └── right_cam1     (N,)    vlen<uint8>
```

## Camera Channel Mapping

| Protobuf Channel | HDF5 Name | Position |
|-----------------|-----------|----------|
| `/sensor_camera_senyun/bl_fisheye/compressed` | `left_cam0` | Back-Left |
| `/sensor_camera_senyun/cl_fisheye/compressed` | `left_cam1` | Center-Left |
| `/sensor_camera_senyun/br_fisheye/compressed` | `right_cam0` | Back-Right |
| `/sensor_camera_senyun/cr_fisheye/compressed` | `right_cam1` | Center-Right |

## Orin–PC Communication

The Orin runs a `Cyber2PcComponent` that batches camera H.265 frames and gripper feedback into `Cyber2pcBatch` protobuf messages, sent over a WebSocket connection to the PC (default port 9003).

Each `Cyber2pcBatch` may contain data for multiple channels. The PC-side `receiver_gripper_cam.py` runs an async WebSocket server that:

1. Parses the protobuf batch
2. Dispatches H.265 camera data to a 4-thread pool for parallel decoding (PyAV)
3. Resizes decoded frames to 640×480 and JPEG-encodes for storage
4. Extracts gripper joint positions from `BodyFeedback` messages
5. Stamps all data with PC `time.time_ns()` on arrival

The camera images in the protobuf carry an Orin-side `header.stamp` (sec/nsec) from the `Tscompressedimage` message, but the current system uses PC receive timestamps for alignment. The Orin timestamp could be used for tighter physical synchronization if needed.

## Diagnostic Tools

```bash
# Stage 1: Check WebSocket reception and decode rate
python diagnose_receiver.py --duration 10

# Stage 2: Check raw pkl data quality
python diagnose_pkl.py dataset/<task>/episode_000

# Stage 3: Check aligned HDF5 quality + optional diagnostic video
python diagnose_episode.py dataset/<task>/episode_000/episode.hdf5 --video

# Multi-sensor timestamp analysis (3 levels)
python analyze_timestamps.py dataset/<task>/episode_000
```

## Configuration

Edit `config.py` to match your robot setup:

```python
SN_MAP = {
    "A": "Rizon4s-063219",
    "B": "Rizon4s-063236",
    "C": "Rizon4s-063201",
    "D": "Rizon4s-063237",
}

LEFT_ARM_ID  = "C"      # Left arm serial number key
RIGHT_ARM_ID = "B"      # Right arm serial number key
RECORD_FREQUENCY = 60   # Arm state sampling rate (Hz)
```

The WebSocket port is configured in `receiver_gripper_cam.py` (`PORT = 9003`) and must match the Orin-side `Cyber2PcComponent` configuration.
