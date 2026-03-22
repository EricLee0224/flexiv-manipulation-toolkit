# Flexiv Manipulation Toolkit

Kinesthetic demonstration data collection, replay, and visualization system for dual Flexiv Rizon4s arms with multi-camera observation from an NVIDIA Orin edge computer.

The system records synchronized data from **7 sources**: left/right arm joint states (60 Hz), left/right gripper positions (~30 Hz via MQTT), and 4 fisheye camera streams (30 Hz H.265 via WebSocket). All sources are independently timestamped and offline-aligned into a single HDF5 episode file.

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
                                          MQTT (protobuf)            │  gripper/gripper_ctrl.py      │
┌────────────────────────────────┐ ◀────────────────────────────────▶│    └─ gripper cmd/feedback    │
│   Orin MQTT broker             │       192.168.20.2:1883           │                               │
│   gripper2 (left)              │                                   │  recorder.py (ArmRecorder)    │
│   gripper4 (right)             │                                   │    └─ arm state @ 60 Hz      │
└────────────────────────────────┘                                   │                               │
                                                                     │  data_collection.py           │
┌────────────────────────────────┐       Flexiv RDK (ethernet)      │    └─ keyboard UI + save      │
│   Flexiv Rizon4s × 2          │ ◀────────────────────────────────▶│                               │
│   Left:  Rizon4s-063201       │       joint/TCP state + control   │  replay_demo.py               │
│   Right: Rizon4s-063236       │                                   │    └─ replay + gripper sync   │
└────────────────────────────────┘                                   └──────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
conda activate flexiv_env

# Core dependencies
pip install numpy h5py opencv-python websockets av protobuf paho-mqtt

# Visualization & analysis
pip install matplotlib scipy
```

The Flexiv RDK Python bindings (`flexivrdk`) must be installed following Flexiv's documentation (v1.8).

### 1. Teach Mode

Put one or both arms into compliant teach mode for manual guidance:

```bash
# Both arms (default)
python teach_mode.py

# Single arm
python teach_mode.py --arm left
python teach_mode.py --arm right
```

Press Enter or Ctrl-C to exit. The arms return to IDLE automatically.

### 2. Collect Data

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

### 3. Build Aligned Dataset

```bash
# Single episode
python build_dataset.py dataset/<task>/episode_000

# All episodes under a task (batch)
python build_dataset.py dataset/<task>
```

Uses `left_cam0` as the 30 Hz reference timeline. All other sources are aligned via nearest-neighbor timestamp matching. Outputs `episode.hdf5` in each episode directory.

### 4. Validate

```bash
python validate_hdf5.py dataset/<task>/episode_000/episode.hdf5
```

Checks HDF5 structure, timestamp monotonicity, shape consistency, JPEG decodability, and frozen-frame detection.

### 5. Visualize

```bash
python visualize_episode.py dataset/<task>/episode_000
```

Generates a 1920×720 real-time video (`full_visualization.mp4`) with:
- 2×2 camera grid (left panel)
- 3D end-effector trajectory with orientation frames (right-top)
- Euler angle curves + gripper state with time cursor (right-bottom)

### 6. Replay a Demonstration

```bash
# Joint replay (recommended for first test)
python replay_demo.py dataset/<task>/episode_000

# Cartesian replay
python replay_demo.py dataset/<task>/episode_000 --mode cartesian

# Half speed
python replay_demo.py dataset/<task>/episode_000 --speed-scale 0.5

# Without gripper commands
python replay_demo.py dataset/<task>/episode_000 --no-gripper

# Dry run: print commands without connecting to hardware
python replay_demo.py dataset/<task>/episode_000 --dry-run
python replay_demo.py dataset/<task>/episode_000 --dry-run --mode cartesian --sample 20
```

Both robots go home first (via `ReturnNewHome` plan), then replay recorded trajectories at 30 Hz with synchronized gripper commands sent to the Orin via MQTT. Progress bar shows real-time status.

### 7. Go Home

```bash
# Both arms (default)
python go_home.py

# Single arm
python go_home.py --arm left
python go_home.py --arm right

# Custom plan and timeout
python go_home.py --plan MyCustomHome --timeout 60
```

## Project Structure

```
flexiv-manipulation-toolkit/
│
├── teach_mode.py               # Teach mode (kinesthetic guidance)
├── data_collection.py          # Main collection script (keyboard UI)
├── build_dataset.py            # Offline alignment → episode.hdf5
├── validate_hdf5.py            # HDF5 structure & quality validation
├── visualize_episode.py        # Full-element video: cameras + 3D + curves
├── replay_demo.py              # Replay demo (joint/cartesian + gripper)
├── go_home.py                  # Move arms to home position
│
├── config.py                   # Robot serial numbers, arm IDs, frequencies
├── recorder.py                 # ArmRecorder: threaded arm state sampling
├── receiver_gripper_cam.py     # WebSocket server: H.265 decode + gripper rx
├── dataset_utils.py            # Path helpers, episode numbering
│
├── robot/
│   ├── flexiv.py               # Flexiv RDK v1.8 wrapper (modes, control, state)
│   └── flexiv_arm.py           # Teach-mode arm with keep-alive loop
├── ee/
│   └── cyber2pc_observer.py    # Thread wrapper for receiver_gripper_cam
├── gripper/
│   ├── gripper_ctrl.py         # MQTT gripper controller (TsWbd protobuf)
│   └── main.py                 # Gripper standalone demo
│
├── diagnose_episode.py         # HDF5 quality check + diagnostic video
├── diagnose_receiver.py        # WebSocket reception diagnostics
├── diagnose_pkl.py             # Raw pkl quality check
├── analyze_timestamps.py       # Multi-sensor timestamp alignment analysis
├── check_video.py              # Simple camera playback from HDF5
├── yuv_camera_visualization.py # YUV camera viewer (Cyber RT, Orin-side)
│
├── proto_py/                   # Generated protobuf Python modules
│   └── modules/tools/cyber2pc/proto/
│       └── Cyber2pcBatch_pb2.py
├── protos/                     # Protobuf source definitions
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
            ├── episode.hdf5
            └── full_visualization.mp4
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

## Gripper Configuration

Grippers are controlled via MQTT through the Orin (`192.168.20.2:1883`):

| Gripper | MQTT Topic | Name |
|---------|-----------|------|
| Left  | `tars/motor/gripper2` | `gripper2` |
| Right | `tars/motor/gripper4` | `gripper4` |

The `GripperController` sends `TsWbd.BodyCommands` protobuf messages with position targets and receives `BodyFeedback` with position/velocity/torque measurements.

## Orin–PC Communication

The Orin runs a `Cyber2PcComponent` that batches camera H.265 frames and gripper feedback into `Cyber2pcBatch` protobuf messages, sent over a WebSocket connection to the PC (default port 9003).

Each `Cyber2pcBatch` may contain data for multiple channels. The PC-side `receiver_gripper_cam.py` runs an async WebSocket server that:

1. Parses the protobuf batch
2. Dispatches H.265 camera data to a 4-thread pool for parallel decoding (PyAV)
3. Resizes decoded frames to 640×480 and JPEG-encodes for storage
4. Extracts gripper joint positions from `BodyFeedback` messages
5. Stamps all data with PC `time.time_ns()` on arrival

## Diagnostic Tools

```bash
# Check WebSocket reception and decode rate
python diagnose_receiver.py --duration 10

# Check raw pkl data quality
python diagnose_pkl.py dataset/<task>/episode_000

# Check aligned HDF5 quality + optional diagnostic video
python diagnose_episode.py dataset/<task>/episode_000/episode.hdf5 --video

# Multi-sensor timestamp analysis
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

Most scripts accept `--left-sn` / `--right-sn` overrides to use different arms without editing config.
