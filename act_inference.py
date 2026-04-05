"""
ACT policy inference on dual Flexiv arms with MQTT grippers.

Loads ACT checkpoint trained with ARX_PLAY_plus/mobile_aloha, reads live
camera frames from Cyber2PCObserver and joint states from FlexivRobot,
runs the policy at ~30 Hz, and sends joint commands + gripper targets.

Usage:
    python act_inference.py --ckpt_dir /path/to/weights

    # dry run (print actions, no hardware)
    python act_inference.py --ckpt_dir /path/to/weights --dry-run

    # custom frequency / chunk
    python act_inference.py --ckpt_dir /path/to/weights --freq 30

    # skip gripper
    python act_inference.py --ckpt_dir /path/to/weights --no-gripper
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# ACT model lives in ARX_PLAY_plus/mobile_aloha — add to path so we can
# import the same policy class used during training.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ACT_ROOT = (_SCRIPT_DIR / "../ARX_PLAY_plus/mobile_aloha").resolve()
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from utils.policy import ACTPolicy  # noqa: E402

import config  # noqa: E402  (flexiv-manipulation-toolkit/config.py)

# ---------------------------------------------------------------------------
# Pre-load protobuf modules from proto_py/ and alias them under the bare
# names that gripper/gripper_ctrl.py expects (``import TsWbd_pb2`` etc.).
# This prevents the duplicate-descriptor conflict between proto_py/ and
# gripper/'s own copies of the same .proto definitions.
# ---------------------------------------------------------------------------
_PROTO_PY = str(_SCRIPT_DIR / "proto_py")
if _PROTO_PY not in sys.path:
    sys.path.insert(0, _PROTO_PY)

from modules.ts_common_msgs.proto_basic_msgs import time_pb2 as _time_pb2  # noqa: E402
from modules.ts_common_msgs.proto_std_msgs import Header_pb2 as _Header_pb2  # noqa: E402
from modules.ts_common_msgs.proto_ts_wbd_msgs import TsWbd_pb2 as _TsWbd_pb2  # noqa: E402

sys.modules.setdefault("time_pb2", _time_pb2)
sys.modules.setdefault("Header_pb2", _Header_pb2)
sys.modules.setdefault("TsWbd_pb2", _TsWbd_pb2)

# ---------------------------------------------------------------------------
# Constants matching training (FLEXIV_JOINTS_PER_ARM = 8 in train.py)
# ---------------------------------------------------------------------------
JOINTS_PER_ARM = 8          # 7 joint angles + 1 gripper per arm
ACTION_DIM_HALF = JOINTS_PER_ARM * 2  # first half of the doubled action
LEFT_GRIPPER_IDX = 7        # index within per-arm 8-D vector
RIGHT_GRIPPER_IDX = 15      # same but for right arm in the 16-D action

DEFAULT_MQTT_HOST = "192.168.20.2"
DEFAULT_MQTT_PORT = 1883
LEFT_GRIPPER_NAME = "gripper4"
RIGHT_GRIPPER_NAME = "gripper2"

# Camera name mapping: ACT key -> receiver_gripper_cam shared_buffer key
ACT_TO_LIVE_CAM = {
    "left_leftcam":   "left_cam0",
    "left_rightcam":  "left_cam1",
    "right_leftcam":  "right_cam0",
    "right_rightcam": "right_cam1",
}

HOME_PLAN = "ReturnNewHome"
HOME_TIMEOUT = 120.0


# ===================================================================
# Model loading
# ===================================================================

def load_act_policy(ckpt_dir: str) -> tuple[ACTPolicy, dict, dict]:
    """Load ACT policy + training args + normalization stats."""
    ckpt_dir = Path(ckpt_dir)

    args_path = ckpt_dir / "args.yaml"
    with open(args_path, "r") as f:
        train_args = yaml.safe_load(f)

    stats_path = ckpt_dir / "dataset_stats.pkl"
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    camera_names = train_args["camera_names"]
    chunk_size = train_args["chunk_size"]
    hidden_dim = train_args["hidden_dim"]

    policy_config = {
        "policy_class": "ACT",
        "lr": train_args["lr"],
        "lr_backbone": train_args["lr_backbone"],
        "weight_decay": train_args["weight_decay"],
        "loss_function": train_args["loss_function"],
        "backbone": train_args["backbone"],
        "chunk_size": chunk_size,
        "hidden_dim": hidden_dim,
        "camera_names": camera_names,
        "position_embedding": train_args["position_embedding"],
        "masks": train_args["masks"],
        "dilation": train_args["dilation"],
        "use_base": train_args["use_base"],
        "use_depth_image": train_args["use_depth_image"],
        "enc_layers": train_args["enc_layers"],
        "dec_layers": train_args["dec_layers"],
        "nheads": train_args["nheads"],
        "dropout": train_args["dropout"],
        "pre_norm": train_args["pre_norm"],
        "kl_weight": train_args["kl_weight"],
        "dim_feedforward": train_args["dim_feedforward"],
        "use_qvel": train_args["use_qvel"],
        "use_effort": train_args["use_effort"],
        "use_eef_states": train_args.get("use_eef_states", False),
        "use_eef_action": train_args.get("use_eef_action", False),
        "command_list": [],
        "joints_per_arm": JOINTS_PER_ARM,
    }

    per_arm = JOINTS_PER_ARM
    policy_config["states_dim"] = per_arm
    policy_config["action_dim"] = per_arm

    policy_config["states_dim"] += per_arm if train_args["use_qvel"] else 0
    policy_config["states_dim"] += 1 if train_args["use_effort"] else 0
    policy_config["states_dim"] *= 2

    policy_config["action_dim"] *= 2
    policy_config["action_dim"] += 6 if train_args["use_base"] else 0
    policy_config["action_dim"] *= 2

    policy = ACTPolicy(policy_config)
    ckpt_name = train_args.get("ckpt_name", "policy_best.ckpt")
    ckpt_path = ckpt_dir / "policy_best.ckpt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / ckpt_name
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print(f"Loaded checkpoint: {ckpt_path}  ({loading_status})")

    policy.cuda()
    policy.eval()

    print(f"  action_dim={policy_config['action_dim']}  states_dim={policy_config['states_dim']}")
    print(f"  cameras={camera_names}  chunk_size={chunk_size}")

    return policy, stats, train_args


# ===================================================================
# Observation helpers
# ===================================================================

def build_image_tensor(observer, camera_names: list[str]) -> torch.Tensor:
    """Read live camera frames -> (1, N_cam, 3, H, W) float CUDA tensor."""
    cam_buf = observer.get_cam()
    imgs = []
    for act_name in camera_names:
        live_name = ACT_TO_LIVE_CAM[act_name]
        img = cam_buf.get(live_name)
        if img is None:
            raise RuntimeError(f"Camera {live_name} not available yet (waiting for stream)")
        imgs.append(img)  # (H, W, 3) BGR uint8

    stacked = np.stack(imgs, axis=0)                        # (N, H, W, 3)
    stacked = stacked.transpose(0, 3, 1, 2)                 # (N, 3, H, W)
    tensor = torch.from_numpy(stacked.astype(np.float32) / 255.0)
    return tensor.cuda().unsqueeze(0)                        # (1, N, 3, H, W)


def build_state_tensors(
    left_robot, right_robot, observer,
    stats: dict,
    use_qvel: bool,
    use_effort: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left/right proprioceptive state tensors matching EpisodicDataset layout.

    Per arm: qpos (8) [+ qvel (8)] [+ effort_last (1)]
    Then: left_states = concat(left, right), right_states = same.
    """
    left_s = left_robot.get_state()
    right_s = right_robot.get_state()

    left_q = np.array(left_s["q"], dtype=np.float32)        # (7,)
    right_q = np.array(right_s["q"], dtype=np.float32)       # (7,)

    # Read latest gripper positions from observer shared buffer
    gripper_buf = observer.get_gripper()
    left_g = float(gripper_buf["left"][-1]["pos"]) if gripper_buf["left"] else 0.0
    right_g = float(gripper_buf["right"][-1]["pos"]) if gripper_buf["right"] else 0.0

    left_qpos = np.append(left_q, left_g).astype(np.float32)    # (8,)
    right_qpos = np.append(right_q, right_g).astype(np.float32)  # (8,)

    left_states = left_qpos
    right_states = right_qpos

    if use_qvel:
        left_dq = np.array(left_s["dq"], dtype=np.float32)
        right_dq = np.array(right_s["dq"], dtype=np.float32)
        left_qvel = np.append(left_dq, 0.0).astype(np.float32)
        right_qvel = np.append(right_dq, 0.0).astype(np.float32)
        left_states = np.concatenate([left_states, left_qvel])
        right_states = np.concatenate([right_states, right_qvel])

    if use_effort:
        left_tau = np.array(left_robot.get_joint_torque(), dtype=np.float32)
        right_tau = np.array(right_robot.get_joint_torque(), dtype=np.float32)
        left_states = np.append(left_states, left_tau[-1])
        right_states = np.append(right_states, right_tau[-1])

    combined = np.concatenate([left_states, right_states])
    left_out = combined.copy()
    right_out = combined.copy()

    left_out = (left_out - stats["left_states_mean"]) / stats["left_states_std"]
    right_out = (right_out - stats["right_states_mean"]) / stats["right_states_std"]

    left_t = torch.from_numpy(left_out).float().cuda().unsqueeze(0)
    right_t = torch.from_numpy(right_out).float().cuda().unsqueeze(0)

    return left_t, right_t


def postprocess_action(raw: np.ndarray, stats: dict) -> np.ndarray:
    """Denormalize the first 16 dims (joint action) from the model output."""
    action = raw * stats["action_std"] + stats["action_mean"]
    return action[:ACTION_DIM_HALF]  # (16,) — discard the doubled zero half


# ===================================================================
# Hardware init (same pattern as replay_demo.py)
# ===================================================================

def init_hardware(args):
    from robot.flexiv import FlexivRobot
    _gripper_dir = str(_SCRIPT_DIR / "gripper")
    if _gripper_dir not in sys.path:
        sys.path.insert(0, _gripper_dir)
    from gripper_ctrl import GripperController  # uses pre-aliased proto modules

    left_sn = args.left_sn or config.SN_MAP[config.LEFT_ARM_ID]
    right_sn = args.right_sn or config.SN_MAP[config.RIGHT_ARM_ID]

    print(f"Connecting left arm  [{left_sn}] ...")
    left_robot = FlexivRobot(left_sn)
    print(f"Connecting right arm [{right_sn}] ...")
    right_robot = FlexivRobot(right_sn)

    left_g, right_g = None, None
    if not args.no_gripper:
        left_g = GripperController(gripper_name=LEFT_GRIPPER_NAME,
                                   broker=args.mqtt_host, port=args.mqtt_port)
        right_g = GripperController(gripper_name=RIGHT_GRIPPER_NAME,
                                    broker=args.mqtt_host, port=args.mqtt_port)
        left_g.start()
        right_g.start()
        time.sleep(1.0)
        print("Grippers MQTT connected.")

    return left_robot, right_robot, left_g, right_g


def go_home(left_robot, right_robot):
    print(f"Moving both arms home [{HOME_PLAN}] ...")
    left_robot.go_home_by_plan(plan_name=HOME_PLAN, timeout=HOME_TIMEOUT)
    right_robot.go_home_by_plan(plan_name=HOME_PLAN, timeout=HOME_TIMEOUT)
    print("Both arms at home.")


def send_action(
    left_robot, right_robot,
    left_g, right_g,
    action_16: np.ndarray,
    max_vel: float,
    max_acc: float,
):
    """Decompose 16-D action into left/right joint commands + gripper MQTT."""
    left_joints = action_16[:7].tolist()
    left_gripper = float(action_16[LEFT_GRIPPER_IDX])
    right_joints = action_16[8:15].tolist()
    right_gripper = float(action_16[RIGHT_GRIPPER_IDX])

    max_vel_vec = [max_vel] * 7
    max_acc_vec = [max_acc] * 7
    zero_vel = [0.0] * 7

    left_robot.send_joint_position(
        target_pos=left_joints,
        target_vel=zero_vel,
        max_vel=max_vel_vec,
        max_acc=max_acc_vec,
    )
    right_robot.send_joint_position(
        target_pos=right_joints,
        target_vel=zero_vel,
        max_vel=max_vel_vec,
        max_acc=max_acc_vec,
    )

    if left_g is not None:
        left_g.try_control(position=left_gripper)
    if right_g is not None:
        right_g.try_control(position=right_gripper)


# ===================================================================
# Temporal aggregation (exponential weighting, same as ARX inference)
# ===================================================================

def temporal_agg_step(
    all_time_actions: np.ndarray,
    timestep: int,
    new_actions: np.ndarray,
    chunk_size: int,
    k: float = 0.01,
) -> np.ndarray:
    """Write chunk into buffer, read out weighted aggregate for current step."""
    all_time_actions[timestep, timestep:timestep + chunk_size] = new_actions
    col = all_time_actions[:, timestep]
    populated = np.all(col != 0, axis=1)
    col = col[populated]
    weights = np.exp(-k * np.arange(len(col)))
    weights /= weights.sum()
    return (col * weights[:, np.newaxis]).sum(axis=0)


# ===================================================================
# Main inference loop
# ===================================================================

def inference_loop(args, policy, stats, train_args, observer,
                   left_robot, right_robot, left_g, right_g):
    camera_names = train_args["camera_names"]
    chunk_size = train_args["chunk_size"]
    action_dim = policy.model.action_dim if hasattr(policy.model, "action_dim") else ACTION_DIM_HALF * 2
    use_qvel = train_args["use_qvel"]
    use_effort = train_args["use_effort"]
    temporal_agg = train_args.get("temporal_agg", True)

    max_steps = args.max_steps
    dt = 1.0 / args.freq

    if temporal_agg:
        buf_len = max_steps + chunk_size
        all_time_actions = np.zeros((max_steps, buf_len, action_dim), dtype=np.float32)

    print(f"\nInference: {max_steps} steps, {args.freq} Hz, temporal_agg={temporal_agg}")
    print("Waiting for camera stream ...")

    while True:
        cam_buf = observer.get_cam()
        if all(cam_buf.get(ACT_TO_LIVE_CAM[n]) is not None for n in camera_names):
            break
        time.sleep(0.1)
    print("Camera stream ready.")

    dry_run = args.dry_run

    if not dry_run:
        left_robot.to_joint_position_mode()
        right_robot.to_joint_position_mode()

    print("Running policy ... (Ctrl+C to stop)\n")

    try:
        with torch.inference_mode():
            for t in range(max_steps):
                t0 = time.time()

                image_tensor = build_image_tensor(observer, camera_names)

                if dry_run:
                    per = JOINTS_PER_ARM
                    dummy = np.zeros(per, dtype=np.float32)
                    combined = np.concatenate([dummy, dummy])  # left(8) + right(8) = 16
                    left_s = (combined - stats["left_states_mean"]) / stats["left_states_std"]
                    right_s = (combined - stats["right_states_mean"]) / stats["right_states_std"]
                    left_s = torch.from_numpy(left_s).float().cuda().unsqueeze(0)
                    right_s = torch.from_numpy(right_s).float().cuda().unsqueeze(0)
                else:
                    left_s, right_s = build_state_tensors(
                        left_robot, right_robot, observer,
                        stats, use_qvel, use_effort,
                    )

                all_actions, _ = policy(image_tensor, None, left_s, right_s)
                # all_actions: (1, chunk_size, action_dim)

                if temporal_agg:
                    raw = temporal_agg_step(
                        all_time_actions, t,
                        all_actions.cpu().numpy()[0],
                        chunk_size,
                    )
                else:
                    raw = all_actions[0, 0].cpu().numpy()

                action_16 = postprocess_action(raw, stats)

                if dry_run:
                    if t % 30 == 0:
                        print(f"  t={t:4d}  L_q={action_16[:7].round(3)}  "
                              f"L_g={action_16[7]:.3f}  "
                              f"R_q={action_16[8:15].round(3)}  "
                              f"R_g={action_16[15]:.3f}")
                else:
                    send_action(left_robot, right_robot, left_g, right_g,
                                action_16, args.max_vel, args.max_acc)

                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if t > 0 and t % 100 == 0:
                    actual_hz = 1.0 / max(time.time() - t0, 1e-6)
                    print(f"  step {t}/{max_steps}  loop={elapsed*1000:.1f}ms  ~{actual_hz:.0f}Hz")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("Inference finished.")


# ===================================================================
# Entry point
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="ACT policy inference on Flexiv dual arms")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Directory with policy_best.ckpt, args.yaml, dataset_stats.pkl")
    p.add_argument("--max-steps", type=int, default=10000, help="Max inference steps")
    p.add_argument("--freq", type=float, default=30.0, help="Control loop frequency (Hz)")
    p.add_argument("--max-vel", type=float, default=1.0, help="Joint max velocity (rad/s)")
    p.add_argument("--max-acc", type=float, default=2.0, help="Joint max acceleration (rad/s²)")
    p.add_argument("--no-gripper", action="store_true", help="Skip gripper commands")
    p.add_argument("--mqtt-host", type=str, default=DEFAULT_MQTT_HOST)
    p.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    p.add_argument("--left-sn", default=None, help="Left arm serial number override")
    p.add_argument("--right-sn", default=None, help="Right arm serial number override")
    p.add_argument("--home-plan", type=str, default=HOME_PLAN, help="Home plan name")
    p.add_argument("--dry-run", action="store_true",
                   help="Print actions without connecting to hardware")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load model
    policy, stats, train_args = load_act_policy(args.ckpt_dir)

    # 2. Start camera/gripper receiver
    from ee.cyber2pc_observer import Cyber2PCObserver
    observer = Cyber2PCObserver()
    observer.start()
    print("Cyber2PCObserver started (camera + gripper stream).")

    if args.dry_run:
        print("\n=== DRY RUN: no hardware connection ===\n")
        # Wait for camera frames then run loop printing actions
        inference_loop(args, policy, stats, train_args, observer,
                       None, None, None, None)
        return

    # 3. Init hardware
    left_robot, right_robot, left_g, right_g = init_hardware(args)

    # 4. Go home
    input("\nPress ENTER to go home and start inference ...")
    go_home(left_robot, right_robot)

    # 5. Run
    try:
        inference_loop(args, policy, stats, train_args, observer,
                       left_robot, right_robot, left_g, right_g)
    finally:
        for label, robot in [("left", left_robot), ("right", right_robot)]:
            try:
                robot.stop()
            except Exception as e:
                print(f"Warning: failed to stop {label} robot: {e}")
        for label, g in [("left", left_g), ("right", right_g)]:
            if g is not None:
                try:
                    g.stop()
                except Exception as e:
                    print(f"Warning: failed to stop {label} gripper: {e}")
        observer.stop()


if __name__ == "__main__":
    main()
