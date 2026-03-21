import pickle
import time
import numpy as np

from robot.flexiv_arm import FlexivArm
from ee.cyber2pc_observer import Cyber2PCObserver
from recorder import ArmRecorder
from dataset_utils import get_next_episode_dir

import config


def save_arm_pkl(arm_data, path):
    """Stack arm recordings into arrays and save as .pkl."""
    if not arm_data:
        print("  arm: no data")
        return

    timestamps = np.array([d["ts"] for d in arm_data], dtype=np.int64)

    def stack_field(side, key):
        return np.array([d[side][key] for d in arm_data])

    payload = {
        "timestamps": timestamps,
        "left_q": stack_field("left", "q"),
        "left_dq": stack_field("left", "dq"),
        "left_tcp_pose": stack_field("left", "tcp_pose"),
        "left_tcp_vel": stack_field("left", "tcp_vel"),
        "right_q": stack_field("right", "q"),
        "right_dq": stack_field("right", "dq"),
        "right_tcp_pose": stack_field("right", "tcp_pose"),
        "right_tcp_vel": stack_field("right", "tcp_vel"),
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    n = len(timestamps)
    dt_ms = np.median(np.diff(timestamps)) / 1e6 if n >= 2 else 0
    print(f"  arm.pkl: {n} frames, ~{1e3/dt_ms:.0f} Hz" if dt_ms > 0 else f"  arm.pkl: {n} frames")


def save_cam_pkl(cam_data, episode_dir):
    """Save each camera channel as a separate .pkl."""
    for name, frames in cam_data.items():
        path = episode_dir / f"{name}.pkl"

        if not frames:
            print(f"  {name}.pkl: no data")
            continue

        timestamps = np.array([f["ts"] for f in frames], dtype=np.int64)
        jpeg_list = [f["data"] for f in frames]

        payload = {"timestamps": timestamps, "frames": jpeg_list}

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        n = len(timestamps)
        dt_ms = np.median(np.diff(timestamps)) / 1e6 if n >= 2 else 0
        print(f"  {name}.pkl: {n} frames, ~{1e3/dt_ms:.0f} Hz" if dt_ms > 0 else f"  {name}.pkl: {n} frames")


def save_gripper_pkl(gripper_data, path):
    """Save gripper recordings as .pkl."""
    payload = {}
    for side in ("left", "right"):
        entries = gripper_data.get(side, [])
        if entries:
            payload[f"{side}_timestamps"] = np.array([e["ts"] for e in entries], dtype=np.int64)
            payload[f"{side}_pos"] = np.array([e["pos"] for e in entries], dtype=np.float64)
        else:
            payload[f"{side}_timestamps"] = np.empty(0, dtype=np.int64)
            payload[f"{side}_pos"] = np.empty(0, dtype=np.float64)

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    for side in ("left", "right"):
        n = len(payload[f"{side}_timestamps"])
        print(f"  gripper {side}: {n} samples")


def main():

    left_sn = config.SN_MAP[config.LEFT_ARM_ID]
    right_sn = config.SN_MAP[config.RIGHT_ARM_ID]

    task_name = input("Enter task name: ").strip()
    if not task_name:
        print("Task name cannot be empty.")
        return

    # --------------------
    # Init robots
    # --------------------

    arm_left = FlexivArm(left_sn)
    arm_right = FlexivArm(right_sn)

    # --------------------
    # Start Cyber2PC observer
    # --------------------

    observer = Cyber2PCObserver()
    observer.start()

    # --------------------
    # Arm recorder (arm-only, high rate)
    # --------------------

    arm_recorder = ArmRecorder(arm_left, arm_right, freq=config.RECORD_FREQUENCY)

    print("")
    print("Robots initialized.")
    print("Keyboard control:")
    print(" e -> enable teach")
    print(" s -> disable teach")
    print(" r -> start recording")
    print(" t -> stop recording & save")
    print(" q -> quit")
    print("")

    try:

        while True:

            cmd = input("> ").strip().lower()

            if cmd == "e":

                arm_left.enable_teach()
                arm_right.enable_teach()
                print("Teach mode enabled.")

            elif cmd == "s":

                arm_left.disable_teach()
                arm_right.disable_teach()
                print("Teach mode disabled.")

            elif cmd == "r":

                observer.start_recording()
                arm_recorder.start()
                print("Recording started (all sources independent).")

            elif cmd == "t":

                print("Flushing camera decode pipeline ...")
                observer.stop_recording()
                arm_recorder.stop()
                print("Flush complete.")

                from receiver_gripper_cam import get_decode_stats
                for ch, st in get_decode_stats().items():
                    short = ch.split("/")[-2] if "/" in ch else ch
                    elapsed = time.time() - st["start_time"]
                    fps = st["frames_out"] / elapsed if elapsed > 0 else 0
                    err = st["errors"]
                    print(f"  decode [{short}]: {st['frames_out']} frames in {elapsed:.1f}s ({fps:.1f} fps), {err} errors")

                episode_dir = get_next_episode_dir(task_name)
                print(f"Saving to {episode_dir} ...")

                save_arm_pkl(arm_recorder.get_data(), episode_dir / "arm.pkl")
                save_cam_pkl(observer.get_recorded_cam(), episode_dir)
                save_gripper_pkl(observer.get_recorded_gripper(), episode_dir / "gripper.pkl")

                print(f"Episode saved → {episode_dir}")
                print("Ready. Press 'r' to record again, 'q' to quit.")

            elif cmd == "q":

                break

    finally:

        if arm_recorder.is_running():
            arm_recorder.stop()

        arm_left.disable_teach()
        arm_right.disable_teach()

        observer.stop()

        print("Program finished.")


if __name__ == "__main__":
    main()
