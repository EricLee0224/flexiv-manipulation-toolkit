# replay_demo.py
import time
import argparse
import h5py
import numpy as np

import config
from robot.flexiv import FlexivRobot

# Replay a recorded dual-arm demo from HDF5.
#
# Supported replay modes:
#   1) joint
#      - replay recorded joint positions
#      - more stable
#      - recommended for first validation
#
#   2) cartesian
#      - replay recorded TCP poses
#      - closer to demonstration semantics
#      - more sensitive to control behavior
#
# Safety / workflow:
#   - before replay starts, BOTH robots will first execute the home plan
#   - default home plan name: "ReturnNewHome"
#   - recommended to test joint replay first at reduced speed
#
# Typical usage:
#
#   # joint replay, safer first test
#   python replay_demo.py --mode joint --speed-scale 0.5 --max-vel 0.5 --max-acc 1.0
#
#   # joint replay using recorded timing
#   python replay_demo.py --mode joint
#
#   # cartesian replay at half speed
#   python replay_demo.py --mode cartesian --speed-scale 0.5
#
#   # replay with explicit frequency
#   python replay_demo.py --mode joint --freq 30
#
#   # replay another dataset file
#   python replay_demo.py --file episode_000.hdf5 --mode joint
#
# Notes:
#   - this script replays recorded STATE trajectories, not raw control actions
#   - replay frequency is inferred from timestamps unless --freq is specified
#   - speed-scale > 1.0 means faster replay, < 1.0 means slower replay


def load_demo(file_path: str):
    print(f"Loading dataset: {file_path}")

    with h5py.File(file_path, "r") as f:
        data = {
            "timestamp": f["timestamp"][:],
            "left_q": f["left_arm/q"][:],
            "left_tcp_pose": f["left_arm/tcp_pose"][:],
            "right_q": f["right_arm/q"][:],
            "right_tcp_pose": f["right_arm/tcp_pose"][:],
        }

    print(f"Frames: {len(data['timestamp'])}")
    return data


def compute_dt_from_timestamps(timestamp: np.ndarray, default_freq: float) -> float:
    if len(timestamp) < 2:
        return 1.0 / default_freq

    dt = np.diff(timestamp).astype(np.float64) / 1e9
    median_dt = float(np.median(dt))

    if median_dt <= 0:
        return 1.0 / default_freq

    return median_dt


def go_home_both(left_robot: FlexivRobot, right_robot: FlexivRobot, plan_name: str = "ReturnNewHome"):
    print(f"Moving both robots to home using plan [{plan_name}] ...")

    left_robot.go_home_by_plan(plan_name=plan_name, timeout=120.0)
    right_robot.go_home_by_plan(plan_name=plan_name, timeout=120.0)

    print("Both robots are at home.")


def replay_joint(
    left_robot: FlexivRobot,
    right_robot: FlexivRobot,
    left_q: np.ndarray,
    right_q: np.ndarray,
    dt: float,
    max_vel: float,
    max_acc: float,
):
    print("Switching both robots to joint position mode ...")
    left_robot.to_joint_position_mode()
    right_robot.to_joint_position_mode()

    max_vel_vec = [max_vel] * 7
    max_acc_vec = [max_acc] * 7
    zero_vel = [0.0] * 7

    n = len(left_q)
    print("Replay mode: joint")

    for i in range(n):
        t0 = time.time()

        left_robot.send_joint_position(
            target_pos=left_q[i].tolist(),
            target_vel=zero_vel,
            max_vel=max_vel_vec,
            max_acc=max_acc_vec,
        )

        right_robot.send_joint_position(
            target_pos=right_q[i].tolist(),
            target_vel=zero_vel,
            max_vel=max_vel_vec,
            max_acc=max_acc_vec,
        )

        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def replay_cartesian(
    left_robot: FlexivRobot,
    right_robot: FlexivRobot,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    dt: float,
):
    print("Switching both robots to Cartesian motion-force mode ...")
    left_robot.to_cartesian_motion_force_mode()
    right_robot.to_cartesian_motion_force_mode()

    print("Replay mode: cartesian")

    n = len(left_pose)
    for i in range(n):
        t0 = time.time()

        left_robot.send_cartesian_motion_force(left_pose[i].tolist())
        right_robot.send_cartesian_motion_force(right_pose[i].tolist())

        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=config.DATASET_FILE,
        help="Path to demo hdf5 file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["joint", "cartesian"],
        default="joint",
        help="Replay mode",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=None,
        help="Replay frequency in Hz. If not set, infer from timestamps.",
    )
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=1.0,
        help="Replay speed scale. >1 faster, <1 slower.",
    )
    parser.add_argument(
        "--max-vel",
        type=float,
        default=1.0,
        help="Joint replay max velocity per joint",
    )
    parser.add_argument(
        "--max-acc",
        type=float,
        default=2.0,
        help="Joint replay max acceleration per joint",
    )
    parser.add_argument(
        "--home-plan",
        type=str,
        default="ReturnNewHome",
        help="Plan name used to move both robots home before replay",
    )
    args = parser.parse_args()

    data = load_demo(args.file)

    left_sn = config.SN_MAP[config.LEFT_ARM_ID]
    right_sn = config.SN_MAP[config.RIGHT_ARM_ID]

    print(f"Connecting left robot : {config.LEFT_ARM_ID} -> {left_sn}")
    left_robot = FlexivRobot(left_sn)

    print(f"Connecting right robot: {config.RIGHT_ARM_ID} -> {right_sn}")
    right_robot = FlexivRobot(right_sn)

    if args.freq is None:
        base_dt = compute_dt_from_timestamps(data["timestamp"], config.RECORD_FREQUENCY)
    else:
        base_dt = 1.0 / args.freq

    dt = base_dt / args.speed_scale

    print(f"Replay dt: {dt:.6f} s")
    print(f"Replay freq: {1.0 / dt:.2f} Hz")

    input("Press ENTER to move both robots home and start replay...")

    try:
        go_home_both(left_robot, right_robot, plan_name=args.home_plan)

        if args.mode == "joint":
            replay_joint(
                left_robot=left_robot,
                right_robot=right_robot,
                left_q=data["left_q"],
                right_q=data["right_q"],
                dt=dt,
                max_vel=args.max_vel,
                max_acc=args.max_acc,
            )
        else:
            replay_cartesian(
                left_robot=left_robot,
                right_robot=right_robot,
                left_pose=data["left_tcp_pose"],
                right_pose=data["right_tcp_pose"],
                dt=dt,
            )

        print("Replay finished.")

    finally:
        try:
            left_robot.stop()
        except Exception as e:
            print(f"Warning: failed to stop left robot: {e}")

        try:
            right_robot.stop()
        except Exception as e:
            print(f"Warning: failed to stop right robot: {e}")


if __name__ == "__main__":
    main()