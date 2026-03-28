"""
Replay a recorded dual-arm episode from HDF5, including gripper commands.

Supported replay modes:
  1) joint     – replay recorded joint positions (more stable, recommended first)
  2) cartesian – replay recorded TCP poses (closer to demonstration semantics)

Gripper positions are sent to the Orin via MQTT at the same rate as arm commands.

Safety / workflow:
  - before replay, BOTH robots execute the home plan
  - recommended to test joint replay first at reduced speed

Usage:
    # replay an episode directory (looks for episode.hdf5 inside)
    python replay_demo.py dataset/place_tube_0322/episode_000

    # replay with explicit HDF5 path
    python replay_demo.py --file demo.hdf5

    # joint replay at half speed, limited velocity
    python replay_demo.py dataset/place_tube_0322/episode_000 --mode joint --speed-scale 0.5

    # cartesian replay
    python replay_demo.py dataset/place_tube_0322/episode_000 --mode cartesian

    # skip gripper commands
    python replay_demo.py dataset/place_tube_0322/episode_000 --no-gripper

    # dry run: print commands without connecting to hardware
    python replay_demo.py dataset/place_tube_0322/episode_000 --dry-run
    python replay_demo.py dataset/place_tube_0322/episode_000 --dry-run --sample 20
"""

import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np

import config


MQTT_BROKER_IP = "192.168.20.2"
MQTT_BROKER_PORT = 1883
LEFT_GRIPPER_NAME = "gripper4"
RIGHT_GRIPPER_NAME = "gripper2"


def resolve_hdf5_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        candidate = p / "episode.hdf5"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No episode.hdf5 in {p}")
    if p.exists():
        return p
    raise FileNotFoundError(f"{p} does not exist")


def load_episode(file_path: Path) -> dict:
    print(f"Loading: {file_path}")
    with h5py.File(file_path, "r") as f:
        data = {
            "timestamp": f["timestamp"][:],
            "left_q": f["left_arm/q"][:],
            "left_tcp_pose": f["left_arm/tcp_pose"][:],
            "right_q": f["right_arm/q"][:],
            "right_tcp_pose": f["right_arm/tcp_pose"][:],
            "left_gripper": f["left_arm/gripper"][:],
            "right_gripper": f["right_arm/gripper"][:],
        }
    n = len(data["timestamp"])
    dur = (data["timestamp"][-1] - data["timestamp"][0]) / 1e9 if n >= 2 else 0
    print(f"  {n} frames, {dur:.1f}s")
    return data


def compute_dt(timestamps: np.ndarray, default_freq: float) -> float:
    if len(timestamps) < 2:
        return 1.0 / default_freq
    dt = float(np.median(np.diff(timestamps).astype(np.float64) / 1e9))
    return dt if dt > 0 else 1.0 / default_freq


def _import_hardware():
    """Lazy import of hardware dependencies (flexivrdk, gripper MQTT)."""
    from robot.flexiv import FlexivRobot

    sys.path.insert(0, str(Path(__file__).resolve().parent / "gripper"))
    from gripper_ctrl import GripperController

    return FlexivRobot, GripperController


def go_home_both(left, right, plan: str):
    print(f"Moving both robots home [{plan}] ...")
    left.go_home_by_plan(plan_name=plan, timeout=120.0)
    right.go_home_by_plan(plan_name=plan, timeout=120.0)
    print("Both robots at home.")


def init_grippers(GripperController):
    left_g = GripperController(gripper_name=LEFT_GRIPPER_NAME,
                               broker=MQTT_BROKER_IP, port=MQTT_BROKER_PORT)
    right_g = GripperController(gripper_name=RIGHT_GRIPPER_NAME,
                                broker=MQTT_BROKER_IP, port=MQTT_BROKER_PORT)
    left_g.start()
    right_g.start()
    time.sleep(0.5)
    print(f"Grippers connected: left={LEFT_GRIPPER_NAME}, right={RIGHT_GRIPPER_NAME}")
    return left_g, right_g


def _print_progress(i: int, n: int, start_time: float, dt: float):
    if i == 0 or (i + 1) % 30 != 0 and i != n - 1:
        return
    pct = (i + 1) / n
    elapsed = time.time() - start_time
    remaining = (n - i - 1) * dt
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {bar} {pct*100:5.1f}%  {i+1}/{n}  elapsed {elapsed:.1f}s  ETA {remaining:.1f}s", end="", flush=True)
    if i == n - 1:
        print()


def replay_joint(
    left_robot, right_robot,
    left_g, right_g,
    data: dict, dt: float,
    max_vel: float, max_acc: float,
):
    left_robot.to_joint_position_mode()
    right_robot.to_joint_position_mode()

    max_vel_vec = [max_vel] * 7
    max_acc_vec = [max_acc] * 7
    zero_vel = [0.0] * 7

    n = len(data["left_q"])
    total_sec = n * dt
    print(f"Replaying {n} frames (joint mode, dt={dt*1000:.1f}ms, ~{total_sec:.1f}s) ...")

    replay_start = time.time()
    for i in range(n):
        t0 = time.time()

        left_robot.send_joint_position(
            target_pos=data["left_q"][i].tolist(),
            target_vel=zero_vel,
            max_vel=max_vel_vec,
            max_acc=max_acc_vec,
        )
        right_robot.send_joint_position(
            target_pos=data["right_q"][i].tolist(),
            target_vel=zero_vel,
            max_vel=max_vel_vec,
            max_acc=max_acc_vec,
        )

        if left_g is not None:
            left_g.try_control(position=float(data["left_gripper"][i]))
        if right_g is not None:
            right_g.try_control(position=float(data["right_gripper"][i]))

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        _print_progress(i, n, replay_start, dt)


def replay_cartesian(
    left_robot, right_robot,
    left_g, right_g,
    data: dict, dt: float,
):
    print("Zeroing F/T sensors ...")
    left_robot.zero_ft_sensor()
    right_robot.zero_ft_sensor()

    print("Switching to cartesian motion-force mode ...")
    left_robot.stop()
    right_robot.stop()
    left_robot.to_cartesian_motion_force_mode()
    right_robot.to_cartesian_motion_force_mode()

    n = len(data["left_tcp_pose"])
    total_sec = n * dt
    print(f"Replaying {n} frames (cartesian mode, dt={dt*1000:.1f}ms, ~{total_sec:.1f}s) ...")

    replay_start = time.time()
    for i in range(n):
        t0 = time.time()

        left_robot.send_cartesian_motion_force(data["left_tcp_pose"][i].tolist())
        right_robot.send_cartesian_motion_force(data["right_tcp_pose"][i].tolist())

        if left_g is not None:
            left_g.try_control(position=float(data["left_gripper"][i]))
        if right_g is not None:
            right_g.try_control(position=float(data["right_gripper"][i]))

        elapsed = time.time() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        _print_progress(i, n, replay_start, dt)

        if i % 300 == 0:
            print(f"  {i}/{n} ({100*i/n:.0f}%)")


def fmt_arr(arr, precision=4):
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


def dry_run(data: dict, dt: float, args):
    n = len(data["timestamp"])
    ts = data["timestamp"]
    ts_sec = (ts - ts[0]) / 1e9

    max_vel_vec = [args.max_vel] * 7
    max_acc_vec = [args.max_acc] * 7

    sample_count = min(args.sample, n)
    indices = np.linspace(0, n - 1, sample_count, dtype=int)

    print("")
    print("=" * 80)
    print(f"DRY RUN — {n} frames, {ts_sec[-1]:.1f}s, mode={args.mode}")
    print(f"  dt={dt*1000:.1f}ms ({1/dt:.1f}Hz), speed_scale={args.speed_scale}")
    if args.mode == "joint":
        print(f"  max_vel={args.max_vel}, max_acc={args.max_acc}")
    print(f"  gripper={'OFF' if args.no_gripper else 'ON'}")
    print(f"  showing {sample_count} sampled frames")
    print("=" * 80)

    for i in indices:
        t = ts_sec[i]
        print(f"\n--- frame {i}/{n}  t={t:.2f}s ---")

        if args.mode == "joint":
            print(f"  LEFT  SendJointPosition:")
            print(f"    target_pos = {fmt_arr(data['left_q'][i])}")
            print(f"    target_vel = {fmt_arr([0.0]*7)}")
            print(f"    max_vel    = {fmt_arr(max_vel_vec)}")
            print(f"    max_acc    = {fmt_arr(max_acc_vec)}")
            print(f"  RIGHT SendJointPosition:")
            print(f"    target_pos = {fmt_arr(data['right_q'][i])}")
            print(f"    target_vel = {fmt_arr([0.0]*7)}")
            print(f"    max_vel    = {fmt_arr(max_vel_vec)}")
            print(f"    max_acc    = {fmt_arr(max_acc_vec)}")
        else:
            print(f"  LEFT  SendCartesianMotionForce:")
            print(f"    pose = {fmt_arr(data['left_tcp_pose'][i])}")
            print(f"  RIGHT SendCartesianMotionForce:")
            print(f"    pose = {fmt_arr(data['right_tcp_pose'][i])}")

        if not args.no_gripper:
            lp = float(data["left_gripper"][i])
            rp = float(data["right_gripper"][i])
            print(f"  LEFT  gripper  → MQTT {LEFT_GRIPPER_NAME}  position={lp:.4f}")
            print(f"  RIGHT gripper  → MQTT {RIGHT_GRIPPER_NAME}  position={rp:.4f}")

    # summary stats
    print("\n" + "=" * 80)
    print("DATA RANGE SUMMARY")
    print("=" * 80)
    for side in ("left", "right"):
        q = data[f"{side}_q"]
        tcp = data[f"{side}_tcp_pose"]
        g = data[f"{side}_gripper"]
        print(f"\n  {side.upper()} ARM:")
        print(f"    q        min = {fmt_arr(q.min(axis=0))}")
        print(f"    q        max = {fmt_arr(q.max(axis=0))}")
        print(f"    tcp xyz  min = {fmt_arr(tcp[:,:3].min(axis=0))}")
        print(f"    tcp xyz  max = {fmt_arr(tcp[:,:3].max(axis=0))}")
        print(f"    gripper  min = {g.min():.4f},  max = {g.max():.4f}")

    print("")


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded dual-arm episode.")
    parser.add_argument(
        "episode", nargs="?", default=None,
        help="Episode directory or HDF5 file path",
    )
    parser.add_argument("--file", type=str, default=None, help="(legacy) HDF5 file path")
    parser.add_argument(
        "--mode", choices=["joint", "cartesian"], default="joint",
        help="Replay mode (default: joint)",
    )
    parser.add_argument("--freq", type=float, default=None, help="Override replay frequency (Hz)")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--max-vel", type=float, default=1.0, help="Joint max velocity (rad/s)")
    parser.add_argument("--max-acc", type=float, default=2.0, help="Joint max acceleration (rad/s²)")
    parser.add_argument("--home-plan", type=str, default="ReturnNewHome", help="Home plan name")
    parser.add_argument("--no-gripper", action="store_true", help="Skip gripper replay")
    parser.add_argument("--left-sn", default=None, help="Left arm serial number override")
    parser.add_argument("--right-sn", default=None, help="Right arm serial number override")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without connecting to hardware")
    parser.add_argument("--sample", type=int, default=10, help="Number of evenly-spaced frames to show in dry run (default: 10)")
    args = parser.parse_args()

    path_str = args.episode or args.file or config.DATASET_FILE
    hdf5_path = resolve_hdf5_path(path_str)
    data = load_episode(hdf5_path)

    if args.freq is not None:
        dt = 1.0 / args.freq / args.speed_scale
    else:
        dt = compute_dt(data["timestamp"], 30.0) / args.speed_scale

    print(f"Replay dt: {dt*1000:.1f} ms ({1.0/dt:.1f} Hz)")

    if args.dry_run:
        dry_run(data, dt, args)
        return

    FlexivRobot, GripperController = _import_hardware()

    left_sn = args.left_sn or config.SN_MAP[config.LEFT_ARM_ID]
    right_sn = args.right_sn or config.SN_MAP[config.RIGHT_ARM_ID]

    print(f"Connecting left arm  [{left_sn}] ...")
    left_robot = FlexivRobot(left_sn)
    print(f"Connecting right arm [{right_sn}] ...")
    right_robot = FlexivRobot(right_sn)

    left_g, right_g = None, None
    if not args.no_gripper:
        left_g, right_g = init_grippers(GripperController)

    input("\nPress ENTER to go home and start replay ...")

    try:
        go_home_both(left_robot, right_robot, args.home_plan)

        if args.mode == "joint":
            replay_joint(left_robot, right_robot, left_g, right_g,
                         data, dt, args.max_vel, args.max_acc)
        else:
            replay_cartesian(left_robot, right_robot, left_g, right_g,
                             data, dt)

        print("Replay finished.")

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


if __name__ == "__main__":
    main()
