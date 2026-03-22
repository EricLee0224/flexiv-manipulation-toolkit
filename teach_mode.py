"""Put one or both Flexiv arms into teach (kinesthetic guidance) mode."""

import argparse
import signal
import sys

from robot.flexiv_arm import FlexivArm
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Enable teach mode on Flexiv arms.")
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to enable teach mode on (default: both)",
    )
    parser.add_argument(
        "--left-sn",
        default=config.SN_MAP[config.LEFT_ARM_ID],
        help=f"Left arm serial number (default: {config.SN_MAP[config.LEFT_ARM_ID]})",
    )
    parser.add_argument(
        "--right-sn",
        default=config.SN_MAP[config.RIGHT_ARM_ID],
        help=f"Right arm serial number (default: {config.SN_MAP[config.RIGHT_ARM_ID]})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    arms: list[FlexivArm] = []

    if args.arm in ("left", "both"):
        print(f"[left] Initializing ({args.left_sn}) ...")
        arm_left = FlexivArm(args.left_sn)
        arms.append(arm_left)

    if args.arm in ("right", "both"):
        print(f"[right] Initializing ({args.right_sn}) ...")
        arm_right = FlexivArm(args.right_sn)
        arms.append(arm_right)

    def shutdown():
        print("\nDisabling teach mode ...")
        for arm in arms:
            arm.disable_teach()
        print("Done.")

    def on_signal(sig, _frame):
        shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for arm in arms:
        arm.enable_teach()

    print("")
    print("=== Teach mode active ===")
    print("You can now move the arm(s) freely by hand.")
    print("Press Enter or Ctrl-C to exit.")
    print("")

    try:
        input()
    except EOFError:
        pass

    shutdown()


if __name__ == "__main__":
    main()
