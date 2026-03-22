"""Move one or both Flexiv arms to home position via a tablet plan."""

import argparse
import sys

from robot.flexiv import FlexivRobot
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Move Flexiv arms to home position.")
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to move home (default: both)",
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
    parser.add_argument(
        "--plan",
        default="ReturnNewHome",
        help="Plan name to execute (default: ReturnNewHome)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for the plan to finish (default: 120)",
    )
    return parser.parse_args()


def go_home(robot: FlexivRobot, label: str, plan_name: str, timeout: float) -> bool:
    try:
        print(f"[{label}] Executing plan: {plan_name} ...")
        robot.go_home_by_plan(plan_name=plan_name, timeout=timeout)
        print(f"[{label}] {plan_name} finished.")
        return True
    except Exception as e:
        print(f"[{label}] Failed: {e}")
        return False


def main():
    args = parse_args()
    ok = True

    if args.arm in ("left", "both"):
        print(f"[left] Initializing ({args.left_sn}) ...")
        robot_left = FlexivRobot(args.left_sn)
        if not go_home(robot_left, "left", args.plan, args.timeout):
            ok = False

    if args.arm in ("right", "both"):
        print(f"[right] Initializing ({args.right_sn}) ...")
        robot_right = FlexivRobot(args.right_sn)
        if not go_home(robot_right, "right", args.plan, args.timeout):
            ok = False

    if ok:
        print("All arms returned home successfully.")
    else:
        print("Some arms failed to return home.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
