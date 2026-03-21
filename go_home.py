# go_home.py

import config
from robot.flexiv import FlexivRobot


def select_robot():
    """
    Select one robot by logical ID.
    """
    print("Available robots:")
    for rid, sn in config.SN_MAP.items():
        print(f"  {rid} -> {sn}")

    while True:
        robot_id = input("Select robot ID (A/B/C/D): ").strip().upper()
        if robot_id in config.SN_MAP:
            return robot_id, config.SN_MAP[robot_id]
        print("Invalid robot ID, please try again.")


def initialize_robot(robot_sn):
    """
    Initialize robot connection by serial number.
    """
    try:
        robot = FlexivRobot(robot_sn)
        return robot
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        return None


def move_to_home(robot):
    """
    Move robot to home using the configured tablet plan.
    """
    try:
        print("Executing plan: ReturnNewHome ...")
        robot.go_home_by_plan(plan_name="ReturnNewHome", timeout=120.0)
        print("ReturnNewHome finished.")
        return True
    except Exception as e:
        print(f"Failed to execute ReturnNewHome: {e}")
        return False


def main():
    print("--------------------------- Go Home ---------------------------")

    robot_id, robot_sn = select_robot()
    print(f"Selected robot: ID={robot_id}, SN={robot_sn}")

    robot = initialize_robot(robot_sn)
    if robot is None:
        print("Failed to initialize robot, exiting...")
        return 1

    if not move_to_home(robot):
        print("Failed to move robot to home, exiting...")
        return 1

    print(f"Rizon4s ID: {robot_id}, SN: {robot_sn} successfully returned home")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())