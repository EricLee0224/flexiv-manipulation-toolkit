# robot/flexiv.py

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import flexivrdk


class ModeMap:
    """Logical mode names mapped to Flexiv RDK v1.8 Mode enums."""

    idle = "IDLE"
    primitive = "NRT_PRIMITIVE_EXECUTION"
    plan = "NRT_PLAN_EXECUTION"
    joint_position = "NRT_JOINT_POSITION"
    joint_impedance = "NRT_JOINT_IMPEDANCE"
    cartesian_motion_force = "NRT_CARTESIAN_MOTION_FORCE"


class FlexivRobot:
    """
    Flexiv robot wrapper for RDK v1.8.
    """

    # Match C++ TeachModeController
    KX_TEACH = [60.0, 60.0, 60.0, 3.0, 3.0, 3.0]
    ZX_TEACH = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]

    def __init__(
        self,
        robot_sn: str,
        auto_clear_fault: bool = True,
        auto_enable: bool = True,
        auto_wait_operational: bool = True,
        operational_timeout: Optional[float] = None,
    ):
        self.robot_sn = robot_sn
        self.robot = None
        self.mode = flexivrdk.Mode

        self._connect()

        if auto_clear_fault:
            self.try_clear_fault()

        if auto_enable:
            self.enable()

        if auto_wait_operational:
            self.wait_until_operational(timeout=operational_timeout)

    # ==========================================================================
    # Initialization / connection
    # ==========================================================================

    def _connect(self) -> None:
        print(f"Connecting to robot [{self.robot_sn}] ...")
        self.robot = flexivrdk.Robot(self.robot_sn)
        print("Robot interface created.")

    def try_clear_fault(self) -> bool:
        if not self.robot.fault():
            return True

        print("Fault detected, trying to clear ...")
        ok = self.robot.ClearFault()
        if not ok:
            print("Fault cannot be cleared.")
            return False

        time.sleep(1.0)
        print("Fault cleared.")
        return True

    def clear_fault(self) -> None:
        if not self.try_clear_fault():
            raise RuntimeError("Failed to clear robot fault")

    def enable(self) -> None:
        print("Enabling robot ...")
        self.robot.Enable()

    def wait_until_operational(
        self,
        timeout: Optional[float] = None,
        period: float = 1.0,
    ) -> None:
        start = time.time()
        while not self.robot.operational():
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Timed out waiting for robot to become operational")
            time.sleep(period)

        print("Robot is now operational.")

    # ==========================================================================
    # Robot status / info
    # ==========================================================================

    def is_fault(self) -> bool:
        return self.robot.fault()

    def is_operational(self) -> bool:
        return self.robot.operational()

    def is_busy(self) -> bool:
        return self.robot.busy()

    def is_recovery_state(self) -> bool:
        return self.robot.recovery()

    def run_auto_recovery(self) -> None:
        self.robot.RunAutoRecovery()

    def info(self):
        return self.robot.info()

    def digital_inputs(self):
        return self.robot.digital_inputs()

    # ==========================================================================
    # State access
    # ==========================================================================

    def state_obj(self):
        return self.robot.states()

    def get_state(self) -> Dict[str, np.ndarray]:
        s = self.robot.states()
        return {
            "q": np.array(s.q, dtype=np.float64),
            "dq": np.array(s.dq, dtype=np.float64),
            "tcp_pose": np.array(s.tcp_pose, dtype=np.float64),
            "tcp_vel": np.array(s.tcp_vel, dtype=np.float64),
        }

    def get_full_state(self) -> Dict[str, np.ndarray]:
        s = self.robot.states()
        return {
            "q": np.array(s.q, dtype=np.float64),
            "theta": np.array(s.theta, dtype=np.float64),
            "dq": np.array(s.dq, dtype=np.float64),
            "dtheta": np.array(s.dtheta, dtype=np.float64),
            "tau": np.array(s.tau, dtype=np.float64),
            "tau_des": np.array(s.tau_des, dtype=np.float64),
            "tau_dot": np.array(s.tau_dot, dtype=np.float64),
            "tau_ext": np.array(s.tau_ext, dtype=np.float64),
            "tcp_pose": np.array(s.tcp_pose, dtype=np.float64),
            "tcp_vel": np.array(s.tcp_vel, dtype=np.float64),
            "flange_pose": np.array(s.flange_pose, dtype=np.float64),
            "ft_sensor_raw": np.array(s.ft_sensor_raw, dtype=np.float64),
            "ext_wrench_in_tcp": np.array(s.ext_wrench_in_tcp, dtype=np.float64),
            "ext_wrench_in_world": np.array(s.ext_wrench_in_world, dtype=np.float64),
            "ext_wrench_in_tcp_raw": np.array(
                s.ext_wrench_in_tcp_raw, dtype=np.float64
            ),
            "ext_wrench_in_world_raw": np.array(
                s.ext_wrench_in_world_raw, dtype=np.float64
            ),
        }

    def get_joint_pos(self) -> np.ndarray:
        return np.array(self.robot.states().q, dtype=np.float64)

    def get_joint_vel(self) -> np.ndarray:
        return np.array(self.robot.states().dq, dtype=np.float64)

    def get_joint_torque(self) -> np.ndarray:
        return np.array(self.robot.states().tau, dtype=np.float64)

    def get_tcp_pose(self) -> np.ndarray:
        return np.array(self.robot.states().tcp_pose, dtype=np.float64)

    def get_tcp_vel(self) -> np.ndarray:
        return np.array(self.robot.states().tcp_vel, dtype=np.float64)

    def get_ext_wrench_in_tcp(self) -> np.ndarray:
        return np.array(self.robot.states().ext_wrench_in_tcp, dtype=np.float64)

    def get_ext_wrench_in_world(self) -> np.ndarray:
        return np.array(self.robot.states().ext_wrench_in_world, dtype=np.float64)

    # ==========================================================================
    # Mode switching
    # ==========================================================================

    def mode_value(self, mode_name: str):
        if not hasattr(ModeMap, mode_name):
            raise ValueError(f"Unknown mode name: {mode_name}")
        mode_attr_name = getattr(ModeMap, mode_name)
        return getattr(self.mode, mode_attr_name)

    def switch_mode(self, mode_name: str) -> None:
        target_mode = self.mode_value(mode_name)
        print(f"Switching mode to [{mode_name}] ...")
        self.robot.SwitchMode(target_mode)

    def to_idle(self) -> None:
        self.switch_mode("idle")

    def to_primitive_mode(self) -> None:
        self.switch_mode("primitive")

    def to_plan_mode(self) -> None:
        self.switch_mode("plan")

    def to_joint_position_mode(self) -> None:
        self.switch_mode("joint_position")

    def to_joint_impedance_mode(self) -> None:
        self.switch_mode("joint_impedance")

    def to_cartesian_motion_force_mode(self) -> None:
        self.switch_mode("cartesian_motion_force")

    # ==========================================================================
    # Teach mode
    # ==========================================================================

    def prepare_teach_mode(self, timeout: float = 30.0) -> None:
        """
        Match C++ Start() pre-start sequence:
        1) switch to primitive mode
        2) run ZeroFTSensor
        3) small wait
        4) Stop()
        """
        print("Preparing teach mode: zeroing F/T sensor ...")
        self.zero_ft_sensor(timeout=timeout)
        time.sleep(0.3)
        self.stop()

    def switch_to_teach_mode(self) -> None:
        """
        Match C++ TeachModeController::SwitchToTeachMode().
        """
        print("Switching to teach mode [NRT_CARTESIAN_MOTION_FORCE] ...")
        self.to_cartesian_motion_force_mode()

        self.set_cartesian_impedance(self.KX_TEACH, self.ZX_TEACH)
        self.set_force_control_axis([False, False, False, False, False, False])
        self.set_null_space_objectives(
            linear_manipulability=0.0,
            angular_manipulability=0.0,
            ref_positions_tracking=0.3,
        )

    def enable_teach(self) -> None:
        """
        Enter teach mode. Keep-alive commands are handled by FlexivArm.
        """
        self.prepare_teach_mode()
        self.switch_to_teach_mode()

    def disable_teach(self) -> None:
        """
        Match C++ disable logic: Stop() to return to IDLE.
        """
        self.stop()

    # ==========================================================================
    # Primitive execution
    # ==========================================================================

    def execute_primitive(
        self,
        primitive_name: str,
        params: Optional[Dict[str, Any]] = None,
        switch_mode: bool = True,
    ) -> None:
        if params is None:
            params = {}

        if switch_mode:
            self.to_primitive_mode()

        self.robot.ExecutePrimitive(primitive_name, params)

    def primitive_states(self) -> Dict[str, Any]:
        return self.robot.primitive_states()

    def wait_for_primitive_state(
        self,
        key: str,
        target_value: Any = True,
        timeout: Optional[float] = None,
        period: float = 0.05,
    ) -> None:
        start = time.time()
        while True:
            states = self.robot.primitive_states()
            if key in states and states[key] == target_value:
                return

            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Timed out waiting for primitive state [{key}] == {target_value}"
                )

            time.sleep(period)

    def go_home_by_primitive(self, timeout: Optional[float] = None) -> None:
        self.execute_primitive("Home", {}, switch_mode=True)
        self.wait_for_primitive_state("reachedTarget", True, timeout=timeout)

    def zero_ft_sensor(self, timeout: Optional[float] = None) -> None:
        self.execute_primitive("ZeroFTSensor", {}, switch_mode=True)
        self.wait_for_primitive_state("terminated", True, timeout=timeout)

    # ==========================================================================
    # Plan execution
    # ==========================================================================

    def list_plans(self) -> List[str]:
        return list(self.robot.plan_list())

    def execute_plan(
        self,
        plan: Union[str, int],
        keep_running_after_disconnect: bool = True,
        switch_mode: bool = True,
    ) -> None:
        if switch_mode:
            self.to_plan_mode()

        self.robot.ExecutePlan(plan, keep_running_after_disconnect)

    def plan_info(self):
        return self.robot.plan_info()

    def wait_until_idle(
        self,
        timeout: Optional[float] = None,
        period: float = 0.1,
    ) -> None:
        start = time.time()
        while self.robot.busy():
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError("Timed out waiting for plan/task to finish")
            time.sleep(period)

    def go_home_by_plan(
        self,
        plan_name: str = "ReturnNewHome",
        timeout: Optional[float] = None,
        keep_running_after_disconnect: bool = True,
    ) -> None:
        self.execute_plan(
            plan_name,
            keep_running_after_disconnect=keep_running_after_disconnect,
            switch_mode=True,
        )
        self.wait_until_idle(timeout=timeout)

    # ==========================================================================
    # Non-realtime control
    # ==========================================================================

    def send_joint_position(
        self,
        target_pos: List[float],
        target_vel: List[float],
        max_vel: List[float],
        max_acc: List[float],
        switch_mode: bool = False,
    ) -> None:
        if switch_mode:
            self.to_joint_position_mode()

        self.robot.SendJointPosition(target_pos, target_vel, max_vel, max_acc)

    def send_cartesian_motion_force(
        self,
        target_pose: List[float],
        target_wrench: Optional[List[float]] = None,
        target_stiffness: Optional[List[float]] = None,
        max_linear_vel: Optional[float] = None,
        switch_mode: bool = False,
    ) -> None:
        if switch_mode:
            self.to_cartesian_motion_force_mode()

        if target_wrench is None and target_stiffness is None and max_linear_vel is None:
            self.robot.SendCartesianMotionForce(target_pose)
        elif target_stiffness is None and max_linear_vel is None:
            self.robot.SendCartesianMotionForce(target_pose, target_wrench)
        else:
            if target_wrench is None:
                target_wrench = [0.0] * 6
            if target_stiffness is None:
                target_stiffness = [0.0] * 6
            if max_linear_vel is None:
                max_linear_vel = 0.1

            self.robot.SendCartesianMotionForce(
                target_pose,
                target_wrench,
                target_stiffness,
                max_linear_vel,
            )

    def set_joint_impedance(self, Kq: List[float]) -> None:
        self.robot.SetJointImpedance(Kq)

    def set_cartesian_impedance(
        self,
        Kx: List[float],
        Zx: Optional[List[float]] = None,
    ) -> None:
        if Zx is None:
            self.robot.SetCartesianImpedance(Kx)
        else:
            self.robot.SetCartesianImpedance(Kx, Zx)

    def set_null_space_posture(self, q_ref: List[float]) -> None:
        self.robot.SetNullSpacePosture(q_ref)

    def set_null_space_objectives(
        self,
        linear_manipulability: float,
        angular_manipulability: float,
        ref_positions_tracking: float,
    ) -> None:
        self.robot.SetNullSpaceObjectives(
            linear_manipulability,
            angular_manipulability,
            ref_positions_tracking,
        )

    def set_max_contact_wrench(self, max_wrench: List[float]) -> None:
        self.robot.SetMaxContactWrench(max_wrench)

    def set_force_control_axis(self, axis_mask: List[bool]) -> None:
        self.robot.SetForceControlAxis(axis_mask)

    def set_force_control_frame(self, coord_type) -> None:
        self.robot.SetForceControlFrame(coord_type)

    # ==========================================================================
    # Misc
    # ==========================================================================

    def global_variables(self) -> Dict[str, Any]:
        return self.robot.global_variables()

    def set_global_variables(self, variables: Dict[str, Any]) -> None:
        self.robot.SetGlobalVariables(variables)

    def stop(self) -> None:
        print("Stopping robot ...")
        self.robot.Stop()