import time
import threading
from dataclasses import dataclass
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from google.protobuf.message import DecodeError
import TsWbd_pb2


@dataclass
class GripperFeedback:
    seq: int = -1
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    timestamp_sec: int = 0
    timestamp_nsec: int = 0


class GripperController:
    def __init__(self, gripper_name: int, broker: str = "127.0.0.1", port: int = 1883):
        self._gripper_name = "null"
        self._broker = broker
        self._port = port
        self._pub_topic = f"tars/motor/{gripper_name}"
        self._sub_topic = f"{self._pub_topic}_feedback"

        self._feedback = GripperFeedback()

        self._connected = False
        self._seq = 0
        self._state_lock = threading.Lock()

        self._client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def start(self):
        self._client.connect(self._broker, self._port, 60)
        self._client.loop_start()

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
        with self._state_lock:
            self._connected = False

    def name(self) -> str:
        return self._gripper_name
    
    def get_feedback(self) -> GripperFeedback:
        with self._state_lock:
            return self._feedback

    def try_control(
        self,
        position: float,
        name: str = "",
        velocity: float = 0.0,
        torque: float = 0.0,
    ) -> bool:
        if self._gripper_name == "null":
            return False
        target_name = self._gripper_name
        with self._state_lock:
            connected = self._connected
        if not connected:
            return False
        return self._send_target_once(target_name, float(position), float(velocity), float(torque))

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _build_body_command(
        self,
        joint_name: str,
        position: float,
        velocity: float,
        torque: float,
    ):
        body_commands = TsWbd_pb2.BodyCommands()

        header = getattr(body_commands, "header")
        seq = self._next_seq()
        setattr(header, "seq", seq)
        now_ns = time.time_ns()
        stamp = getattr(header, "stamp")
        setattr(stamp, "sec", int(now_ns // 1_000_000_000))
        setattr(stamp, "nsec", int(now_ns % 1_000_000_000))
        setattr(header, "frame_id", "gripper_controller")

        joints = getattr(body_commands, "joints")
        add_joint = getattr(joints, "add")
        joint_command = add_joint()
        setattr(joint_command, "joint_name", joint_name)
        setattr(joint_command, "seq", seq)
        setattr(joint_command, "timestamp_ns", int(now_ns))
        setattr(joint_command, "state_request", TsWbd_pb2.STATE_REQ_ENABLE)

        controls = getattr(joint_command, "controls")
        add_control = getattr(controls, "add")
        control = add_control()
        setattr(control, "position", float(position))
        setattr(control, "velocity", float(velocity))
        setattr(control, "torque", float(torque))
        return body_commands

    def _send_target_once(self, joint_name: str, position: float, velocity: float, torque: float) -> bool:
        try:
            body_commands = self._build_body_command(joint_name, position, velocity, torque)
            serialize = getattr(body_commands, "SerializeToString")
            payload = serialize()
            result = self._client.publish(self._pub_topic, payload, qos=0, retain=False)
            return result.rc == 0
        except Exception as e:
            print(f"[MQTT] 控制消息发送失败: {type(e).__name__}: {e}")
            return False

    @staticmethod
    def _payload_preview(payload: bytes, max_bytes: int = 24) -> str:
        return payload[:max_bytes].hex()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        with self._state_lock:
            self._connected = reason_code.value == 0
        client.subscribe(self._sub_topic)

    def _on_message(self, client, userdata, msg):
        payload = msg.payload or b""
        if not payload:
            return

        try:
            data = TsWbd_pb2.BodyFeedback()
            parse_from_string = getattr(data, "ParseFromString")
            parse_from_string(payload)
        except DecodeError as e:
            print(
                f"[MQTT] Protobuf 解析失败: {e}; topic={msg.topic}; "
                f"len={len(payload)}; preview_hex={self._payload_preview(payload)}"
            )
            return
        except Exception as e:
            print(
                f"[MQTT] 解析异常: {type(e).__name__}: {e}; topic={msg.topic}; "
                f"len={len(payload)}; preview_hex={self._payload_preview(payload)}"
            )
            return

        try:
            feedback = GripperFeedback()
            joints = list(getattr(data, "joints", []))
            if not joints:
                return

            latest_position = self._feedback.position
            latest_velocity = self._feedback.velocity
            latest_torque = self._feedback.torque
            header = getattr(data, "header", None)
            feedback.seq = int(getattr(header, "seq", -1))
            stamp = getattr(header, "stamp", None)
            feedback.timestamp_sec = int(getattr(stamp, "sec", 0))
            feedback.timestamp_nsec = int(getattr(stamp, "nsec", 0))
            for joint_feedback in joints:
                measurements = list(getattr(joint_feedback, "measurements", []))
                if not measurements:
                    continue
                latest = measurements[-1]
                feedback.position = float(getattr(latest, "position", latest_position))
                feedback.velocity = float(getattr(latest, "velocity", latest_velocity))
                feedback.torque = float(getattr(latest, "torque", latest_torque))
                self._gripper_name = str(getattr(joint_feedback, "joint_name", ""))
            with self._state_lock:
                self._feedback = feedback
        except Exception as e:
            print(f"[MQTT] 反馈处理失败: {type(e).__name__}: {e}")


