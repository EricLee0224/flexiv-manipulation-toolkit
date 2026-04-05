import gripper_ctrl
import time

if __name__ == "__main__":
    mqtt_broker_ip = "192.168.20.2"
    mqtt_broker_port = 1883
    controller = gripper_ctrl.GripperController(gripper_name="gripper2", broker=mqtt_broker_ip, port=mqtt_broker_port)
    controller.start()
    try:
        while True:
            feedback = controller.get_feedback()
            print(f"{controller.name()}: position={feedback.position}, velocity={feedback.velocity}, torque={feedback.torque}, seq={feedback.seq}, timestamp_sec={feedback.timestamp_sec}, timestamp_nsec={feedback.timestamp_nsec}")
            controller.try_control(position=1.0, velocity=0.0, torque=0.0)
            time.sleep(1.0)
    except KeyboardInterrupt:
        controller.stop()