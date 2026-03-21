#!/usr/bin/env python3
import os
import cv2
import sys
import time
import queue
import argparse
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from cyber.python.cyber_py3 import cyber
from cyber.python.cyber_py3 import record

sys.path.append("/apollo/bazel-bin")
from modules.ts_common_msgs.proto_sensor_msgs.Image_pb2 import Image

# YUV 配置
YUV_WIDTH = 1920
YUV_HEIGHT = 1536
YUV_FRAME_SIZE = YUV_WIDTH * YUV_HEIGHT * 3 // 2  # NV12
DISPLAY_SIZE = (960, 768)  # 显示窗口大小


class YUVViewer:
    """YUV图像查看器"""
    
    def __init__(self, camera_name="camera"):
        self.camera_name = camera_name
        self.frame_queue = queue.Queue(maxsize=30)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.frame_count = 0
        self.last_timestamp = 0
    
    def yuv_to_bgr(self, yuv_data):
        """将YUV NV12格式转换为BGR"""
        try:
            yuv = np.frombuffer(yuv_data, dtype=np.uint8)
            if yuv.size != YUV_FRAME_SIZE:
                print(f"[ERROR] 无效的帧大小: 期望 {YUV_FRAME_SIZE}, 实际 {yuv.size}")
                return None
            yuv = yuv.reshape((YUV_HEIGHT * 3 // 2, YUV_WIDTH))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            return bgr
        except Exception as e:
            print(f"[ERROR] YUV转换失败: {e}")
            return None
    
    def process_message(self, msg: Image):
        """处理YUV消息"""
        try:
            bgr_img = self.yuv_to_bgr(msg.data)
            if bgr_img is None:
                return
            
            # 提取时间戳
            timestamp_ms = msg.header.stamp.sec * 1000 + msg.header.stamp.nsec // (1000 * 1000)
            seq = msg.header.seq
            
            # 添加信息文字
            display_img = bgr_img.copy()
            cv2.putText(display_img, f"Frame: {seq}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_img, f"Size: {bgr_img.shape[1]}x{bgr_img.shape[0]}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_img, f"TS: {timestamp_ms}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # 计算帧率
            if self.last_timestamp > 0:
                fps = 1000.0 / max(1, timestamp_ms - self.last_timestamp)
                cv2.putText(display_img, f"FPS: {fps:.1f}", (10, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            self.last_timestamp = timestamp_ms
            
            # 放入队列
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put((display_img, timestamp_ms))
            self.frame_count += 1
            
        except Exception as e:
            print(f"[ERROR] 处理消息失败: {e}")
    
    def callback(self, msg: Image):
        """实时订阅回调"""
        self.executor.submit(self.process_message, msg)
    
    def display_loop(self):
        """显示循环"""
        window_name = f"{self.camera_name} - YUV Camera Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, DISPLAY_SIZE[0], DISPLAY_SIZE[1])
        
        print(f"\n{'='*60}")
        print(f"  {window_name}")
        print(f"{'='*60}")
        print("  [ESC/Q] - 退出")
        print("  [SPACE] - 暂停/继续")
        print(f"{'='*60}\n")
        
        paused = False
        current_frame = None
        current_ts = None
        
        while self.running:
            try:
                if not paused:
                    # 获取最新帧
                    if not self.frame_queue.empty():
                        current_frame, current_ts = self.frame_queue.get()
                
                if current_frame is not None:
                    # 显示图像
                    display_img = cv2.resize(current_frame, DISPLAY_SIZE)
                    if paused:
                        cv2.putText(display_img, "PAUSED", (DISPLAY_SIZE[0]//2 - 100, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.imshow(window_name, display_img)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("\n[INFO] 用户退出")
                    break
                elif key == ord(' '):  # Space
                    paused = not paused
                    status = "暂停" if paused else "继续"
                    print(f"[INFO] {status}")
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"[ERROR] 显示循环错误: {e}")
                time.sleep(0.1)
        
        cv2.destroyWindow(window_name)
        self.running = False
    
    def shutdown(self):
        """关闭"""
        self.running = False
        self.executor.shutdown(wait=True)


def realtime_mode(topic: str, camera_name: str):
    """实时订阅模式"""
    print(f"\n{'='*60}")
    print(f"  实时订阅模式")
    print(f"{'='*60}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")
    
    cyber.init()
    node = cyber.Node(f"yuv_viewer_{camera_name}")
    
    viewer = YUVViewer(camera_name=camera_name)
    viewer.running = True
    
    # 订阅topic
    node.create_reader(topic, Image, viewer.callback)
    print(f"[INFO] 已订阅: {topic}")
    
    # 启动显示线程
    display_thread = threading.Thread(target=viewer.display_loop, daemon=True)
    display_thread.start()
    
    try:
        # 主循环
        while not cyber.is_shutdown() and viewer.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号")
    finally:
        viewer.shutdown()
        cyber.shutdown()
        print(f"[INFO] 共处理 {viewer.frame_count} 帧")
        print("[INFO] 程序结束")


def offline_mode(record_file: str, topic: str, camera_name: str, playback_speed: float = 1.0):
    """离线解析模式"""
    print(f"\n{'='*60}")
    print(f"  离线解析模式")
    print(f"{'='*60}")
    print(f"  Record: {record_file}")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(record_file):
        print(f"[ERROR] Record文件不存在: {record_file}")
        return
    
    cyber.init()
    
    viewer = YUVViewer(camera_name=camera_name)
    viewer.running = True
    
    # 启动显示线程
    display_thread = threading.Thread(target=viewer.display_loop, daemon=True)
    display_thread.start()
    
    try:
        # 读取record文件
        reader = record.RecordReader(record_file)
        print(f"[INFO] 开始读取record文件...")
        
        last_msg_time = None
        msg_count = 0
        
        for channel_name, msg, datatype, timestamp in reader.read_messages():
            if not viewer.running:
                break
            
            # 过滤topic
            if channel_name == topic:
                msg_count += 1
                
                # 解析消息
                image_msg = Image()
                image_msg.ParseFromString(msg)
                viewer.process_message(image_msg)
                
                # 控制播放速度
                if last_msg_time is not None and playback_speed > 0:
                    time_diff = (timestamp - last_msg_time) / 1e9  # 转换为秒
                    sleep_time = time_diff / playback_speed
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 1.0))  # 最多睡眠1秒
                
                last_msg_time = timestamp
                
                if msg_count % 100 == 0:
                    print(f"[INFO] 已处理 {msg_count} 条消息...")
        
        print(f"\n[INFO] Record文件读取完成")
        print(f"[INFO] 共处理 {msg_count} 条消息")
        
        # 等待用户关闭窗口
        print("[INFO] 按任意键退出...")
        while viewer.running:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[INFO] 收到中断信号")
    except Exception as e:
        print(f"[ERROR] 离线解析失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        viewer.shutdown()
        cyber.shutdown()
        print(f"[INFO] 共处理 {viewer.frame_count} 帧")
        print("[INFO] 程序结束")


def main():
    parser = argparse.ArgumentParser(
        description="YUV相机可视化工具 - 支持实时订阅和离线解析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 实时订阅模式
  python3 yuv_camera_visualization.py -m realtime -t /camera_lf_chest_fisheye/image_color/yuv
  
  # 离线解析模式
  python3 yuv_camera_visualization.py -m offline -r /path/to/record.record -t /camera_lf_chest_fisheye/image_color/yuv
        """
    )
    
    parser.add_argument('-m', '--mode', 
                       choices=['realtime', 'offline'], 
                       required=True,
                       help='运行模式: realtime(实时订阅), offline(离线解析)')
    
    parser.add_argument('-t', '--topic',
                       help='要订阅或解析的topic名称')
    
    parser.add_argument('-r', '--record',
                       help='离线模式下的record文件路径')
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == 'realtime':
        if not args.topic:
            parser.error("实时模式需要指定 -t/--topic 参数")
        realtime_mode(args.topic, "camera")
    elif args.mode == 'offline':
        if not args.record or not args.topic:
            parser.error("离线模式需要指定 -r/--record 和 -t/--topic 参数")
        offline_mode(args.record, args.topic, "camera", 1.0)


if __name__ == "__main__":
    main()

