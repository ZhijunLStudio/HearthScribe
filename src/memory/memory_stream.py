import time
from collections import deque
import cv2
from pathlib import Path
from datetime import datetime
import logging
import config

logger = logging.getLogger(__name__)

def draw_debug_info_for_event_frame(frame, detections):
    """一个独立的、用于生成事件帧和预览图的绘图函数"""
    debug_frame = frame.copy()
    for det in detections:
        person_box = det.get('box')
        face_box = det.get('face_box')
        name = det.get('name', 'Unknown')

        if person_box is not None:
            px1, py1, px2, py2 = map(int, person_box)
            person_color = (0, 128, 0) if name != "Unknown" else (0, 0, 128)
            cv2.rectangle(debug_frame, (px1, py1), (px2, py2), person_color, 1)

        if face_box is not None:
            fx1, fy1, fx2, fy2 = map(int, face_box)
            face_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(debug_frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            label = name
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(debug_frame, (fx1, fy1 - label_size[1] - 10), 
                          (fx1 + label_size[0], fy1 - 10), face_color, cv2.FILLED)
            cv2.putText(debug_frame, label, (fx1, fy1 - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        elif person_box is not None:
            px1, py1, _, _ = map(int, person_box)
            person_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.putText(debug_frame, name, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, person_color, 2)
            
    return debug_frame

class MemoryStream:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.is_capturing = False
        self.last_person_seen_time = 0
        self.last_frame_capture_time = 0
        self.buffer = deque()
        self.event_start_time = 0
        logger.info("MemoryStream initialized.")

    def update(self, frame, detections):
        # ... (这个方法保持不变) ...
        current_time = time.time()
        
        if detections:
            if not self.is_capturing:
                self.is_capturing = True
                self.buffer.clear()
                self.event_start_time = current_time
                logging.info("检测到活动，开始捕获新事件...")
            
            self.last_person_seen_time = current_time
            
            if current_time - self.last_frame_capture_time >= config.FRAME_CAPTURE_INTERVAL:
                self.last_frame_capture_time = current_time
                self.buffer.append({
                    "frame": frame.copy(), "detections": detections, "timestamp": current_time
                })
            
            if current_time - self.event_start_time >= config.EVENT_MAX_DURATION_SECONDS:
                logging.info("事件达到最大时长，强制打包。")
                packaged_event = self.package_event()
                self.is_capturing = True
                self.buffer.clear()
                self.event_start_time = current_time
                self.last_frame_capture_time = current_time
                self.buffer.append({
                    "frame": frame.copy(), "detections": detections, "timestamp": current_time
                })
                return packaged_event

        elif self.is_capturing:
            if current_time - self.last_person_seen_time > config.EVENT_INACTIVITY_TIMEOUT:
                logging.info("检测到无活动超时，事件结束。")
                self.is_capturing = False
                return self.package_event()
        
        return None

    def package_event(self):
        """
        打包缓冲区中的帧。
        BINGO! 所有保存的路径都将转换为绝对路径。
        """
        if not self.buffer:
            return None
        
        event_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_dir = self.storage_path / event_timestamp
        event_dir.mkdir()

        processing_buffer = list(self.buffer)
        self.buffer.clear()

        packaged_frames = []
        preview_image_path = None

        for i, data in enumerate(processing_buffer):
            frame_with_debug_info = draw_debug_info_for_event_frame(data["frame"], data["detections"])
            
            # 创建 Path 对象
            debug_frame_path_obj = event_dir / f"frame_{i+1:03d}.jpg"
            cv2.imwrite(str(debug_frame_path_obj), frame_with_debug_info)
            
            # 关键修复：将路径转换为绝对路径再存储
            absolute_frame_path = str(debug_frame_path_obj.resolve())
            
            packaged_frames.append({
                "image_path": absolute_frame_path, # 存储绝对路径
                "detections": data["detections"],
                "timestamp": data["timestamp"]
            })
            
            if i == 0:
                preview_image_path_obj = event_dir / "preview.jpg"
                cv2.imwrite(str(preview_image_path_obj), frame_with_debug_info)
                # 关键修复：同样转换为绝对路径
                preview_image_path = str(preview_image_path_obj.resolve())
                
        logger.info(f"打包 {len(packaged_frames)} 帧带调试信息的图像到事件 {event_timestamp}")
        return {
            "event_id": event_timestamp,
            "frames": packaged_frames,
            "start_time": processing_buffer[0]["timestamp"],
            "end_time": processing_buffer[-1]["timestamp"],
            "preview_image_path": preview_image_path # 现在这里也是绝对路径
        }