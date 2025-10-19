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
        box = det.get('box')
        if box is None: continue
        name = det.get('name', 'Unknown')
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(debug_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
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
                # 无缝开启下一个事件
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
        BINGO! 现在为每一帧都绘制调试信息。
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
            # 1. 对当前帧绘制调试信息
            frame_with_debug_info = draw_debug_info_for_event_frame(data["frame"], data["detections"])

            # 2. 保存带调试信息的帧
            debug_frame_path = str(event_dir / f"frame_{i+1:03d}.jpg")
            cv2.imwrite(debug_frame_path, frame_with_debug_info)
            
            # 3. 将带调试信息的帧路径存入打包数据
            packaged_frames.append({
                "image_path": debug_frame_path,
                "detections": data["detections"],
                "timestamp": data["timestamp"]
            })
            
            # 4. 使用第一帧作为封面图 (preview.jpg)
            if i == 0:
                preview_image_path = str(event_dir / "preview.jpg")
                cv2.imwrite(preview_image_path, frame_with_debug_info)
                
        logger.info(f"打包 {len(packaged_frames)} 帧带调试信息的图像到事件 {event_timestamp}")
        return {
            "event_id": event_timestamp,
            "frames": packaged_frames,
            "start_time": processing_buffer[0]["timestamp"],
            "end_time": processing_buffer[-1]["timestamp"],
            "preview_image_path": preview_image_path
        }