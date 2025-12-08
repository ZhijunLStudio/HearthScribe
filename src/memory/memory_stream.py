# src/memory/memory_stream.py
import time
from collections import deque
import cv2
from pathlib import Path
from datetime import datetime
import logging
import config
import json

logger = logging.getLogger(__name__)

def draw_debug_info_for_event_frame(frame, detections):
    """
    绘制逻辑：
    1. 蓝色框: 身体 (Person Body)
    2. 绿色框: 已知身份的人脸 + 名字
    3. 红色框: 未知身份的人脸/身体
    """
    debug_frame = frame.copy()
    
    for det in detections:
        # 1. 画身体框 (Body Box)
        px1, py1, px2, py2 = map(int, det['box'])
        name = det.get('name', 'Unknown_Body')
        
        # 默认蓝色 (BGR: 255, 0, 0)
        body_color = (255, 0, 0) 
        
        # 画身体矩形
        cv2.rectangle(debug_frame, (px1, py1), (px2, py2), body_color, 2)
        
        # 2. 画人脸框 (Face Box) - 如果有的话
        face_box = det.get('face_box')
        if face_box:
            fx1, fy1, fx2, fy2 = map(int, face_box)
            # 已知身份用绿色，未知用红色
            face_color = (0, 255, 0) if name != "Unknown_Body" else (0, 0, 255)
            
            cv2.rectangle(debug_frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            
            # 标签背景
            label = name
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(debug_frame, (fx1, fy1 - label_size[1] - 10), (fx1 + label_size[0], fy1), face_color, -1)
            # 标签文字
            cv2.putText(debug_frame, label, (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            # 如果没人脸框，但在身体框上标注名字
            cv2.putText(debug_frame, name, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, body_color, 2)
            
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
        
        # 如果有人，或者正在录制中
        if detections:
            if not self.is_capturing:
                self.is_capturing = True
                self.buffer.clear()
                self.event_start_time = current_time
            
            self.last_person_seen_time = current_time
            
            # 采样
            if current_time - self.last_frame_capture_time >= config.FRAME_CAPTURE_INTERVAL:
                self.last_frame_capture_time = current_time
                self.buffer.append({
                    "frame": frame.copy(), "detections": detections, "timestamp": current_time
                })
            
            # 强制切分
            if current_time - self.event_start_time >= config.EVENT_MAX_DURATION_SECONDS:
                packaged = self.package_event()
                self.buffer.clear()
                self.event_start_time = current_time
                return packaged

        elif self.is_capturing:
            # 超时结束
            if current_time - self.last_person_seen_time > config.EVENT_INACTIVITY_TIMEOUT:
                self.is_capturing = False
                return self.package_event()
        
        return None

    def package_event(self):
        if not self.buffer: return None
        
        evt_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        evt_dir = self.storage_path / evt_id
        evt_dir.mkdir(exist_ok=True)

        frames_info = []
        preview_path = None
        
        # 寻找最佳预览图 (人脸最多的一帧)
        best_idx = 0
        max_faces = 0
        
        for i, data in enumerate(self.buffer):
            # 绘图
            viz_frame = draw_debug_info_for_event_frame(data["frame"], data["detections"])
            path = evt_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(path), viz_frame)
            
            frames_info.append({
                "image_path": str(path.resolve()), 
                "detections": data["detections"],
                "timestamp": data["timestamp"]
            })
            
            # 评分
            faces_count = sum(1 for d in data['detections'] if d.get('face_box'))
            if faces_count >= max_faces:
                max_faces = faces_count
                best_idx = i
                
        # 生成预览图
        if frames_info:
            preview_path = frames_info[best_idx]['image_path']

        return {
            "event_id": evt_id,
            "frames": frames_info,
            "start_time": self.buffer[0]["timestamp"],
            "end_time": self.buffer[-1]["timestamp"],
            "preview_image_path": preview_path
        }