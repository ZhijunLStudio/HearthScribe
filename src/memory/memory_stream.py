import time
from collections import deque
import cv2
from pathlib import Path
from datetime import datetime
import logging
import config

logger = logging.getLogger(__name__)

class MemoryStream:
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        self.is_capturing = False
        self.last_person_seen_time = 0
        self.last_frame_capture_time = 0
        self.buffer = deque()
        
        # BINGO! 新增状态变量，用于记录事件开始时间
        self.event_start_time = 0
        
        logger.info(
            f"短期记忆流已初始化 (混合驱动模式)。"
            f"不活动超时: {config.EVENT_INACTIVITY_TIMEOUT}s, "
            f"最大事件时长: {config.EVENT_MAX_DURATION_SECONDS}s"
        )

    def update(self, frame, detections):
        """主更新函数，管理事件的整个生命周期"""
        current_time = time.time()
        
        # --- 如果场景中有人 ---
        if detections:
            # 如果是新事件的开始
            if not self.is_capturing:
                self.is_capturing = True
                self.buffer.clear()
                self.last_frame_capture_time = 0
                # BINGO! 记录新事件的开始时间
                self.event_start_time = current_time
                logger.info(f"检测到活动，开始捕获新事件...")
            
            # 更新最后一次见到人的时间
            self.last_person_seen_time = current_time
            
            # 根据间隔决定是否保存帧
            if current_time - self.last_frame_capture_time >= config.FRAME_CAPTURE_INTERVAL:
                self.last_frame_capture_time = current_time
                self.buffer.append({
                    "frame": frame.copy(),
                    "detections": detections,
                    "timestamp": current_time
                })
                logger.info(f"事件进行中... 已捕获第 {len(self.buffer)} 帧。")
                
            # BINGO! 新增逻辑：检查事件是否超时
            if current_time - self.event_start_time >= config.EVENT_MAX_DURATION_SECONDS:
                logger.info(f"事件已达到 {config.EVENT_MAX_DURATION_SECONDS} 秒最大时长，强制结束并打包。")
                packaged_event = self.package_event()
                
                # 立即开始下一个事件片段的捕获
                self.is_capturing = True
                self.event_start_time = current_time
                # 将当前帧作为新事件的第一帧
                self.buffer.append({
                    "frame": frame.copy(),
                    "detections": detections,
                    "timestamp": current_time
                })
                logger.info("无缝开启下一个事件片段...")
                return packaged_event

        # --- 如果场景中无人，但事件正在捕获中 ---
        elif self.is_capturing:
            # 检查不活动时间是否已超过超时阈值
            if current_time - self.last_person_seen_time > config.EVENT_INACTIVITY_TIMEOUT:
                logger.info(f"检测到超过 {config.EVENT_INACTIVITY_TIMEOUT} 秒无活动，事件结束。")
                self.is_capturing = False
                # BINGO! 事件结束后重置开始时间
                self.event_start_time = 0
                return self.package_event()
        
        return None

    def package_event(self):
        """打包缓冲区中的帧为一个事件，并清空缓冲区"""
        if not self.buffer:
            logger.warning("缓冲区为空，无法打包事件。")
            return None
        
        event_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_dir = self.storage_path / event_timestamp
        event_dir.mkdir()

        # BINGO! 创建一个缓冲区副本进行处理，然后清空原缓冲区
        processing_buffer = list(self.buffer)
        self.buffer.clear()

        packaged_frames = []
        for i, data in enumerate(processing_buffer):
            path = str(event_dir / f"frame_{i+1:03d}.jpg")
            cv2.imwrite(path, data["frame"])
            packaged_frames.append({
                "image_path": path,
                "detections": data["detections"],
                "timestamp": data["timestamp"]
            })
            
        logger.info(f"成功为事件 {event_timestamp} 打包 {len(packaged_frames)} 帧图像。")
        return {
            "event_id": event_timestamp,
            "frames": packaged_frames,
            "start_time": processing_buffer[0]["timestamp"],
            "end_time": processing_buffer[-1]["timestamp"]
        }