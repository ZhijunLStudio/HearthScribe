# src/perception/perception_processor.py (V2.0 - 终极修正版)

import face_recognition
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import logging

logger = logging.getLogger(__name__)

class PerceptionProcessor:
    def __init__(self, known_faces_dir: str):
        logger.info("正在初始化感知处理器...")
        self.yolo = YOLO("weights/yolov8n.pt")
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces(Path(known_faces_dir))
        # 粘性记忆：存储 track_id -> name 的映射
        self.tracked_identities = {}

    def _load_known_faces(self, faces_dir: Path):
        logger.info(f"正在从 {faces_dir} 加载已知人脸...")
        for person_dir in faces_dir.iterdir():
            if person_dir.is_dir():
                for img_path in person_dir.glob("*.jpg*"): # 兼容 .jpeg
                    try:
                        # 使用cv2加载以避免元数据和内存布局问题
                        image_bgr = cv2.imread(str(img_path))
                        if image_bgr is None:
                            logger.warning(f"无法加载图片: {img_path}")
                            continue
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                        
                        encodings = face_recognition.face_encodings(image_rgb)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_dir.name)
                    except Exception as e:
                        logger.error(f"加载并编码图片 {img_path} 失败: {e}")
        logger.info(f"加载完成: {len(self.known_face_names)} 张人脸, {len(set(self.known_face_names))} 位已知人物。")

    def process_frame(self, frame):
        """处理单帧图像，返回带身份的追踪结果"""
        
        # 1. YOLO 追踪
        results = self.yolo.track(frame, classes=0, persist=True, verbose=False)
        if results[0].boxes.id is None:
            return []
        
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        # 2. 机会主义人脸识别
        # 使用 cv2.cvtColor 来创建内存连续的RGB数组副本
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 3. 身份关联与更新
        current_frame_detections = []
        for box, track_id in zip(boxes, track_ids):
            if track_id not in self.tracked_identities:
                px1, py1, px2, py2 = box
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    top, right, bottom, left = face_location
                    if top > py1 and right < px2 and bottom < py2 and left > px1:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        if True in matches:
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                        
                        self.tracked_identities[track_id] = name
                        logger.info(f"新身份关联: Track ID {track_id} -> {name}")
                        break 
            
            current_frame_detections.append({
                "track_id": track_id,
                "box": box,
                "name": self.tracked_identities.get(track_id, "Unknown")
            })

        return current_frame_detections