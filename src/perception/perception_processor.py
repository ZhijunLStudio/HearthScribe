# src/perception/perception_processor.py
import face_recognition
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2
import logging

logger = logging.getLogger(__name__)

def is_box_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

class PerceptionProcessor:
    def __init__(self, known_faces_dir: str):
        # ... (初始化代码保持不变) ...
        logger.info("正在初始化感知处理器...")
        self.yolo = YOLO("yolov8n.pt")
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_known_faces(Path(known_faces_dir))
        self.tracked_identities = {}
        logger.info("感知处理器初始化完成。")

    def _load_known_faces(self, faces_dir: Path):
        # ... (加载人脸代码保持不变) ...
        logger.info(f"正在从 {faces_dir} 加载已知人脸...")
        for person_dir in faces_dir.iterdir():
            if person_dir.is_dir():
                for img_path in person_dir.glob("*.jpg*"):
                    try:
                        image = face_recognition.load_image_file(str(img_path))
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_dir.name)
                    except Exception as e:
                        logger.error(f"加载并编码图片 {img_path} 失败: {e}")
        logger.info(f"加载完成: {len(self.known_face_names)} 张人脸, {len(set(self.known_face_names))} 位已知人物。")


    def process_frame(self, frame):
        # 1. YOLO 追踪人体
        yolo_results = self.yolo.track(frame, classes=0, persist=True, verbose=False)
        if yolo_results[0].boxes.id is None:
            self.tracked_identities.clear()
            return []
        
        person_boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)

        # 2. 全图人脸定位
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        # 3. 关联人脸和人体
        unassigned_face_locations = list(all_face_locations)
        person_to_face_map = {} # track_id -> face_location
        for person_box, track_id in zip(person_boxes, track_ids):
            for i, face_loc in enumerate(unassigned_face_locations):
                face_box = (face_loc[3], face_loc[0], face_loc[1], face_loc[2]) # (x1, y1, x2, y2)
                if is_box_inside(face_box, person_box):
                    person_to_face_map[track_id] = face_loc
                    unassigned_face_locations.pop(i)
                    break 
        
        # 4. 对关联上的人脸进行批量编码
        faces_to_encode_locations = list(person_to_face_map.values())
        all_face_encodings = face_recognition.face_encodings(rgb_frame, faces_to_encode_locations) if faces_to_encode_locations else []
        face_loc_to_encoding_map = dict(zip(faces_to_encode_locations, all_face_encodings))

        # 5. 更新身份并准备输出
        current_frame_detections = []
        processed_track_ids = set()
        for person_box, track_id in zip(person_boxes, track_ids):
            processed_track_ids.add(track_id)
            name = self.tracked_identities.get(track_id, "Unknown")
            
            # BINGO! 初始化 face_box_to_draw 为 None
            face_box_to_draw = None

            if (name == "Unknown") and track_id in person_to_face_map:
                face_loc = person_to_face_map[track_id]
                face_encoding = face_loc_to_encoding_map.get(face_loc)
                
                if face_encoding is not None:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            logger.info(f"身份关联成功: Track ID {track_id} -> {name}")
                    self.tracked_identities[track_id] = name
            
            # BINGO! 无论是否识别成功，只要有关联的人脸框，就把它记录下来
            if track_id in person_to_face_map:
                face_loc = person_to_face_map[track_id]
                # 将 (top, right, bottom, left) 转换为 (x1, y1, x2, y2)
                face_box_to_draw = (face_loc[3], face_loc[0], face_loc[1], face_loc[2])
            
            current_frame_detections.append({
                "track_id": track_id,
                "box": person_box,          # 人体框
                "face_box": face_box_to_draw, # 人脸框 (可能为 None)
                "name": self.tracked_identities.get(track_id, "Unknown")
            })

        # 清理消失的ID
        disappeared_ids = set(self.tracked_identities.keys()) - processed_track_ids
        for old_id in disappeared_ids:
            del self.tracked_identities[old_id]
            
        return current_frame_detections