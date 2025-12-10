# src/perception/perception_processor.py
import cv2
import logging
import numpy as np
import os
from paddlex import create_pipeline
import config
from datetime import datetime  # 1. 引入时间模块

logger = logging.getLogger(__name__)

class PerceptionProcessor:
    def __init__(self, index_dir):
        print(f"  [Perception] 加载目标检测: {config.DET_MODEL_NAME}...")
        self.det_pipeline = create_pipeline(pipeline="object_detection", device="cpu")
        
        print(f"  [Perception] 加载人脸识别: Face Rec...")
        self.face_pipeline = create_pipeline(pipeline="face_recognition", device="cpu")
        
        self.index_dir = index_dir
        self.det_threshold = 0.4
        self.face_threshold = 0.45 
        
        index_file = os.path.join(index_dir, "vector.index")
        if os.path.exists(index_file):
            self.use_face_rec = True
            print(f"  ✅ [Perception] 人脸库加载成功: {index_dir}")
        else:
            self.use_face_rec = False
            print(f"  ⚠️ [Perception] 警告: 在 {index_dir} 未找到 vector.index，身份识别功能已禁用。")

    def process_frame(self, frame_bgr):
        detections = []
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. 目标检测
        try:
            det_output = self.det_pipeline.predict(frame_rgb, threshold=self.det_threshold)
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return []

        person_boxes = []
        
        for res in det_output:
            res_data = res.json if hasattr(res, 'json') else {}
            boxes = res_data.get('res', {}).get('boxes', [])
            for box in boxes:
                if box.get('label') == 'person':
                    coord = [int(c) for c in box['coordinate']]
                    person_boxes.append({
                        "box": coord,
                        "score": box.get('score'),
                        "name": "Unknown_Body",
                        "face_box": None
                    })

        if not person_boxes: return []

        # 2. 获取当前时间字符串 (时:分:秒)
        t_str = datetime.now().strftime("%H:%M:%S")

        print(f"    [{t_str}] 🔍 [视觉] 发现 {len(person_boxes)} 个目标...", end="")

        # 3. 人脸二次确认
        if self.use_face_rec:
            # print(" 核验中...", end="") # 删掉这个，保持输出简洁
            h, w, _ = frame_rgb.shape
            
            for i, person in enumerate(person_boxes):
                px1, py1, px2, py2 = person['box']
                pad = 30 
                roi_x1, roi_y1 = max(0, px1-pad), max(0, py1-pad)
                roi_x2, roi_y2 = min(w, px2+pad), min(h, py2+pad)
                person_roi = frame_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if person_roi.size == 0 or person_roi.shape[0] < 20 or person_roi.shape[1] < 20: continue

                try:
                    face_output = self.face_pipeline.predict(person_roi, index=self.index_dir)
                    
                    found_face_in_loop = False
                    for res in face_output:
                        res_data = res.json if hasattr(res, 'json') else {}
                        if not res_data: continue
                        f_boxes = res_data.get('res', {}).get('boxes', [])
                        if not f_boxes: continue
                        
                        best_face = max(f_boxes, key=lambda x: (x.get('rec_scores') or [0])[0])
                        rec_scores = best_face.get('rec_scores')
                        
                        if rec_scores and rec_scores[0] > self.face_threshold:
                            labels = best_face.get('labels')
                            if labels:
                                name = labels[0]
                                fx = [int(c) for c in best_face['coordinate']]
                                abs_face_box = [roi_x1 + fx[0], roi_y1 + fx[1], roi_x1 + fx[2], roi_y1 + fx[3]]
                                
                                person['name'] = name
                                person['face_box'] = abs_face_box
                                # 4. 打印识别结果带时间
                                print(f"\n      [{t_str}] ✅ 目标{i} 身份确认: {name} ({rec_scores[0]:.2f})", end="")
                                found_face_in_loop = True
                                break
                    
                    if not found_face_in_loop:
                        # 没识别到也打印一下，方便确认
                        print(f"\n      [{t_str}] 👤 目标{i} 未识别身份", end="")

                except Exception as e:
                    pass
        else:
            print(" (身份识别跳过)", end="")

        print("") # 换行
        for p in person_boxes: detections.append(p)
        return detections