import cv2
import logging
import numpy as np
import os
from paddlex import create_pipeline
import config

logger = logging.getLogger(__name__)

class PerceptionProcessor:
    def __init__(self, index_dir):
        # 1. 检查索引目录是否存在
        if not os.path.exists(index_dir):
            logger.warning(f"⚠️ [警告] 人脸索引目录不存在: {index_dir}，人脸识别将无法工作！")
        
        print(f"  [Perception] 加载目标检测: {config.DET_MODEL_NAME}...")
        self.det_pipeline = create_pipeline(pipeline="object_detection", device=config.PADDLE_DEVICE)
        
        print(f"  [Perception] 加载人脸识别: Face Rec...")
        # 注意：PaddleX 的 index 参数通常需要是构建好的索引库路径，而不仅仅是图片文件夹
        self.face_pipeline = create_pipeline(pipeline="face_recognition", device=config.PADDLE_DEVICE)
        
        self.index_dir = index_dir
        self.det_threshold = 0.4
        self.face_threshold = 0.45 
        print("  [Perception] 模块就绪!")

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

        print(f"    🔍 [视觉] 发现 {len(person_boxes)} 个目标，正在核验身份...", end=" ")
        
        # 2. 人脸二次确认
        h, w, _ = frame_rgb.shape
        for i, person in enumerate(person_boxes):
            px1, py1, px2, py2 = person['box']
            
            # 扩大裁剪范围，防止人脸在边缘被切掉
            pad = 30 
            roi_x1, roi_y1 = max(0, px1-pad), max(0, py1-pad)
            roi_x2, roi_y2 = min(w, px2+pad), min(h, py2+pad)
            
            person_roi = frame_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # 如果截取区域太小，跳过
            if person_roi.size == 0 or person_roi.shape[0] < 20 or person_roi.shape[1] < 20: 
                print(f"[目标{i}: 图像过小]", end=" ")
                continue

            try:
                # 关键：这里去掉了 pass，把 index 传进去
                face_output = self.face_pipeline.predict(person_roi, index=self.index_dir)
                
                found_face = False
                for res in face_output:
                    res_data = res.json if hasattr(res, 'json') else {}
                    if not res_data: continue
                    
                    # PaddleX 返回结构可能比较深，这里做保护
                    res_inner = res_data.get('res', {})
                    if not res_inner: continue

                    f_boxes = res_inner.get('boxes', [])
                    if not f_boxes: continue
                    
                    # 找置信度最高的人脸
                    best_face = max(f_boxes, key=lambda x: (x.get('rec_scores') or [0])[0])
                    rec_scores = best_face.get('rec_scores')
                    
                    if rec_scores and rec_scores[0] > self.face_threshold:
                        labels = best_face.get('labels')
                        if labels:
                            name = labels[0]
                            # 计算人脸在大图中的绝对坐标
                            fx = [int(c) for c in best_face['coordinate']]
                            abs_face_box = [roi_x1 + fx[0], roi_y1 + fx[1], roi_x1 + fx[2], roi_y1 + fx[3]]
                            
                            print(f"\n      ✅ 目标{i} 身份确认: {name} ({rec_scores[0]:.2f})", end="")
                            person['name'] = name
                            person['face_box'] = abs_face_box
                            found_face = True
                            break 
                
                if not found_face:
                    print(f"\n      👤 目标{i} 未识别身份 (无人脸或置信度低)", end="")

            except Exception as e:
                # ！！！这里是关键修改！！！
                print(f"\n      ❌ 目标{i} 识别出错: {str(e)}", end="")
                # 如果是路径问题，在这里能看出来

            detections.append(person)
        
        print("") # 换行
        return detections