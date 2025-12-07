import cv2
import logging
from paddlex import create_pipeline
import config

logger = logging.getLogger(__name__)

class PerceptionProcessor:
    def __init__(self, index_dir):
        logger.info("🚀 初始化 PaddleX 感知引擎...")
        
        # 目标检测
        logger.info(f"加载目标检测模型: {config.DET_MODEL_NAME}...")
        self.det_pipeline = create_pipeline(
            pipeline="object_detection", 
            device=config.PADDLE_DEVICE
        )
        
        # 人脸识别
        logger.info("加载人脸识别产线...")
        self.face_pipeline = create_pipeline(
            pipeline="face_recognition",
            device=config.PADDLE_DEVICE
        )
        self.index_dir = index_dir
        
        # 阈值设置：设低一点，方便调试
        self.det_threshold = 0.35 
        self.face_threshold = 0.4

    def process_frame(self, frame_bgr):
        detections = []
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Pipeline A: 目标检测 (寻找人体) ---
        try:
            # 预测时使用较低阈值，以便我们在日志里看到更多信息
            det_output = self.det_pipeline.predict(frame_rgb, threshold=self.det_threshold)
            
            # 用于日志显示的原始检测结果列表
            raw_detections_log = []

            for res in det_output:
                res_dict = res.json if hasattr(res, 'json') else res
                boxes = res_dict.get('boxes', [])
                
                for box in boxes:
                    label = box.get('label')
                    score = box.get('score')
                    # 把所有检测到的东西（不仅仅是人）都记录到日志里
                    raw_detections_log.append(f"{label}({score:.2f})")

                    # 只有 'person' 才会被放入系统的有效检测列表
                    if label == 'person':
                        coord = box.get('coordinate')
                        detections.append({
                            "type": "person",
                            "box": [int(c) for c in coord],
                            "score": score,
                            "name": "Unknown_Body"
                        })
            
            # !!! 核心修改：无论是否有人，都打印模型看到了什么 !!!
            if raw_detections_log:
                logger.info(f"🔍 [底层视觉] 原始检测: {', '.join(raw_detections_log)}")
            else:
                logger.info(f"🔍 [底层视觉] 画面空空如也 (阈值>{self.det_threshold})")

        except Exception as e:
            logger.warning(f"目标检测失败: {e}")

        # 如果没人，直接返回，不浪费算力跑人脸
        if not detections:
            return []

        # --- Pipeline B: 人脸识别 (确定身份) ---
        try:
            face_output = self.face_pipeline.predict(frame_rgb, index=self.index_dir)
            for res in face_output:
                res_dict = res.json if hasattr(res, 'json') else res
                boxes = res_dict.get('boxes', [])
                for box in boxes:
                    score = box['rec_scores'][0] if box.get('rec_scores') else 0
                    if score > self.face_threshold:
                        name = box['labels'][0]
                        logger.info(f"👤 [身份识别] 确认身份: {name} (置信度: {score:.2f})")
                        
                        # 更新 detections
                        detections.append({
                            "type": "face",
                            "box": [int(c) for c in box['coordinate']],
                            "score": score,
                            "name": name
                        })
                    else:
                        logger.info(f"👤 [身份识别] 发现人脸但置信度过低 ({score:.2f})")

        except Exception as e:
            logger.warning(f"人脸识别失败: {e}")

        return detections