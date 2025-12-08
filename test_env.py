import cv2
import time
import logging
import os
import numpy as np
from paddlex import create_pipeline

# --- 环境配置 ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print("\n=============================================")
    print("   PaddleX 实时摄像头推理 (修复版)")
    print("   (按 'q' 键退出)")
    print("=============================================\n")

    # 1. 加载模型
    logging.info("⏳ 加载目标检测 (PicoDet)...")
    det_pipeline = create_pipeline(pipeline="object_detection", device="cpu")
    
    logging.info("⏳ 加载人脸识别 (Face Rec)...")
    face_pipeline = create_pipeline(pipeline="face_recognition", device="cpu")
    
    index_dir = "face_index"

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("❌ 无法打开摄像头")
        return

    logging.info("✅ 开始推理...")
    
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret: break

        # --- A. 目标检测 (蓝色框) ---
        # 传入 numpy 数组 (Opencv Frame)
        det_output = det_pipeline.predict(frame, threshold=0.35)
        
        for res in det_output:
            # !!! 关键修正点：先取 .json，再取 ['res']['boxes'] !!!
            # 文档格式: {'res': {'boxes': [{'label':..., 'coordinate':...}]}}
            res_data = res.json
            boxes = res_data.get('res', {}).get('boxes', [])
            
            for box in boxes:
                # 过滤掉非人类 (PicoDet可能会检出 bottle, chair 等)
                if box.get('label') == 'person':
                    coord = [int(c) for c in box['coordinate']]
                    score = box.get('score')
                    
                    # 画框 (BGR: 蓝色)
                    cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (255, 0, 0), 2)
                    cv2.putText(frame, f"Person {score:.2f}", (coord[0], coord[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --- B. 人脸识别 (绿色框) ---
        try:
            face_output = face_pipeline.predict(frame, index=index_dir)
            for res in face_output:
                res_data = res.json
                # 人脸产线的结构类似，也是在 res['res']['boxes'] 里
                boxes = res_data.get('res', {}).get('boxes', [])
                
                for box in boxes:
                    coord = [int(c) for c in box['coordinate']]
                    # 人脸产线返回的 score 字段可能叫 rec_scores (列表) 或 score
                    # 根据你的日志，它是 rec_scores
                    rec_scores = box.get('rec_scores')
                    score = rec_scores[0] if rec_scores else 0.0
                    
                    labels = box.get('labels')
                    name = labels[0] if labels else "Unknown"
                    
                    if score > 0.4:
                        # 画框 (BGR: 绿色)
                        cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} {score:.2f}", (coord[0], coord[1]-30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception:
            pass

        # 显示 FPS
        fps = 1 / (time.time() - t_start)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('PaddleX Corrected Viz', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()