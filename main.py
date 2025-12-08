import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import time
import logging
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cv2
import config

# 日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def main():
    print("\n=== HearthScribe 空间指挥舱启动 ===\n")

    # 1. 初始化模块
    try:
        from src.perception.perception_processor import PerceptionProcessor
        perception = PerceptionProcessor(index_dir=config.FACE_INDEX_DIR)
        
        from src.memory.memory_stream import MemoryStream
        memory_stream = MemoryStream(config.IMAGE_STORAGE_PATH)
        
        from src.memory.long_term_memory import LongTermMemory
        print("  [Init] 正在连接记忆库...")
        ltm = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        
        from src.cognition.cognitive_core import CognitiveCore
        cognition = CognitiveCore()
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 2. 摄像头
    cap = cv2.VideoCapture(config.SOURCE_VIDEO)
    if not cap.isOpened():
        print("❌ 摄像头故障")
        return
    print(f"\n✅ 摄像头就绪 | 采样策略: 每 {config.PROCESS_INTERVAL} 秒检测一次")

    executor = ThreadPoolExecutor(max_workers=1)
    frame_count = 0
    
    # 假设 FPS=30
    SKIP_FRAMES = int(30 * config.PROCESS_INTERVAL)
    if SKIP_FRAMES < 1: SKIP_FRAMES = 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.5)
                continue
            
            if frame_count % SKIP_FRAMES == 0:
                current_time_str = datetime.now().strftime("%H:%M:%S")
                
                # A. 感知
                detections = perception.process_frame(frame)
                
                # B. 反馈状态
                if not detections:
                    print(f"[{current_time_str}] 💤 空间闲置中...", end='\r')
                
                # C. 记忆流处理
                event_pack = memory_stream.update(frame, detections)
                
                # D. 事件打包 -> 后台分析
                if event_pack:
                    duration = event_pack['end_time'] - event_pack['start_time']
                    print(f"\n📦 [{current_time_str}] 生成事件片段 ({duration:.1f}s) -> 提交大脑分析")
                    executor.submit(bg_analyze, event_pack, cognition, ltm)

            frame_count += 1
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 系统停止")
    finally:
        cap.release()
        executor.shutdown(wait=False)

def bg_analyze(event, cognition, ltm):
    """后台分析线程：负责连接认知与记忆"""
    try:
        # 1. 调用认知核心 (返回包含 score/label 的字典)
        result = cognition.analyze_event(event)
        
        if result:
            # 2. 存入长期记忆 (传入新字段)
            success = ltm.save_event(
                event_data=event, 
                summary=result['summary'], 
                kg_data=result['kg_data'],
                # 关键修改：传递新字段
                scene_label=result.get('scene_label'),
                interaction_score=result.get('interaction_score')
            )
            
            if success:
                print(f"💾 [入库] 场景:{result.get('scene_label')} | 评分:{result.get('interaction_score')} | 摘要:{result['summary'][:20]}...")
            else:
                print("❌ [入库] 数据库写入失败")
                
    except Exception as e:
        print(f"❌ [后台异常] {e}")

if __name__ == "__main__":
    main()