# main.py
import os
# 禁用一些可能导致冲突的并行库设置
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
    print("\n=== HearthScribe 空间指挥舱启动 (高性能版) ===\n")

    # 1. 初始化模块
    try:
        from src.perception.perception_processor import PerceptionProcessor
        # 加载感知模块
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
    
    # 设置摄像头缓冲区大小为1，保证读到的是最新帧（减少延迟）
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"\n✅ 摄像头就绪 | 策略: 基于时间戳，每 {config.PROCESS_INTERVAL} 秒检测一次")

    executor = ThreadPoolExecutor(max_workers=1)
    
    # --- 关键修改：使用时间戳控制频率 ---
    last_process_time = 0 
    
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # 只有当 (当前时间 - 上次检测时间) > 设定间隔 (2秒) 时，才检测
            if current_time - last_process_time >= config.PROCESS_INTERVAL:
                
                last_process_time = current_time # 更新时间戳
                current_time_str = datetime.now().strftime("%H:%M:%S")
                
                # --- 优化：缩小图片进行检测 (大幅提升速度) ---
                # 保持原图 frame 用于保存和显示，复制一个小图 small_frame 用于检测
                # 宽度缩放到 640，高度按比例
                h, w = frame.shape[:2]
                scale = 640 / w
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                
                # A. 感知 (传入小图，速度更快)
                # 注意：PerceptionProcessor 内部返回的坐标是基于小图的
                # 如果需要精确坐标画在原图上，需要把坐标 * (1/scale) 还原
                # 但对于目前的逻辑，只要检测到人就行，坐标略有偏差影响不大
                detections = perception.process_frame(small_frame)
                
                # B. 反馈状态
                if not detections:
                    print(f"[{current_time_str}] 💤 空间闲置中...", end='\r')
                else:
                    # 如果需要保存原图，这里还是传原图给 MemoryStream
                    # 注意：如果 detections 是基于小图的，MemoryStream 画框可能会偏小
                    # 简单修复：把 detections 里的 box 坐标还原
                    for det in detections:
                        if 'box' in det:
                            det['box'] = [int(c / scale) for c in det['box']]
                        if 'face_box' in det and det['face_box']:
                            det['face_box'] = [int(c / scale) for c in det['face_box']]

                    # C. 记忆流处理 (传入高清原图)
                    event_pack = memory_stream.update(frame, detections)
                    
                    # D. 事件打包 -> 后台分析
                    if event_pack:
                        duration = event_pack['end_time'] - event_pack['start_time']
                        print(f"\n📦 [{current_time_str}] 生成事件片段 ({duration:.1f}s) -> 提交大脑分析")
                        executor.submit(bg_analyze, event_pack, cognition, ltm)

            # 这里的 sleep 可以非常短，或者直接去掉，因为上面有 cap.read() 阻塞
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 系统停止")
    finally:
        cap.release()
        executor.shutdown(wait=False)

def bg_analyze(event, cognition, ltm):
    """后台分析线程"""
    try:
        result = cognition.analyze_event(event)
        if result:
            success = ltm.save_event(
                event_data=event, 
                summary=result['summary'], 
                kg_data=result['kg_data'],
                scene_label=result.get('scene_label'),
                interaction_score=result.get('interaction_score')
            )
            if success:
                print(f"💾 [入库] {result.get('scene_label')} | 评分:{result.get('interaction_score')} | {result['summary'][:15]}...")
    except Exception as e:
        print(f"❌ [后台异常] {e}")

if __name__ == "__main__":
    main()