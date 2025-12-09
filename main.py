# main.py
import os
# 禁用并行库冲突警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import time
import logging
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import cv2
import config

# 日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# --- 核心修复：无阻塞摄像头读取类 ---
class CameraLoader:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        # 设置缓冲区大小为1（尝试物理减少延迟）
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # 极短的休眠，避免死循环占满 CPU，但要足够快以清空 Buffer
            time.sleep(0.005) 

    def read(self):
        with self.read_lock:
            if not self.grabbed: return None
            return self.frame.copy()

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

def main():
    print("\n=== HearthScribe 空间指挥舱启动 (零延迟版) ===\n")

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

    # 2. 启动摄像头线程
    try:
        print(f"\n🎥 正在启动摄像头线程 (Source: {config.SOURCE_VIDEO})...")
        cam_loader = CameraLoader(config.SOURCE_VIDEO).start()
        print(f"✅ 摄像头就绪 | 策略: 实时获取最新帧")
    except Exception as e:
        print(f"❌ 摄像头启动失败: {e}")
        return

    executor = ThreadPoolExecutor(max_workers=1)
    last_process_time = 0 
    
    try:
        while True:
            # 直接获取最新一帧 (Zero Latency)
            frame = cam_loader.read()
            
            if frame is None: 
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            
            # 控制检测频率
            if current_time - last_process_time >= config.PROCESS_INTERVAL:
                
                last_process_time = current_time
                current_time_str = datetime.now().strftime("%H:%M:%S")
                
                # --- 图像缩放加速 ---
                h, w = frame.shape[:2]
                scale = 640 / w
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                
                # A. 感知
                detections = perception.process_frame(small_frame)
                
                # B. 坐标还原 & 状态反馈
                if not detections:
                    print(f"[{current_time_str}] 💤 空间闲置中...", end='\r')
                else:
                    # 还原坐标到原图尺寸
                    for det in detections:
                        if 'box' in det:
                            det['box'] = [int(c / scale) for c in det['box']]
                        if 'face_box' in det and det['face_box']:
                            det['face_box'] = [int(c / scale) for c in det['face_box']]

                    # C. 记忆流
                    event_pack = memory_stream.update(frame, detections)
                    
                    # D. 后台分析
                    if event_pack:
                        duration = event_pack['end_time'] - event_pack['start_time']
                        print(f"\n📦 [{current_time_str}] 生成事件片段 ({duration:.1f}s) -> 提交大脑分析")
                        executor.submit(bg_analyze, event_pack, cognition, ltm)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 系统停止")
    finally:
        cam_loader.stop()
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
                # 打印更详细的日志以便调试
                label = result.get('scene_label')
                score = result.get('interaction_score')
                print(f"💾 [入库] {label} (Score:{score}) | {result['summary'][:20]}...")
    except Exception as e:
        print(f"❌ [后台异常] {e}")

if __name__ == "__main__":
    main()