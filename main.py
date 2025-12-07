import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
import config

from src.perception.perception_processor import PerceptionProcessor
from src.memory.memory_stream import MemoryStream
from src.memory.long_term_memory import LongTermMemory
from src.cognition.cognitive_core import CognitiveCore

# 配置日志输出格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("========================================")
    logging.info("   HearthScribe 智能看护代理启动中...   ")
    logging.info("========================================")
    
    # 1. 初始化
    try:
        perception = PerceptionProcessor(index_dir=config.FACE_INDEX_DIR)
        memory_stream = MemoryStream(config.IMAGE_STORAGE_PATH)
        ltm = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        cognition = CognitiveCore()
        logging.info("✅ 所有核心模块初始化成功。")
    except Exception as e:
        logging.critical(f"❌ 初始化失败: {e}", exc_info=True)
        return

    # 2. 打开摄像头
    cap = cv2.VideoCapture(config.SOURCE_VIDEO)
    if not cap.isOpened():
        logging.critical(f"❌ 无法连接摄像头 (ID: {config.SOURCE_VIDEO})")
        return

    executor = ThreadPoolExecutor(max_workers=2)
    frame_count = 0
    
    # 计算采样间隔 (例如每30帧采一次)
    PROCESS_INTERVAL_FRAMES = 30 // config.SAMPLE_FPS 
    if PROCESS_INTERVAL_FRAMES < 1: PROCESS_INTERVAL_FRAMES = 1

    logging.info(f"🎥 监控服务已启动。采样频率: 每 {PROCESS_INTERVAL_FRAMES} 帧分析一次。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                logging.warning("⚠️ 视频流中断，尝试重连...")
                time.sleep(1)
                continue
            
            # 降频处理
            if frame_count % PROCESS_INTERVAL_FRAMES == 0:
                # 打印心跳，证明程序还活着
                logging.info(f"📸 [Frame {frame_count}] 正在采样分析...")
                
                # A. 感知 (PerceptionProcessor 现在会自己打印详细日志)
                detections = perception.process_frame(frame)
                
                if detections:
                    names = [d['name'] for d in detections if d.get('name')]
                    logging.info(f"🎯 最终有效目标: {len(detections)} 个 {names}")
                else:
                    # 这一行虽然和 Perception 重复，但作为主流程的确认很有必要
                    logging.info("💨 当前帧无有效人物目标。")

                # B. 记忆流 (Memory Stream)
                # 只有当 detections 不为空，或者 MemoryStream 正在录制中时，这里才会有逻辑
                event_pack = memory_stream.update(frame, detections)
                
                # C. 认知分析 (Cognition)
                if event_pack:
                    event_id = event_pack['event_id']
                    duration = event_pack['end_time'] - event_pack['start_time']
                    logging.info(f"📦 [事件切片] 生成新事件 {event_id} (时长: {duration:.1f}s)，推送到后台分析...")
                    executor.submit(bg_analyze, event_pack, cognition, ltm)
            
            frame_count += 1
            # 简单的休眠防止空转 CPU 占用过高 (因为没有imshow的阻塞了)
            time.sleep(0.01)

    except KeyboardInterrupt:
        logging.info("\n🛑 接收到退出指令，正在关闭系统...")
    finally:
        cap.release()
        executor.shutdown(wait=False)
        logging.info("👋 系统已安全退出。")

def bg_analyze(event, cognition, ltm):
    """后台分析线程"""
    eid = event['event_id']
    logging.info(f"🧠 [后台] 正在调用 ERNIE 模型分析事件 {eid}...")
    try:
        result = cognition.analyze_event(event)
        if result:
            success = ltm.save_event(event, result['summary'], result['kg_data'])
            if success:
                logging.info(f"✅ [入库成功] 事件 {eid}: {result['summary'][:30]}...")
            else:
                logging.error(f"❌ [入库失败] 事件 {eid} 数据库写入失败")
        else:
            logging.warning(f"⚠️ [分析跳过] 事件 {eid} 未生成有效摘要")
    except Exception as e:
        logging.error(f"❌ [后台异常] 事件 {eid} 处理出错: {e}", exc_info=True)

if __name__ == "__main__":
    main()