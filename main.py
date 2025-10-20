import cv2
import time
from pathlib import Path
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# --- 路径设置 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- 模块导入 ---
from perception.perception_processor import PerceptionProcessor
from memory.memory_stream import MemoryStream
from memory.long_term_memory import LongTermMemory
from cognition.cognitive_core import CognitiveCore
import config

# --- 全局变量 ---
latest_frame = None
frame_lock = threading.Lock()
is_running = True

def setup_logging():
    # ... (这个函数保持不变，内容省略)
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')
    file_handler = logging.FileHandler(log_dir / "main_collector.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def camera_thread_func():
    # ... (这个函数保持不变，内容省略)
    global latest_frame, is_running
    cap = cv2.VideoCapture(config.SOURCE_VIDEO)
    if not cap.isOpened():
        logging.critical("摄像头线程无法打开摄像头。")
        is_running = False
        return
    logging.info("摄像头已成功打开。")
    time.sleep(1.0) 
    while is_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            logging.warning("摄像头未能读取到帧。")
            time.sleep(0.5)
    cap.release()
    logging.info("摄像头线程已停止。")

def process_event_in_background(packaged_event, cognition, long_term_memory):
    # ... (这个函数保持不变，内容省略)
    event_id = packaged_event['event_id']
    logging.info(f"🚀 [后台] 开始处理事件 {event_id}")
    try:
        result = cognition.analyze_event(packaged_event)
        if not result or not result.get('summary'):
             logging.error(f"❌ [后台] 事件 {event_id} 分析失败，摘要为空，将被丢弃！")
             return
        success = long_term_memory.save_event(
            event_data=packaged_event,
            summary=result['summary'],
            kg_data=result.get('kg_data')
        )
        if success:
            logging.info(f"✅ [后台] 事件 {event_id} 处理并保存完毕。")
        else:
             logging.error(f"❌ [后台] 事件 {event_id} 保存到数据库失败！")
    except Exception as e:
        logging.critical(f"💥 [后台] 处理事件 {event_id} 时发生严重错误: {e}", exc_info=True)


def main_loop():
    global is_running
    setup_logging()
    logging.info("--- HearthScribe Agent (后台模式) 启动中 ---")

    try:
        perception = PerceptionProcessor(config.KNOWN_FACES_DIR)
        short_term_memory = MemoryStream(config.IMAGE_STORAGE_PATH)
        long_term_memory = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        cognition = CognitiveCore()
        logging.info("所有模块初始化成功。")
    except Exception as e:
        logging.critical(f"模块初始化失败，程序无法启动: {e}", exc_info=True)
        return

    executor = ThreadPoolExecutor(max_workers=3)
    cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    cam_thread.start()

    logging.info("等待第一帧图像...")
    while latest_frame is None and is_running:
        time.sleep(0.5)

    if not is_running:
        logging.error("摄像头未能提供图像，程序退出。")
        return
    
    logging.info("--- 系统已就绪，开始监控 (按 Ctrl+C 停止) ---")

    try:
        while is_running:
            start_time = time.time()
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                current_frame = latest_frame.copy()

            detections = perception.process_frame(current_frame)
            
            # BINGO! 恢复了每次循环都打印的日志，方便你观察
            if detections:
                logging.info(f"感知完成, 检测到 {len(detections)} 个目标: {[d['name'] for d in detections]}")

            packaged_event = short_term_memory.update(current_frame, detections)

            if packaged_event:
                logging.info(f"打包事件 {packaged_event['event_id']} 完成，提交到后台处理。")
                executor.submit(process_event_in_background, packaged_event, cognition, long_term_memory)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, config.PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("\n检测到用户中断 (Ctrl+C)...")
    finally:
        logging.info("正在关闭系统...")
        is_running = False
        if cam_thread.is_alive():
            cam_thread.join()
        executor.shutdown(wait=True)
        logging.info("系统已关闭。")

if __name__ == "__main__":
    main_loop()