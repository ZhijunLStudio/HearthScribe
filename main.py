import cv2
import time
from pathlib import Path
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# --- 确保src目录在Python路径中 ---
# (如果你的 main_collector.py 和 src 在同一级，这个设置是正确的)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- 从src导入我们的模块 ---
from perception.perception_processor import PerceptionProcessor
from memory.memory_stream import MemoryStream
from memory.long_term_memory import LongTermMemory
from cognition.cognitive_core import CognitiveCore
import config

# --- 全局变量和线程锁 ---
latest_frame = None
frame_lock = threading.Lock()
is_running = True
DEBUG_WINDOW_ENABLED = True  # 设置为 False 可以关闭窗口，纯后台运行

def setup_logging():
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')

    file_handler = logging.FileHandler(log_dir / "main_collector.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    logging.getLogger("httpx").setLevel(logging.WARNING)

def camera_thread_func():
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
            logging.warning("摄像头未能读取到帧，可能是视频结束或设备问题。")
            time.sleep(0.5)

    cap.release()
    logging.info("摄像头线程已停止。")

def draw_debug_info(frame, detections):
    debug_frame = frame.copy()
    for det in detections:
        box = det.get('box')
        if box is None: continue
        track_id = det.get('track_id', '?')
        name = det.get('name', 'Unknown')
        
        x1, y1, x2, y2 = map(int, box)
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {name}"
        
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y1_label = max(y1, label_size[1] + 10)
        
        cv2.rectangle(debug_frame, (x1, y1_label - label_size[1] - 10), 
                      (x1 + label_size[0], y1_label + base_line - 10), color, cv2.FILLED)
        cv2.putText(debug_frame, label, (x1, y1_label - 7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return debug_frame

def process_event_in_background(packaged_event, cognition, long_term_memory):
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
    logging.info("--- HearthScribe Agent 启动中 ---")

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
    
    logging.info("--- 系统已就绪，开始监控 ---")
    if DEBUG_WINDOW_ENABLED:
        cv2.namedWindow("HearthScribe - Debug View", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HearthScribe - Debug View", 1280, 720)

    try:
        while is_running:
            start_time = time.time()
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.1)
                    continue
                current_frame = latest_frame.copy()

            detections = perception.process_frame(current_frame)
            logging.info(f"感知完成, 检测到 {len(detections)} 个目标: {[d['name'] for d in detections]}")
            packaged_event = short_term_memory.update(current_frame, detections)

            if packaged_event:
                logging.info(f"打包事件 {packaged_event['event_id']} 完成，提交到后台处理。")
                executor.submit(process_event_in_background, packaged_event, cognition, long_term_memory)
            
            if DEBUG_WINDOW_ENABLED:
                debug_frame = draw_debug_info(current_frame, detections)
                cv2.imshow("HearthScribe - Debug View", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    is_running = False
                    break
            
            elapsed = time.time() - start_time
            sleep_time = max(0, config.PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("\n检测到用户中断...")
    finally:
        logging.info("正在关闭系统...")
        is_running = False
        if cam_thread.is_alive():
            cam_thread.join()
        executor.shutdown(wait=True)
        if DEBUG_WINDOW_ENABLED:
            cv2.destroyAllWindows()
        logging.info("系统已关闭。")

if __name__ == "__main__":
    main_loop()