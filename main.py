import cv2
import time
from pathlib import Path
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# --- 确保src目录在Python路径中 ---
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

def setup_logging():
    """配置全局日志记录器"""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    # BINGO: 优化日志格式，包含模块名
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')

    file_handler = logging.FileHandler(log_dir / "main_collector.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    # BINGO: 为一些特别吵的库设置更高的日志级别，避免刷屏
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def camera_thread_func():
    """一个专门的线程，只负责以最快速度读取摄像头，并更新全局的最新帧。"""
    global latest_frame, is_running
    cap = cv2.VideoCapture(config.SOURCE_VIDEO)
    if not cap.isOpened():
        logging.critical("CRITICAL: 摄像头线程无法打开摄像头。")
        is_running = False
        return
        
    logging.info("摄像头已成功打开。")
    # BINGO: 添加一点延迟，确保摄像头硬件准备就绪
    time.sleep(1.0) 

    while is_running:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            # BINGO: 如果读取失败，可能是视频文件结束或设备断开
            logging.warning("摄像头未能读取到帧，可能是视频结束或设备问题。")
            time.sleep(0.5)

    cap.release()
    logging.info("摄像头线程已停止。")


def process_event_in_background(packaged_event, cognition, long_term_memory):
    """这个函数在后台线程中执行，负责所有耗时的认知和记忆任务。"""
    event_id = packaged_event['event_id']
    
    # BINGO! 添加顶层try-except块，捕获所有未预料的异常
    try:
        logging.info(f"[后台任务] 开始处理事件 {event_id}...")
        
        start_time = time.time()
        analysis_result = cognition.analyze_event_with_lvm(packaged_event)
        end_time = time.time()
        
        # BINGO! 增加判断，只有成功获取结果才记录耗时
        if analysis_result:
            logging.info(f"[后台任务] 事件 {event_id} 的LVM分析耗时: {end_time - start_time:.2f} 秒。")

        if analysis_result and "summary" in analysis_result:
            summary = analysis_result["summary"]
            logging.info(f"[后台任务] 认知核心分析结果: \"{summary}\"")
            participants = set(d['name'] for f in packaged_event['frames'] for d in f['detections'])
            image_paths = [f["image_path"] for f in packaged_event["frames"]]
            long_term_memory.save_event_memory(
                event_id=event_id,
                timestamp=packaged_event["start_time"],
                summary=summary,
                participants=list(participants),
                image_paths=image_paths
            )
        else:
            logging.warning(f"[后台任务] 认知核心未能成功分析事件 {event_id}。分析结果为: {analysis_result}")
            
        logging.info(f"[后台任务] 事件 {event_id} 处理完毕。")

    except Exception as e:
        # BINGO! exc_info=True 会打印完整的错误堆栈信息，非常关键！
        logging.error(f"[后台任务] 处理事件 {event_id} 时发生致命错误: {e}", exc_info=True)


def main_loop():
    global is_running
    setup_logging()
    logging.info("========================================")
    logging.info("正在初始化家居记忆助手 (数据采集)...")

    # BINGO: 将初始化过程包裹在try-except中，以便捕获配置错误等
    try:
        perception = PerceptionProcessor(known_faces_dir=config.KNOWN_FACES_DIR)
        short_term_memory = MemoryStream(storage_path=config.IMAGE_STORAGE_PATH)
        long_term_memory = LongTermMemory(db_path=config.DB_PATH)
        cognition = CognitiveCore()
    except Exception as e:
        logging.critical(f"模块初始化失败: {e}", exc_info=True)
        return

    executor = ThreadPoolExecutor(max_workers=2)

    cam_thread = threading.Thread(target=camera_thread_func, daemon=True)
    cam_thread.start()
    logging.info("摄像头线程已启动。等待第一帧图像...")

    while latest_frame is None and is_running and cam_thread.is_alive():
        time.sleep(0.5)

    if not is_running or not cam_thread.is_alive():
        logging.error("系统初始化失败，摄像头未能提供图像，即将退出。")
        if cam_thread.is_alive():
            is_running = False
            cam_thread.join()
        return

    logging.info("--- 系统已就绪，开始监控 ---")

    try:
        while is_running:
            start_process_time = time.time() # BINGO: 记录循环开始时间

            with frame_lock:
                if latest_frame is None:
                    continue
                current_frame = latest_frame.copy()

            # BINGO: 计时感知处理
            start_perception_time = time.time()
            detections = perception.process_frame(current_frame)
            end_perception_time = time.time()
            
            # BINGO: 添加详细的检测日志
            if detections:
                detected_names = [d['name'] for d in detections]
                logging.info(
                    f"感知处理完成 (耗时: {end_perception_time - start_perception_time:.3f}s). "
                    f"检测到 {len(detections)} 个目标: {detected_names}"
                )
            
            packaged_event = short_term_memory.update(current_frame, detections)

            if packaged_event:
                logging.info(f"打包事件 {packaged_event['event_id']} 完成，提交到后台处理。")
                executor.submit(process_event_in_background, packaged_event, cognition, long_term_memory)
            
            # BINGO: 确保主循环频率稳定
            elapsed_time = time.time() - start_process_time
            sleep_time = max(0, config.PROCESS_INTERVAL - elapsed_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("\n检测到用户中断，正在关闭系统...")
    finally:
        is_running = False
        if cam_thread.is_alive():
            cam_thread.join()
        executor.shutdown(wait=True)
        logging.info("系统已关闭。")


if __name__ == "__main__":
    main_loop()