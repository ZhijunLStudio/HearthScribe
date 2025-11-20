import time
from pathlib import Path
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

# --- 路径设置 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- 全局变量 ---
latest_frame = None
frame_lock = threading.Lock()
is_running = True

def setup_logging():
    # ... (此函数不变)
    log_dir = Path("./logs"); log_dir.mkdir(exist_ok=True)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s')
    file_handler = logging.FileHandler(log_dir / "main_agent.log", mode='a', encoding='utf-8'); file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler(sys.stdout); stream_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger(); root_logger.handlers.clear(); root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler); root_logger.addHandler(stream_handler)
    logging.getLogger("httpx").setLevel(logging.WARNING); logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# =============================================================
# !!! 核心改动：独立的、带重试逻辑的摄像头初始化函数 !!!
# =============================================================
def try_get_camera(max_indices_to_check=5, width=640, height=480, fps=30):
    """
    在一个最小化的环境中，自动探测并尝试打开第一个可用的摄像头。
    如果失败，会返回 None。
    """
    import cv2
    
    for index in range(max_indices_to_check):
        logging.info(f"[摄像头守护] 正在尝试打开摄像头索引 {index}...")
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            logging.info(f"[摄像头守护] 索引 {index} 已打开，正在验证...")
            time.sleep(1.0)
            ret, frame = cap.read()
            if ret and frame is not None:
                logging.info(f"✅ [摄像头守护] 成功！找到可用摄像头于索引 {index}。")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                return cap
            else:
                logging.warning(f"[摄像头守护] 索引 {index} 能打开但无法读取帧。")
                cap.release()
    return None

# =============================================================
# !!! 核心改动：永不退出的摄像头守护线程 !!!
# =============================================================
def camera_guardian_thread():
    global latest_frame, is_running
    
    while is_running:
        cap = None
        # --- 连接循环 ---
        while is_running and cap is None:
            cap = try_get_camera()
            if cap is None:
                logging.error("[摄像头守护] 未找到可用摄像头，将在10秒后重试...")
                time.sleep(10)
        
        logging.info("[摄像头守护] 摄像头已连接，开始读取视频流。")
        read_failures = 0
        max_read_failures = 30 # 增加容错次数
        
        # --- 读取循环 ---
        while is_running:
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame.copy()
                read_failures = 0
                time.sleep(0.01) # 短暂休眠，让出CPU
            else:
                read_failures += 1
                logging.warning(f"[摄像头守护] 读取帧失败 ({read_failures}/{max_read_failures})")
                if read_failures > max_read_failures:
                    logging.critical("[摄像头守护] 连续读取帧失败，判定摄像头已断开。将进入重连模式。")
                    with frame_lock:
                        latest_frame = None # 通知主循环图像已丢失
                    break # 跳出读取循环，回到外层的连接循环
                time.sleep(0.5)

        # 释放无效的摄像头对象
        if cap:
            cap.release()
            
    logging.info("[摄像头守护] 守护线程已停止。")

def process_event_in_background(packaged_event, cognition, long_term_memory):
    # ... (此函数不变)
    event_id = packaged_event['event_id']; logging.info(f"🚀 [后台] 开始处理事件 {event_id}")
    try:
        result = cognition.analyze_event(packaged_event)
        if not result or not result.get('summary'):
             logging.error(f"❌ [后台] 事件 {event_id} 分析失败！"); return
        if long_term_memory.save_event(event_data=packaged_event, summary=result['summary'], kg_data=result.get('kg_data')):
            logging.info(f"✅ [后台] 事件 {event_id} 处理并保存完毕。")
        else:
             logging.error(f"❌ [后台] 事件 {event_id} 保存到数据库失败！")
    except Exception as e:
        logging.critical(f"💥 [后台] 处理事件 {event_id} 时发生严重错误: {e}", exc_info=True)

def main_loop():
    global is_running, latest_frame
    setup_logging()
    logging.info("--- HearthScribe Agent (后台模式) 启动中 ---")

    import config

    # 启动摄像头守护线程
    cam_thread = threading.Thread(target=camera_guardian_thread, daemon=True)
    cam_thread.start()

    logging.info("等待摄像头守护线程提供第一帧图像...")
    start_wait_time = time.time()
    while latest_frame is None and is_running:
        if time.time() - start_wait_time > 30: # 等待时间可以长一点
            logging.critical("启动超时(30秒)，仍未获取到第一帧图像。请检查摄像头硬件。程序将继续尝试后台连接。")
            break # 不再退出，让守护线程继续工作
        time.sleep(1)

    if latest_frame is not None:
        logging.info(f"✅ 成功获取第一帧图像！")
    
    # 无论是否获取到第一帧，都继续加载模型，因为守护线程会持续尝试
    try:
        logging.info("--- 开始加载AI模型和数据库 ---")
        from perception.perception_processor import PerceptionProcessor
        from memory.memory_stream import MemoryStream
        from memory.long_term_memory import LongTermMemory
        from cognition.cognitive_core import CognitiveCore
        
        perception = PerceptionProcessor(config.KNOWN_FACES_DIR)
        short_term_memory = MemoryStream(config.IMAGE_STORAGE_PATH)
        long_term_memory = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        cognition = CognitiveCore()
        logging.info("✅ 所有AI模块初始化成功。")
    except Exception as e:
        logging.critical(f"模块初始化失败: {e}", exc_info=True)
        is_running = False; cam_thread.join(); return

    executor = ThreadPoolExecutor(max_workers=3)
    logging.info("--- 系统已就绪，开始监控 (按 Ctrl+C 停止) ---")
    
    try:
        while is_running:
            current_frame = None
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()
            
            # 如果当前没有图像（摄像头断开），主循环就暂停并等待
            if current_frame is None:
                logging.warning("主循环：未获取到有效图像，等待摄像头恢复...")
                time.sleep(2)
                continue

            start_time = time.time()
            
            import cv2 # 保持动态导入
            detections = perception.process_frame(current_frame)
            if detections: logging.info(f"感知完成, 检测到 {len(detections)} 个目标: {[d['name'] for d in detections]}")
            
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
        logging.info("正在关闭系统..."); is_running = False
        if cam_thread.is_alive(): cam_thread.join()
        executor.shutdown(wait=True); logging.info("系统已关闭。")

if __name__ == "__main__":
    main_loop()