# tools/collect_faces.py
import cv2
from pathlib import Path
from ultralytics import YOLO
import face_recognition
import time
from concurrent.futures import ThreadPoolExecutor
import sys

# --- 用户可配置参数 ---
SOURCE_VIDEO = 0
OUTPUT_DIR = Path("./face_crops")
TARGET_FPS_PROCESS = 1
DEFAULT_FRAME_SKIP = 30
MAX_WORKERS = 4

# --- 全局变量用于控制保存频率 ---
last_save_time = 0
SAVE_INTERVAL = 1.0

def process_frame_async(frame, frame_number, yolo_model):
    """
    V5版本核心处理逻辑：
    1. YOLO粗定位"人"。
    2. 只在"人"的区域内，运行face_recognition精确定位"人脸"。
    3. 裁剪并保存第一张符合频率限制的"人脸"。
    """
    global last_save_time
    
    try:
        t_start = time.perf_counter()

        # --- 阶段一: YOLO粗定位"人"，这步是关键的过滤器 ---
        person_results = yolo_model(frame, classes=0, verbose=False)
        t_yolo_done = time.perf_counter()

        face_locations_found = []
        t_preprocess_start = t_yolo_done

        # --- 阶段二: 只在YOLO找到的人形区域内进行人脸检测 ---
        # 并且只在可以保存时才进行检测，最大化节省资源
        current_time = time.time()
        if person_results[0].boxes and (current_time - last_save_time >= SAVE_INTERVAL):
            for person_box in person_results[0].boxes:
                p_x1, p_y1, p_x2, p_y2 = map(int, person_box.xyxy[0])
                person_roi = frame[p_y1:p_y2, p_x1:p_x2]

                if person_roi.size == 0:
                    continue
                
                # BGR -> RGB转换
                rgb_person_roi = person_roi[:, :, ::-1]
                
                # 在小区域内进行人脸检测
                # face_locations的坐标是相对于person_roi的
                locations = face_recognition.face_locations(rgb_person_roi, model="hog")
                
                # 将相对坐标转换回绝对坐标并存储
                for top, right, bottom, left in locations:
                    face_locations_found.append((p_y1 + top, p_x1 + right, p_y1 + bottom, p_x1 + left))

        t_face_detect_done = time.perf_counter()

        # --- 阶段三: 保存第一张找到的人脸 ---
        saved_this_frame = False
        if face_locations_found:
            top, right, bottom, left = face_locations_found[0]
            face_crop = frame[top:bottom, left:right]
            
            if face_crop.size > 0:
                filename = OUTPUT_DIR / f"face_{int(current_time)}.jpg"
                cv2.imwrite(str(filename), face_crop)
                last_save_time = current_time # 成功保存后才更新时间
                saved_this_frame = True
        
        t_save_done = time.perf_counter()

        # --- 性能日志 ---
        yolo_ms = (t_yolo_done - t_start) * 1000
        face_detect_ms = (t_face_detect_done - t_preprocess_start) * 1000
        save_ms = (t_save_done - t_face_detect_done) * 1000 if saved_this_frame else 0
        total_ms = (t_save_done - t_start) * 1000

        log_message = (
            f"Frame: {frame_count: <5} | "
            f"YOLO: {yolo_ms:<5.1f}ms | "
            f"FaceDetect: {face_detect_ms:<5.1f}ms | "
            f"Save: {save_ms:<5.1f}ms | "
            f"Total: {total_ms:<6.1f}ms | "
            f"Faces: {len(face_locations_found): <2} | "
            f"Saved: {'Yes' if saved_this_frame else 'No'}"
        )
        print(log_message + " " * 10, end='\r')

    except Exception as e:
        print(f"\n[线程错误] 处理帧 {frame_number} 失败: {e}", file=sys.stderr)


def main():
    """
    主函数：纯后台运行，修正了YOLO逻辑并增加了摄像头健壮性。
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("人脸数据采集脚本启动 (V5 - 逻辑修正与健壮性提升)...")
    print(f"裁剪出的人脸图像将保存在: {OUTPUT_DIR.resolve()}")
    print("按 Ctrl+C 停止程序。")

    model = YOLO("weights/yolov8n.pt")
    print("YOLOv8n 模型加载成功。")

    # --- 摄像头健壮性提升 ---
    # 在macOS上明确指定API后端，并增加延时
    cap = cv2.VideoCapture(SOURCE_VIDEO, cv2.CAP_AVFOUNDATION)
    time.sleep(2.0) # 给摄像头2秒钟的初始化时间

    if not cap.isOpened():
        print("错误: 无法打开摄像头。请检查：", file=sys.stderr)
        print("1. 摄像头是否连接正常？", file=sys.stderr)
        print("2. 其他程序是否正在使用摄像头？", file=sys.stderr)
        print("3. 在macOS的'系统设置'->'隐私与安全性'中，是否已为终端/IDE授予摄像头权限？", file=sys.stderr)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        frame_skip = round(fps / TARGET_FPS_PROCESS) if TARGET_FPS_PROCESS > 0 else 1
        print(f"摄像头FPS: {fps:.2f}。目标处理速率: {TARGET_FPS_PROCESS} FPS。将每隔 {frame_skip} 帧处理一帧。")
    else:
        frame_skip = DEFAULT_FRAME_SKIP
        print(f"无法获取摄像头FPS。将使用默认值，每隔 {frame_skip} 帧处理一帧。")

    global frame_count
    frame_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 增加一个小延时再试一次，应对瞬间的读取失败
                    time.sleep(0.1)
                    ret, frame = cap.read()
                    if not ret:
                        print("\n视频流结束或读取失败。")
                        break

                frame_count += 1
                
                if frame_count % frame_skip == 0:
                    executor.submit(process_frame_async, frame.copy(), frame_count, model)
                
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n检测到用户中断 (Ctrl+C)...")
        finally:
            print("\n正在等待所有后台任务完成...")
            cap.release()
            print("脚本已优雅地停止。")

if __name__ == "__main__":
    main()