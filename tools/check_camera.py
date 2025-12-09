import cv2
import time

def open_camera(index):
    """尝试打开指定索引的摄像头"""
    cap = cv2.VideoCapture(index)
    # 设置分辨率（可选，有些摄像头默认分辨率很低）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头索引: {index}")
        return None
    
    # 尝试读取一帧来确认摄像头真的在工作
    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ 摄像头 {index} 已打开，但无法读取画面 (可能是黑屏或权限问题)")
        # 即使无法读取，也返回对象以便后续重试，但在本逻辑中我们倾向于认为它不可用
        # 不过为了防止只是第一帧的问题，还是返回 cap
    
    print(f"✅ 成功连接摄像头索引: {index}")
    return cap

def main():
    current_index = 0
    cap = open_camera(current_index)

    print("\n" + "="*40)
    print(" 🎥 摄像头可视化工具")
    print(" 按键说明:")
    print(" [S] 或 [Space] : 切换到下一个摄像头")
    print(" [Q] or [Esc]   : 退出程序")
    print("="*40 + "\n")

    while True:
        if cap is None or not cap.isOpened():
            # 如果当前摄像头不可用，尝试显示一个黑屏或者提示信息
            # 这里简单处理：如果没有摄像头，尝试重连或者等待用户切换
            key = cv2.waitKey(100) & 0xFF
        else:
            ret, frame = cap.read()
            
            if ret:
                # 在画面左上角显示当前摄像头的编号
                text = f"Camera Index: {current_index}"
                cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 显示画面
                cv2.imshow('Camera Feed', frame)
            else:
                print(f"摄像头 {current_index} 读取失败，请尝试切换...")

        # 监听按键
        key = cv2.waitKey(1) & 0xFF

        # 按 'q' 或 'Esc' 退出
        if key == ord('q') or key == 27:
            break
        
        # 按 's' 或 '空格' 切换下一个摄像头
        elif key == ord('s') or key == 32:
            print("🔄 正在切换摄像头...")
            if cap:
                cap.release()
            
            # 尝试下一个索引
            # 通常电脑摄像头不会超过 5 个，我们循环检测
            next_found = False
            # 简单的逻辑：尝试下一个，如果失败则继续尝试，直到找到或回到原点
            # 这里为了简单，直接 +1，用户可以一直按直到找到画面
            current_index += 1
            # 如果索引太大（比如超过4），通常意味着没有更多摄像头了，可以重置回0
            # 但有些虚拟摄像头ID可能很大，这里暂不自动重置，由用户决定
            if current_index > 4: 
                print("索引已超过4，重置回 0")
                current_index = 0
            
            cap = open_camera(current_index)

    # 清理工作
    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()