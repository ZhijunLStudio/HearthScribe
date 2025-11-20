import cv2
import time
import sys

def test_camera_background(camera_index):
    """在纯后台模式下测试摄像头"""
    print(f"\n--- 正在后台测试摄像头索引: {camera_index} ---")
    
    # 使用简单索引，因为我们知道它可能有效
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"  ❌ 失败：无法通过索引 {camera_index} 打开摄像头。")
        return False
        
    print(f"  ✅ 成功打开摄像头索引 {camera_index}。")
    
    # 给予摄像头充足的启动时间
    print("     等待2秒让视频流稳定...")
    time.sleep(2.0)
    
    print("     尝试读取5帧图像...")
    frames_read = 0
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"       - 成功读取第 {i+1} 帧，尺寸: {frame.shape}")
            frames_read += 1
        else:
            print(f"       - ❌ 失败：未能读取第 {i+1} 帧。")
        time.sleep(0.5) # 模拟处理间隔

    cap.release()
    print("     摄像头已释放。")
    
    if frames_read > 0:
        print(f"  ✅ 测试成功！摄像头索引 {camera_index} 可以稳定提供图像。")
        return True
    else:
        print(f"  ❌ 测试失败！虽然摄像头能打开，但无法稳定读取图像。")
        return False

def main():
    print("===================================")
    print(" Jetson 摄像头后台测试脚本")
    print("===================================")
    
    # 根据您之前的 ls 输出，我们知道可能有 0 和 2
    # 但根据测试脚本的成功日志，0 已经成功打开过一次
    indices_to_test = [0, 1, 2] 
    
    print(f"\n将依次测试以下索引: {indices_to_test}")
    
    successful_index = None
    
    for index in indices_to_test:
        if test_camera_background(index):
            successful_index = index
            # 找到第一个能用的就停止
            break
            
    print("\n--- 测试总结 ---")
    if successful_index is not None:
        print(f"✅ 最终确认！摄像头索引 {successful_index} 在后台模式下工作正常。")
        print("\n   >>> 请在您的 `config.py` 文件中，进行如下设置: <<<")
        print(f"   >>> SOURCE_VIDEO = {successful_index} <<<")
    else:
        print("❌ 所有尝试的索引都未能稳定读取图像。")
        print("   问题可能与多线程下的资源竞争或系统权限有关。")
        print("   请尝试以下步骤：")
        print("   1. 运行 `sudo usermod -aG video $(whoami)` 并重启Jetson。")
        print("   2. 确保没有其他程序（即使是僵尸进程）正在占用摄像头。")

if __name__ == "__main__":
    main()