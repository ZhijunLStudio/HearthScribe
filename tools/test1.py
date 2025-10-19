# test1.py (V2 - 使用cv2加载图像，解决元数据问题)
import face_recognition
import cv2  # 引入OpenCV
import sys

# --- 图片路径 ---
my_picture_path = "known_faces/lizhijun/face_1756780946.jpg"
unknown_picture_path = "known_faces/lizhijun/face_1756780993.jpg"

print(f"正在使用 OpenCV 加载已知图片: {my_picture_path}")
try:
    # 使用 cv2.imread 加载图片
    picture_of_me_bgr = cv2.imread(my_picture_path)
    if picture_of_me_bgr is None:
        raise FileNotFoundError
    # 将 BGR 转换为 RGB
    # picture_of_me = picture_of_me_bgr[:, :, ::-1]
    picture_of_me = cv2.cvtColor(picture_of_me_bgr, cv2.COLOR_BGR2RGB)
except FileNotFoundError:
    print(f"错误: 找不到文件或无法加载图片 {my_picture_path}")
    sys.exit(1)

# --- 计算已知图片的人脸编码 ---
my_face_encodings = face_recognition.face_encodings(picture_of_me)

if not my_face_encodings:
    print(f"错误: 在图片 '{my_picture_path}' 中未能检测到任何人脸。")
    sys.exit(1)

my_face_encoding = my_face_encodings[0]
print("成功为已知图片生成人脸编码。")

# --- 加载并处理未知图片 ---
print(f"\n正在使用 OpenCV 加载未知图片: {unknown_picture_path}")
try:
    # 同样使用 cv2.imread 加载
    unknown_picture_bgr = cv2.imread(unknown_picture_path)
    if unknown_picture_bgr is None:
        raise FileNotFoundError
    # 转换为 RGB
    # unknown_picture = unknown_picture_bgr[:, :, ::-1]
    unknown_picture = cv2.cvtColor(unknown_picture_bgr, cv2.COLOR_BGR2RGB)
except FileNotFoundError:
    print(f"错误: 找不到文件或无法加载图片 {unknown_picture_path}")
    sys.exit(1)

# --- 计算未知图片的人脸编码 ---
unknown_face_encodings = face_recognition.face_encodings(unknown_picture)

if not unknown_face_encodings:
    print(f"警告: 在图片 '{unknown_picture_path}' 中未能检测到任何人脸。")
    sys.exit(1)

unknown_face_encoding = unknown_face_encodings[0]
print("成功为未知图片生成人脸编码。")

# --- 进行比较 ---
print("\n正在比较两张人脸...")
results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("\n结果: 这两张照片是同一个人！")
else:
    print("\n结果: 这两张照片不是同一个人。")