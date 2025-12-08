# config.py
import os
from dotenv import load_dotenv

# 1. 强制优先加载 .env
load_dotenv(override=True)

# --- 基础配置 ---
# 摄像头索引或视频路径
SOURCE_VIDEO = 0  
# 检测频率 (秒)
PROCESS_INTERVAL = 2
# 高频采样/切片逻辑
FRAME_CAPTURE_INTERVAL = 2
EVENT_MAX_DURATION_SECONDS = 300
EVENT_INACTIVITY_TIMEOUT = 30

# --- 存储路径 ---
LANCEDB_PATH = "./memory_db/lancedb"
SQLITE_DB_PATH = "./memory_db/knowledge.db"
IMAGE_STORAGE_PATH = "./event_images"
FACE_INDEX_DIR = "./face_index"

# --- 模型 API 配置 (关键修复) ---
# 优先读取 ERNIE_API_KEY，如果没有则尝试读取 API_KEY
API_KEY = os.getenv("ERNIE_API_KEY") or os.getenv("API_KEY", "")
BASE_URL = os.getenv("ERNIE_BASE_URL") or os.getenv("BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")

# 模型名称
AI_VL_MODEL = "ernie-4.5-turbo-vl"  # 视觉模型
AI_THINKING_MODEL = "ernie-4.5-vl-28b-a3b-thinking" # 思考/总结模型
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# PaddlePaddle 设置
PADDLE_DEVICE = "cpu" # 或 "gpu"
DET_MODEL_NAME = "PPLCNet_x1_0_person_detection"

# 检查 Key 是否存在
if not API_KEY:
    print("⚠️ 警告: 未检测到 API_KEY，请检查 .env 文件！")