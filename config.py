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
EVENT_MAX_DURATION_SECONDS = 60
EVENT_INACTIVITY_TIMEOUT = 30

# --- 存储路径 ---
LANCEDB_PATH = "./memory_db/lancedb"
SQLITE_DB_PATH = "./memory_db/knowledge.db"
IMAGE_STORAGE_PATH = "./event_images"
FACE_INDEX_DIR = "./face_index"

# --- 视觉大模型 API 配置 (LVM) ---
LVM_API_KEY = os.getenv("LVM_API_KEY", "")
LVM_BASE_URL = os.getenv("LVM_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
LVM_MODEL_NAME = "ernie-4.5-turbo-vl"  # 视觉模型

# --- 语言大模型 API 配置 (LLM) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
LLM_MODEL_NAME = "ernie-4.5-21b-a3b-thinking"  # 思考/总结模型

# --- 其他模型 ---
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# PaddlePaddle 设置
DET_MODEL_NAME = "PPLCNet_x1_0_person_detection"

# 检查 Key 是否存在
if not LVM_API_KEY:
    print("⚠️ 警告: 未检测到 LVM_API_KEY，请检查 .env 文件！")
if not LLM_API_KEY:
    print("⚠️ 警告: 未检测到 LLM_API_KEY，请检查 .env 文件！")
