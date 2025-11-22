# config.py
import os
from dotenv import load_dotenv

# --- 加载 .env 文件中的环境变量 ---
load_dotenv()

# --- 摄像头与感知配置 ---
SOURCE_VIDEO = 0 

# --- 核心采样与处理频率 ---
PROCESS_INTERVAL = 2 # 主循环稍微快一点，提高响应速度

# --- 动态事件配置 ---
FRAME_CAPTURE_INTERVAL = 5    # 每5秒抓一帧，减少冗余
EVENT_INACTIVITY_TIMEOUT = 60 # 60秒无人，事件结束
EVENT_MAX_DURATION_SECONDS = 300 # 最长5分钟

# --- LVM/LLM API 配置 ---
# 注意：在最终部署时，这些URL应指向你本地部署的模型服务
LVM_API_KEY = os.getenv("LVM_API_KEY", "your_api_key_here")
LVM_BASE_URL = os.getenv("LVM_BASE_URL", "http://localhost:8000/v1")
LVM_MODEL_NAME = os.getenv("LVM_MODEL_NAME", "local-lvm-model")

LLM_API_KEY = os.getenv("LLM_API_KEY", "your_api_key_here")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "local-llm-model")

# --- 存储配置 ---
LANCEDB_PATH = "./memory_db/lancedb"        # LanceDB 向量库路径
SQLITE_DB_PATH = "./memory_db/knowledge.db" # SQLite 知识图谱路径
IMAGE_STORAGE_PATH = "./event_images"       # 图片存储路径
KNOWN_FACES_DIR = "./known_faces"           # 已知人脸目录

# --- 报告存储路径 ---
DAILY_REPORTS_PATH = "./daily_reports"      # 每日自动生成报告的存储路径

EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"