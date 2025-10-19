import os
from dotenv import load_dotenv

# --- 加载 .env 文件中的环境变量 (如果你用了的话) ---
load_dotenv()

# --- 摄像头与感知配置 ---
SOURCE_VIDEO = 0  # 如果用文件测试，改为视频路径，如 "./test_video.mp4"

# --- 核心采样与处理频率 ---
PROCESS_INTERVAL = 0.5 # 主循环稍微快一点，提高响应速度

# --- 动态事件配置 ---
FRAME_CAPTURE_INTERVAL = 2    # 每2秒抓一帧，减少冗余
EVENT_INACTIVITY_TIMEOUT = 60 # 60秒无人，事件结束
EVENT_MAX_DURATION_SECONDS = 300 # 最长5分钟

# --- LVM/LLM API 配置 (请确保在 .env 或这里填入了正确的值) ---
LVM_API_KEY = os.getenv("LVM_API_KEY", "")
LVM_BASE_URL = os.getenv("LVM_BASE_URL", "")
LVM_MODEL_NAME = os.getenv("LVM_MODEL_NAME", "ernie-4.5-turbo-vl")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ernie-4.5-turbo-vl")

# --- 存储配置 ---
LANCEDB_PATH = "./memory_db/lancedb"        # LanceDB 向量库路径
SQLITE_DB_PATH = "./memory_db/knowledge.db" # SQLite 知识图谱路径
IMAGE_STORAGE_PATH = "./event_images"       # 图片存储路径
KNOWN_FACES_DIR = "./known_faces"           # 已知人脸目录