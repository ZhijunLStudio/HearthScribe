import os
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

# --- 核心凭证 (请务必填入你的真实 Key) ---
# 你的报错说明这里的 os.getenv 读出来可能是 None 或者空字符串
# 为了测试，你可以先暂时把 Key 硬编码在这里，跑通了再改回去
ERNIE_API_KEY = os.getenv("ERNIE_API_KEY", "") 
ERNIE_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"

API_KEY = ERNIE_API_KEY
BASE_URL = ERNIE_BASE_URL

# --- 模型名称 ---
AI_VL_MODEL = "ernie-4.5-turbo-vl"         
AI_THINKING_MODEL = "ernie-4.5-vl-28b-a3b-thinking" 

# --- 硬件与输入 ---
SOURCE_VIDEO = 0 
PADDLE_DEVICE = "cpu"
FACE_INDEX_DIR = "face_index"
DET_MODEL_NAME = "PicoDet-S"

# --- 存储路径 ---
LANCEDB_PATH = "./memory_db/lancedb"
SQLITE_DB_PATH = "./memory_db/knowledge.db"
IMAGE_STORAGE_PATH = "./event_images"
DAILY_REPORTS_PATH = "./daily_reports"
KNOWN_FACES_DIR = "./known_faces"

# --- 嵌入模型 ---
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# --- 运行策略 ---
PROCESS_INTERVAL = 1
SAMPLE_FPS = 1 
EVENT_DURATION = 30
FRAME_CAPTURE_INTERVAL = 5 
EVENT_INACTIVITY_TIMEOUT = 60
EVENT_MAX_DURATION_SECONDS = 300