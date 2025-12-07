import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# --- 核心凭证 (ERNIE/OpenAI) ---
# 建议在 .env 文件中设置 ERNIE_API_KEY 和 ERNIE_BASE_URL
# 如果没有 .env，这里会使用默认值或报错
ERNIE_API_KEY = os.getenv("ERNIE_API_KEY", "your_api_key_here")
ERNIE_BASE_URL = os.getenv("ERNIE_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3")

# --- 兼容性映射 (关键修复) ---
# 将所有不同模块可能用到的 Key 都指向同一个 ERNIE Key
# 这样无论代码里写的是 LLM_API_KEY 还是 API_KEY，都能正常工作
API_KEY = ERNIE_API_KEY
BASE_URL = ERNIE_BASE_URL

LLM_API_KEY = ERNIE_API_KEY
LLM_BASE_URL = ERNIE_BASE_URL
LLM_MODEL_NAME = "ernie-4.5-vl-28b-a3b-thinking" # 思考模型

LVM_API_KEY = ERNIE_API_KEY
LVM_BASE_URL = ERNIE_BASE_URL
LVM_MODEL_NAME = "ernie-4.5-turbo-vl" # 视觉模型

# --- 硬件与输入 ---
SOURCE_VIDEO = 0  # 0 代表默认摄像头，也可以填视频路径 "test.mp4"

# --- PaddleX 配置 ---
# Mac用户通常没有NVIDIA显卡，Paddle会自动切换到CPU，忽略相关警告即可
PADDLE_DEVICE = "gpu:0" 
FACE_INDEX_DIR = "face_index"
DET_MODEL_NAME = "PicoDet-S" # 目标检测模型

# --- 存储路径 ---
LANCEDB_PATH = "./memory_db/lancedb"
SQLITE_DB_PATH = "./memory_db/knowledge.db"
IMAGE_STORAGE_PATH = "./event_images"
DAILY_REPORTS_PATH = "./daily_reports"
KNOWN_FACES_DIR = "./known_faces"

# --- 向量嵌入模型 (修复上一个报错) ---
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# --- 采样与运行策略 ---
PROCESS_INTERVAL = 1      # 主循环处理间隔(秒)
SAMPLE_FPS = 1            # 视觉分析采样率
EVENT_DURATION = 30       # 事件切片时长(秒)
FRAME_CAPTURE_INTERVAL = 5 # 每5秒保存一帧
EVENT_INACTIVITY_TIMEOUT = 60 # 画面静止超时
EVENT_MAX_DURATION_SECONDS = 300