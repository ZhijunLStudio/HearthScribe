import os
from dotenv import load_dotenv

# --- 加载 .env 文件中的环境变量 ---
# 这会寻找项目根目录下的 .env 文件并加载它
load_dotenv()

# --- 摄像头与感知配置 ---
SOURCE_VIDEO = 0

# --- 核心采样与处理频率 ---
PROCESS_INTERVAL = 2

# --- 动态事件配置 ---
FRAME_CAPTURE_INTERVAL = 10
EVENT_INACTIVITY_TIMEOUT = 60
EVENT_MAX_DURATION_SECONDS = 300

# --- 视觉大模型 (LVM) API 配置 ---
# 使用 os.getenv() 从环境变量中读取配置
# 第二个参数是默认值，如果环境变量中没有找到，就会使用它
LVM_API_KEY = os.getenv("LVM_API_KEY", "YOUR_LVM_API_KEY_NOT_SET")
LVM_BASE_URL = os.getenv("LVM_BASE_URL", "https://example.com/v1")
LVM_MODEL_NAME = os.getenv("LVM_MODEL_NAME", "default-lvm-model")

# --- 语言大模型 (LLM) API 配置 (用于总结和问答) ---
LLM_API_KEY = os.getenv("LLM_API_KEY", "YOUR_LLM_API_KEY_NOT_SET")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://example.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "default-llm-model")

# --- 存储配置 ---
DB_PATH = "./memory_db"
IMAGE_STORAGE_PATH = "./event_images"
KNOWN_FACES_DIR = "./known_faces"

# --- 添加一个简单的检查，提醒用户设置密钥 ---
if LVM_API_KEY == "YOUR_LVM_API_KEY_NOT_SET" or LLM_API_KEY == "YOUR_LLM_API_KEY_NOT_SET":
    print("\n" + "="*50)
    print("⚠️  警告: API密钥未在 .env 文件中设置!")
    print("请复制 .env.example 文件为 .env，并填入您的API密钥。")
    print("="*50 + "\n")