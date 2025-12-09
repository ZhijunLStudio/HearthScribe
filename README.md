# HearthScribe: 基于文心大模型的适老化智能看护系统

**HearthScribe** 是一个深度集成 **百度文心大模型** 的适老化智能看护系统。它不仅仅是一个摄像头监控程序，更是一个拥有“感知”、“认知”和“记忆”能力的家庭守护者，专为长者设计。

本项目充分利用文心大模型强大的多模态理解与逻辑推理能力，能实时识别长者与家人的互动，利用 **ERNIE-4.5-VL** 捕捉生活瞬间，生成语义化摘要，并构建本地知识图谱。通过自然语言交互，用户可检索记忆、生成健康报告，实现 24 小时无微不至的 AI 关怀。

![alt text](images/homepage.png)

## ✨ 核心功能

*   **👁️ 智能感知与身份识别**
    *   使用 **PaddleX 的 PicoDet-S (PPLCNet_x1_0_person_detection)** 进行边缘端实时人体检测与追踪，轻量高效，适用于 CPU/GPU 环境。
    *   集成 **PaddleX 的 PP-YOLOE_plus-S_face** (人脸检测) + **ResNet50_face** (人脸识别)，自动识别长者及家庭成员身份，支持高精度特征提取。
    *   **摄像头守护进程**：具备断线自动重连和错误恢复机制，确保持续稳定运行，24 小时不间断守护。

*   **🧠 文心驱动的语义记忆与健康认知**
    *   **动态采样**：基于长者活动检测自动开始/停止录制，过滤无效画面，专注关键时刻。
    *   **视觉认知 (ERNIE-4.5-VL)**：调用文心视觉大模型分析事件截图，精准理解长者行为、物体交互与场景细节（如“长者正在沙发上阅读，手里拿着一杯水”）。
    *   **知识提取**：从摘要中提取实体（长者、物品、地点）和关系，构建结构化知识图谱，支持关系推理。

*   **💾 双重记忆存储 (RAG)**
    *   **向量记忆 (LanceDB)**：存储事件摘要的语义向量，支持模糊搜索长者历史活动。
    *   **结构化记忆 (SQLite)**：存储元数据、知识图谱三元组和原始图片路径，便于健康数据追踪。

*   **💬 自然语言交互与关怀报告**
    *   **智能路由**：Master Agent 理解用户意图，自动调度“记忆检索”、“图谱推理”或“健康总结”专家。
    *   **多轮问答 (ERNIE-4.5)**：基于 RAG，利用文心大模型的中文理解能力，用温暖口吻回答关于长者历史的复杂问题（如“妈妈今天精神如何？”）。
    *   **自动适老报告**：生成日报、周报，涵盖生活画像、健康评估、风险检测与专属建议，帮助子女远程关怀。

## 🏗️ 系统架构与运行逻辑

系统主要由以下几个核心模块组成，专为适老场景优化：

1.  **感知层 (`perception_processor.py`)**:
    *   处理视频流，利用 PaddleX 的 PicoDet-S 进行人体检测与追踪，结合 PP-YOLOE_plus-S_face + ResNet50_face 将检测框映射为长者姓名。
2.  **记忆流 (`memory_stream.py`)**:
    *   作为短期记忆缓冲区。当画面检测到长者活动时开始缓存帧，静止超过阈值（如30秒）则判定事件结束，打包成“Event”。
3.  **认知核心 (`cognitive_core.py`)**:
    *   接收打包的 Event，选取关键帧发送给 **LVM (视觉大模型)** 生成摘要，评估交互与风险。
    *   调用 **LLM** 从摘要中提取 Knowledge Graph (KG) 数据，注入健康洞察。
4.  **长期记忆 (`long_term_memory.py`)**:
    *   将文本摘要通过 Embedding 模型转换为向量存入 LanceDB。
    *   将 KG 数据和事件元数据存入 SQLite，支持长期健康趋势分析。
5.  **代理交互 (`master_agent.py` & `web_utils.py`)**:
    *   处理用户查询，协调各模块检索信息，最终生成温暖的关怀回答与报告。

<div align="center">
  <img src="images/system.png" style="width: 60%; max-height: none; display: block;" />
</div>

## 📂 目录结构

```text
HearthScribe/
├── main.py                 # 主程序入口：启动摄像头采集与后台处理循环
├── config.py               # 全局配置文件
├── .env                    # 环境变量（API Key等）
├── requirements.txt        # 依赖库
├── known_faces/            # 已知人脸库目录（优先录入长者照片）
│   ├── grandma/            # 长者照片文件夹
│   └── family_member/      # 家庭成员照片文件夹
├── memory_db/              # 数据库存储路径 (自动生成)
├── event_images/           # 事件图片存储路径 (自动生成)
└── src/
    ├── agent/              # 代理与路由逻辑
    ├── cognition/          # 视觉理解与认知核心
    ├── memory/             # 向量库与数据库管理
    ├── perception/         # PaddleX 检测与人脸识别
    └── app/                # UI与任务接口
```

## 🚀 快速开始

### 1. 环境准备

建议使用 Python 3.10+ 环境，并安装 PaddlePaddle（推荐 CPU/GPU 版本）。

```bash
# 克隆仓库
git clone https://github.com/ZhijunLStudio/HearthScribe.git
cd HearthScribe

# 安装依赖（包括 PaddleX）
pip install -r requirements.txt
pip install paddlepaddle  # 或 paddlepaddle-gpu
pip install paddlex  # 核心 PaddleX 库
```

### 2. 模型与配置

#### 自动下载的模型
首次运行时，代码会自动从 PaddleX 官方模型库下载以下模型到本地缓存（`~/.paddlex/official_models/`）：
*   **PPLCNet_x1_0_person_detection (PicoDet-S)**: 用于人体检测，轻量级实时追踪。
*   **PP-YOLOE_plus-S_face**: 用于人脸检测。
*   **ResNet50_face**: 用于人脸特征提取与识别。
*   **SentenceTransformer (all-MiniLM-L6-v2)**: 用于生成文本向量嵌入。

如果需要重新下载，删除 `~/.paddlex/official_models/` 目录即可。

#### 配置文件 (.env)
在项目根目录创建 `.env` 文件，配置你的大模型 API（支持 OpenAI 格式，可接入本地 vLLM 或云端服务）。已解耦为视觉大模型 (LVM) 和语言大模型 (LLM) 配置：

```ini
# 视觉大模型配置 (LVM)
LVM_API_KEY=sk-xxxx
LVM_BASE_URL=https://api.example.com/v1
LVM_MODEL_NAME=ernie-4.5-turbo-vl  # 或本地模型名称

# 语言大模型配置 (LLM)
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.example.com/v1
LLM_MODEL_NAME=ernie-4.5-21b-a3b-thinking
```

### 3. 录入人脸数据
在 `known_faces` 目录下创建以人名命名的文件夹，并放入该人的清晰面部照片（jpg/png）。**优先录入长者照片**，以提升识别准确率。
例如：
*   `./known_faces/grandma/photo1.jpg`
*   `./known_faces/son/photo1.jpg`

系统启动时会自动加载并编码这些人脸（使用 ResNet50_face 提取特征）。

### 4. 运行系统

**启动后台采集与监控：**

```bash
python main.py
```
*程序启动后会初始化摄像头守护线程，加载 PaddleX 模型，并开始在后台静默记录长者活动。*

**启动 Web 交互界面 ：**

```bash
streamlit run tools/app_streamlit.py
```

## ⚙️ 关键配置说明 (`config.py`)

你可以根据硬件性能调整 `config.py` 中的参数，优化适老守护体验：

*   **`PROCESS_INTERVAL`**: 主循环处理间隔，调大可降低 CPU 占用（默认 2 秒）。
*   **`FRAME_CAPTURE_INTERVAL`**: 事件记录时每隔几秒抓取一帧（默认 2 秒），影响 LVM 理解的细粒度。
*   **`EVENT_INACTIVITY_TIMEOUT`**: 画面静止多久后判定事件结束（默认 30 秒），适合长者慢节奏活动。
*   **`EVENT_MAX_DURATION_SECONDS`**: 单个事件最大时长，超过会强制切分（默认 60 秒）。
*   **`DET_MODEL_NAME`**: 检测模型名称，默认 `"PPLCNet_x1_0_person_detection"` (PicoDet-S)，可切换其他 PaddleX 模型。
*   **`PADDLE_DEVICE`**: PaddlePaddle 运行设备，默认 `"cpu"`，GPU 用户可设为 `"gpu"`。