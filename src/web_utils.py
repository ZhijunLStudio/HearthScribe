# src/web_utils.py
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from openai import OpenAI
import config
import logging
import time

# 初始化一次性的模块实例
from src.memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)

# --- 全局实例 ---
# 将初始化包裹在try-except中，以便在UI中显示错误
try:
    MEMORY = LongTermMemory(db_path=config.DB_PATH)
    LLM_CLIENT = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    logger.info("Web Utils: 后端模块初始化成功。")
except Exception as e:
    # 打印到控制台，以便调试
    print(f"CRITICAL: Failed to initialize modules for web app: {e}")
    # 在日志中记录详细错误
    logging.critical("Web Utils: 后端模块初始化失败。", exc_info=True)
    MEMORY = None
    LLM_CLIENT = None

def get_all_memories_df():
    """获取所有记忆并返回DataFrame"""
    if MEMORY is None:
        logger.error("MEMORY模块未初始化，无法获取记忆。")
        return pd.DataFrame()
    try:
        return MEMORY.table.to_pandas()
    except Exception as e:
        logger.error(f"从LanceDB获取记忆失败: {e}", exc_info=True)
        return pd.DataFrame()

def format_memories_for_display(df):
    """格式化DataFrame以便在UI中显示"""
    if df.empty:
        return [], "暂无记忆数据。", pd.DataFrame()
        
    # 时间格式化
    df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 参与者格式化
    def safe_json_loads(x):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return [] # 如果解析失败，返回空列表
    df['participants_list'] = df['participants'].apply(safe_json_loads)
    df['participants'] = df['participants_list'].apply(lambda x: ', '.join(x) if x else '无')

    # 提取第一张图片作为预览图
    def get_first_image(paths_json):
        try:
            paths = json.loads(paths_json)
            return paths[0] if paths else None
        except (json.JSONDecodeError, TypeError):
            return None
    df['preview_image'] = df['image_paths'].apply(get_first_image)

    # 准备用于显示的DataFrame，并按时间倒序排列
    display_df = df[['time', 'summary', 'participants']].sort_values(by='time', ascending=False)
    
    # 创建Gallery所需格式
    gallery_data = []
    if 'preview_image' in df.columns:
        valid_previews = df.dropna(subset=['preview_image'])
        for _, row in valid_previews.sort_values(by='time', ascending=False).iterrows():
            gallery_data.append((row['preview_image'], f"{row['time']}\n{row['summary']}"))

    status = f"共找到 {len(df)} 条记忆记录。"
    return gallery_data, status, display_df

def answer_question(question, history):
    """RAG问答逻辑 (作为生成器，用于流式输出)"""
    if MEMORY is None or LLM_CLIENT is None:
        yield "错误：后端模块未成功初始化。"
        return
        
    logging.info(f"收到RAG问题: {question}")
    # 增加搜索结果数量以获得更丰富上下文
    retrieved_memories = MEMORY.search_memory(question, top_k=5)

    if not retrieved_memories:
        yield "抱歉，关于这个问题，记忆库中没有找到相关记录。"
        return

    context = "以下是一些按时间排序的相关记忆片段：\n\n"
    for i, mem in enumerate(retrieved_memories):
        event_time = datetime.fromtimestamp(mem['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        context += f"--- 记忆片段 {i+1} (时间: {event_time}) ---\n"
        context += f"摘要: {mem['summary']}\n"
        try:
            participants_list = json.loads(mem.get('participants', '[]'))
            participants_str = ', '.join(participants_list) if participants_list else '未知'
        except:
            participants_str = '未知'
        context += f"参与者: {participants_str}\n\n"

    # BINGO! 核心修正：完全客观、中立的Prompt
    prompt = f"""
你是一个AI记忆分析助手。你的任务是根据下面提供的【背景记忆信息】，客观、准确地回答用户的问题。

【背景记忆信息】:
{context}
---
用户的问题是: "{question}"

请严格遵守以下规则回答：
1.  你的回答必须完全基于上面提供的【背景记忆信息】。
2.  始终使用第三人称来描述事件和人物。例如：“根据记忆，lizhijun当时正在...” 或 “在那个时间点，一个穿红衣服的人...”。
3.  **绝对不要**使用“我”或“你”等人称代词来指代AI自己或用户。
4.  如果信息不足以回答，就明确说明“根据现有的记忆片段，无法确定...”。
"""
    try:
        response = LLM_CLIENT.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # 进一步降低温度，让回答更客观
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                yield content
                time.sleep(0.02)
    except Exception as e:
        logging.error(f"调用LLM生成答案失败: {e}", exc_info=True)
        yield f"调用大模型时出错: {e}"

def run_analysis(start_date, end_date):
    """执行指定时间范围的总结分析"""
    if MEMORY is None or LLM_CLIENT is None:
        return "错误：后端模块未成功初始化。"

    logging.info(f"开始执行分析，时间范围: {start_date} 到 {end_date}")
    
    try:
        all_memories = get_all_memories_df()
        if all_memories.empty:
            return "数据库中没有任何记忆，无法进行分析。"

        start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
        end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

        period_memories = all_memories[(all_memories['timestamp'] >= start_ts) & (all_memories['timestamp'] <= end_ts)]
        
        if period_memories.empty:
            return f"在 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 期间没有任何记忆记录。"

        context = f"以下是从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 期间，按时间顺序记录的所有活动片段。\n\n"
        for _, row in period_memories.sort_values(by='timestamp').iterrows():
            event_time = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M')
            try:
                participants = json.loads(row.get('participants', '[]'))
            except:
                participants = []
            context += f"- 时间: {event_time}, 人物: {participants or '未知'}, 事件: {row['summary']}\n"

        # BINGO! 同样修正这里的Prompt为中立视角
        prompt = f"""
你是一位专业的行为数据分析师。请将以下在指定时间段内记录的活动日志，整合成一份客观的、结构化的分析报告。

报告应该包含：
1. **整体摘要**: 高度概括这段时间内观察到的主要活动和状态。
2. **主要活动**: 列出几个最主要的活动类别及其描述。
3. **行为模式或异常**: 指出任何有趣的习惯、趋势或与平时不同的地方。

【活动记录】:
{context}
---
请生成这份分析报告，使用Markdown格式化，使其清晰易读。
"""
        response = LLM_CLIENT.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        report_text = response.choices[0].message.content
        return report_text

    except Exception as e:
        logging.error(f"执行分析失败: {e}", exc_info=True)
        return f"执行分析时出错: {e}"

def get_time_range():
    """获取数据库中记忆的最早和最晚时间"""
    df = get_all_memories_df()
    if df.empty:
        now = datetime.now()
        return now.date(), now.date()
    min_date = pd.to_datetime(df['timestamp'].min(), unit='s').date()
    max_date = pd.to_datetime(df['timestamp'].max(), unit='s').date()
    return min_date, max_date