# src/web_utils.py
import pandas as pd
import json
import logging
import time
from datetime import datetime, timedelta
from openai import OpenAI
import config
from pyvis.network import Network
import networkx as nx
import streamlit as st

logger = logging.getLogger(__name__)

# --- 单例初始化 ---
# 使用 Streamlit 的缓存机制来创建和管理全局唯一的数据库和客户端实例
@st.cache_resource
def get_memory_instance():
    try:
        from src.memory.long_term_memory import LongTermMemory
        logger.info("Initializing LongTermMemory instance for web app...")
        return LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
    except Exception as e:
        st.error(f"Failed to initialize LongTermMemory: {e}")
        return None

@st.cache_resource
def get_llm_client():
    logger.info("Initializing OpenAI client instance for web app...")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

# --- 知识图谱可视化 ---
def generate_knowledge_graph_html(limit=200):
    MEMORY = get_memory_instance()
    if not MEMORY: return "<div>Backend Error: Memory module not initialized.</div>"
    relations = MEMORY.get_all_kg_data(limit=limit)
    if not relations: return "<div>暂无知识图谱数据。请先让 Agent 运行并观察一些事件。</div>"
    
    G = nx.DiGraph()
    for r in relations:
        G.add_node(r['source'], group=r['source_type'], title=f"{r['source_type']}: {r['source']}")
        G.add_node(r['target'], group=r['target_type'], title=f"{r['target_type']}: {r['target']}")
        G.add_edge(r['source'], r['target'], label=r['relation'], title=f"Relation in Event: {r['event_id']}")
    
    net = Network(height="600px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 16, "face": "tahoma" }, "borderWidth": 2, "shadow": true },
      "edges": { "color": { "inherit": true }, "smooth": { "type": "continuous" }, "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } } },
      "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 }, "maxVelocity": 50, "solver": "forceAtlas2Based", "timestep": 0.35, "stabilization": { "iterations": 150 } },
      "groups": {
          "Person": { "color": "#FFADAD", "shape": "dot", "size": 25 },
          "Object": { "color": "#A0C4FF", "shape": "box" },
          "Location": { "color": "#9BF699", "shape": "triangle" },
          "Activity": { "color": "#FDFFB6", "shape": "diamond" }
      }
    }
    """)
    
    return net.generate_html(name='temp_graph.html', local=False)

# --- 问答功能 ---
def answer_question(question):
    MEMORY = get_memory_instance()
    LLM_CLIENT = get_llm_client()
    if not MEMORY or not LLM_CLIENT:
        yield "错误：后端服务未连接。"
        return

    rich_events = MEMORY.hybrid_search(question, top_k=4)
    if not rich_events:
        yield "我的记忆中似乎没有与此相关的信息。"
        return

    context_str = "这是基于我记忆检索到的相关事件及其知识图谱细节：\n\n"
    for i, event in enumerate(rich_events):
        time_str = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
        context_str += f"--- 事件片段 {i+1} [{time_str}] ---\n"
        context_str += f"【摘要】: {event['summary']}\n"
        context_str += f"【知识图谱关系】: {event['kg_text']}\n\n"

    system_prompt = "你是一个AI Agent的记忆核心。请基于提供的【记忆上下文】，用自然、流畅的口吻回答用户。利用知识图谱中的细节让回答更智能。如果信息不足，就坦诚说明。"
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"【记忆上下文】\n{context_str}\n\n用户问题: {question}"}]
        response = LLM_CLIENT.chat.completions.create(model=config.LLM_MODEL_NAME, messages=messages, stream=True, temperature=0.4)
        for chunk in response:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                time.sleep(0.01)
    except Exception as e:
        yield f"生成回答时出错: {e}"

# --- 事件流数据获取 (修复图片破图) ---
def get_recent_events_df(limit=50):
    MEMORY = get_memory_instance()
    if not MEMORY: return pd.DataFrame()
    
    cursor = MEMORY.sqlite_conn.cursor()
    cursor.execute("SELECT start_time, summary, preview_image_path FROM events WHERE preview_image_path IS NOT NULL AND preview_image_path != '' ORDER BY start_time DESC LIMIT ?", (limit,))
    events_data = [dict(row) for row in cursor.fetchall()]
    
    if not events_data:
        return pd.DataFrame(columns=['Time', 'Summary', 'Preview'])

    # 将图片路径转换为图片二进制数据
    for event in events_data:
        try:
            with open(event['preview_image_path'], 'rb') as f:
                event['preview_image_data'] = f.read()
        except FileNotFoundError:
            event['preview_image_data'] = None
            logger.warning(f"Preview image not found at path: {event['preview_image_path']}")
        except Exception as e:
            event['preview_image_data'] = None
            logger.error(f"Error reading preview image {event['preview_image_path']}: {e}")

    df = pd.DataFrame(events_data)
    df['Time'] = pd.to_datetime(df['start_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.rename(columns={'summary': 'Summary', 'preview_image_data': 'Preview'}, inplace=True)
    return df[['Time', 'Summary', 'Preview']]

# --- 报告生成功能 (新增) ---
def generate_summary_report(start_date, end_date):
    MEMORY = get_memory_instance()
    LLM_CLIENT = get_llm_client()
    if not MEMORY or not LLM_CLIENT:
        return "错误: 后端服务未连接。"

    logger.info(f"开始生成报告，时间范围: {start_date} -> {end_date}")

    start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
    end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

    try:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute(
            "SELECT start_time, summary FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time ASC",
            (start_ts, end_ts)
        )
        events = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"从SQLite查询事件失败: {e}", exc_info=True)
        return f"数据库查询失败: {e}"

    if not events:
        return f"在 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 期间没有发现任何记忆记录。"

    context_str = f"以下是从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 期间，按时间顺序记录的所有活动摘要：\n\n"
    for event in events:
        event_time = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M')
        context_str += f"- [{event_time}] {event['summary']}\n"
    
    report_prompt = f"""
你是一位专业的家庭生活分析师和日记作者。请将以下在指定时间段内记录的零散活动日志，整合成一份结构清晰、洞察深刻的分析报告。

【活动记录】:
{context_str}
---
【你的任务】:
请生成一份Markdown格式的报告，必须包含以下几个部分：

### 1. 总体摘要 (Summary)
用一段生动的文字，高度概括这段时间内的主要生活状态和核心活动。

### 2. 主要活动时间线 (Key Activities Timeline)
以列表形式，列出几个最重要的、有代表性的活动或事件，并简要描述。

### 3. 行为模式与洞察 (Behavioral Patterns & Insights)
分析是否存在任何有趣的习惯、规律或趋势。例如：
- 某人是否在特定时间段倾向于做某件事？
- 主要的活动区域是哪里？
- 是否有任何与平时不同的“异常”事件？

### 4. 情感与氛围评估 (Mood & Atmosphere Assessment)
根据事件描述的字里行间，推测这段时间的整体氛围是怎样的（例如：高效、放松、忙碌、平静等），并给出你的理由。

请以客观、关怀的口吻撰写这份报告。
"""
    try:
        logger.info("正在调用LLM生成报告...")
        response = LLM_CLIENT.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=[{"role": "user", "content": report_prompt}],
            temperature=0.5
        )
        report_text = response.choices[0].message.content
        return report_text
    except Exception as e:
        logger.error(f"调用LLM生成报告失败: {e}", exc_info=True)
        return f"调用大模型生成报告时出错: {e}"