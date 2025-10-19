import pandas as pd
import json
import logging
import time
from datetime import datetime
from openai import OpenAI
import config
from pyvis.network import Network
import networkx as nx
import streamlit as st

logger = logging.getLogger(__name__)

# --- BINGO! 延迟加载，避免启动时调用 Streamlit 命令 ---
# 将初始化函数保留，但不立即调用
@st.cache_resource
def get_memory_instance():
    try:
        from src.memory.long_term_memory import LongTermMemory
        logger.info("Initializing LongTermMemory instance...")
        return LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
    except Exception as e:
        st.error(f"Failed to initialize LongTermMemory: {e}")
        return None

@st.cache_resource
def get_llm_client():
    logger.info("Initializing OpenAI client instance...")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

# --- 在每个需要它们的函数内部去获取实例 ---

def generate_knowledge_graph_html(limit=200):
    MEMORY = get_memory_instance() # 在函数内部获取
    if not MEMORY: return "<div>Backend Error</div>"
    relations = MEMORY.get_all_kg_data(limit=limit)
    if not relations: return "<div>暂无知识图谱数据。</div>"
    
    G = nx.DiGraph()
    for r in relations:
        G.add_node(r['source'], group=r['source_type'], title=r['source_type'])
        G.add_node(r['target'], group=r['target_type'], title=r['target_type'])
        G.add_edge(r['source'], r['target'], label=r['relation'], title=f"Event: {r['event_id']}")
    
    net = Network(height="600px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    net.from_nx(G)
    net.set_options("""
    var options = { "nodes": { "font": { "size": 16 }, "borderWidth": 2, "shadow": true }, "edges": { "color": { "inherit": true }, "smooth": { "type": "continuous" }, "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } } }, "physics": { "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 }, "maxVelocity": 50, "solver": "forceAtlas2Based", "timestep": 0.35, "stabilization": { "iterations": 150 } }, "groups": { "Person": { "color": "#FF9999", "shape": "dot" }, "Object": { "color": "#99CCFF", "shape": "box" }, "Location": { "color": "#99FF99", "shape": "triangle" }, "Activity": { "color": "#FFFF99", "shape": "diamond" } } }
    """)
    
    # Pyvis 的 generate_html 可以直接返回字符串，避免写临时文件
    return net.generate_html(name='temp_graph.html', local=False) # 使用CDN资源，更干净

def answer_question(question):
    MEMORY = get_memory_instance()
    LLM_CLIENT = get_llm_client()

    if not MEMORY or not LLM_CLIENT:
        yield "错误：后端未连接。"
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

def get_recent_events_df(limit=50):
    MEMORY = get_memory_instance()
    if not MEMORY: return pd.DataFrame()
    cursor = MEMORY.sqlite_conn.cursor()
    cursor.execute("SELECT start_time, summary, preview_image_path FROM events WHERE preview_image_path IS NOT NULL ORDER BY start_time DESC LIMIT ?", (limit,))
    events_data = [dict(row) for row in cursor.fetchall()]
    if not events_data: return pd.DataFrame(columns=['Time', 'Summary', 'Preview'])
    df = pd.DataFrame(events_data)
    df['Time'] = pd.to_datetime(df['start_time'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
    df.rename(columns={'summary': 'Summary', 'preview_image_path': 'Preview'}, inplace=True)
    return df[['Time', 'Summary', 'Preview']]