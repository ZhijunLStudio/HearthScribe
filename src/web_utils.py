import pandas as pd
import json
import logging
import time
from datetime import datetime
from openai import OpenAI
import config
# BINGO! 导入 pyvis 用于图谱生成
from pyvis.network import Network
import networkx as nx

# 初始化单例
try:
    from src.memory.long_term_memory import LongTermMemory
    MEMORY = LongTermMemory(config.DB_PATH, config.SQLITE_DB_PATH)
    LLM_CLIENT = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
except Exception as e:
    logging.critical(f"Web Utils backend init failed: {e}", exc_info=True)
    MEMORY = None
    LLM_CLIENT = None

logger = logging.getLogger(__name__)

# --- 1. 知识图谱可视化核心函数 ---
def generate_knowledge_graph_html(limit=200):
    """
    从 SQLite 获取最新的关系数据，并使用 Pyvis 生成一个互动的 HTML 图谱。
    返回: HTML 字符串
    """
    if not MEMORY: return "<div>Backend Error</div>"

    # 1. 获取数据
    relations = MEMORY.get_all_kg_data(limit=limit)
    if not relations:
        return "<div>暂无足够的知识图谱数据。请先让 Agent 观察一些事件。</div>"

    # 2. 构建 NetworkX 图
    G = nx.DiGraph()
    for r in relations:
        # 添加节点 (带颜色区分类型)
        G.add_node(r['source'], group=r['source_type'], title=r['source_type'])
        G.add_node(r['target'], group=r['target_type'], title=r['target_type'])
        # 添加边 (带标签)
        G.add_edge(r['source'], r['target'], label=r['relation'], title=f"Event: {r['event_id']}")

    # 3. 使用 Pyvis 转换和美化
    net = Network(height="600px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    net.from_nx(G)
    
    # 配置物理引擎让图更好看
    net.set_options("""
    var options = {
      "nodes": {
        "font": { "size": 16, "face": "tahoma" },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": { "inherit": true },
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
      },
      "physics": {
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 100,
            "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      },
      "groups": {
          "Person": { "color": "#FF9999", "shape": "dot" },
          "Object": { "color": "#99CCFF", "shape": "box" },
          "Location": { "color": "#99FF99", "shape": "triangle" },
          "Activity": { "color": "#FFFF99", "shape": "diamond" }
      }
    }
    """)
    
    # 4. 导出为 HTML 字符串
    try:
        # save() 实际上会写文件，我们用 write_html 把内容写到内存StringIO可能更干净，
        # 但 Pyvis 的 API 设计比较偏向文件。这里生成临时文件再读取。
        tmp_path = "temp_graph.html"
        net.save_graph(tmp_path)
        with open(tmp_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        return f"<div>Error generating graph: {e}</div>"

# --- 2. 升级后的 RAG 问答 (完整上下文，不简化) ---
def answer_question(question, history):
    if not MEMORY or not LLM_CLIENT:
        yield "错误：后端未连接。"
        return

    # 1. 混合检索 (Hybrid Search)
    # 这一步会先用向量搜 LanceDB，再拿 ID 去 SQLite 查完整的事件和 KG
    rich_events = MEMORY.hybrid_search(question, top_k=4)

    if not rich_events:
        yield "我的记忆中似乎没有与此相关的信息。"
        return

    # 2. 构建“富”上下文
    context_str = "这是基于我记忆检索到的相关事件及其知识图谱细节：\n\n"
    for i, event in enumerate(rich_events):
        time_str = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M:%S')
        context_str += f"--- 事件片段 {i+1} [{time_str}] ---\n"
        context_str += f"【摘要】: {event['summary']}\n"
        # BINGO! 这里加入了完整的 KG 细节，不再是“简化处理”
        context_str += f"【知识图谱关系】: {event['kg_text']}\n\n"

    # 3. 生成回答
    system_prompt = """
你通过一个具备视觉和认知能力的 AI Agent 的记忆来回答问题。
请基于提供的【记忆上下文】，用自然、流畅、类似人类助手的口吻回答用户。
- 如果记忆中有明确的实体关系（如“A在B地点”或“A使用了B”），请在回答中利用这些细节，这会让回答显得更聪明。
- 坦诚：如果上下文中没有足够信息，就直接说不知道，不要编造。
- 不要提及“根据记忆片段1...”这种技术术语，融合成一段连贯的叙述。
"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【记忆上下文】\n{context_str}\n\n用户问题: {question}"}
        ]
        response = LLM_CLIENT.chat.completions.create(
            model=config.LLM_MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.4 # 稍微提高一点温度，让回答更自然
        )
        for chunk in response:
            if chunk.choices.delta.content:
                yield chunk.choices.delta.content
                time.sleep(0.01)
    except Exception as e:
        yield f"生成回答时出错: {e}"

# --- 3. 其他辅助函数 (保持或微调) ---
def get_recent_events_df(limit=50):
    """获取最近的事件列表用于表格显示"""
    if not MEMORY: return pd.DataFrame()
    cursor = MEMORY.sqlite_conn.cursor()
    cursor.execute("SELECT start_time, summary FROM events ORDER BY start_time DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=['timestamp', 'summary'])
    if not df.empty:
        df['time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        return df[['time', 'summary']]
    return pd.DataFrame(columns=['time', 'summary'])