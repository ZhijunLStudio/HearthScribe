# src/web_utils.py (最终完整版)
import logging
from datetime import datetime
from pathlib import Path
import networkx as nx
from pyvis.network import Network
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# 确保能找到src目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.memory.long_term_memory import LongTermMemory
from src.agent.master_agent import MasterAgent

logger = logging.getLogger(__name__)

# --- 单例模式实例化 ---
_memory_instance = None
_master_agent_instance = None

def get_memory_instance():
    """获取LongTermMemory的单例"""
    global _memory_instance
    if _memory_instance is None:
        try:
            logger.info("Initializing LongTermMemory instance for web app...")
            _memory_instance = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize LongTermMemory: {e}", exc_info=True)
    return _memory_instance

def get_master_agent():
    """获取MasterAgent的单例"""
    global _master_agent_instance
    if _master_agent_instance is None:
        memory = get_memory_instance()
        if memory:
            logger.info("Initializing MasterAgent instance for web app...")
            _master_agent_instance = MasterAgent(memory)
    return _master_agent_instance

# --- 实例化 ---
# 在模块加载时就执行实例化，方便其他模块直接使用
MEMORY = get_memory_instance()
MASTER_AGENT = get_master_agent()

# --- 仪表盘功能 ---
def get_dashboard_stats():
    if not MEMORY: return {"new_memories": 0}
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    try:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE start_time >= ?", (today_start,))
        new_memories = cursor.fetchone()[0]
    except Exception:
        new_memories = 0
        
    return {"new_memories": new_memories}

def get_entity_count():
    if not MEMORY: return 0
    try:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM entities")
        return cursor.fetchone()[0]
    except Exception:
        return 0

def get_relation_count():
    if not MEMORY: return 0
    try:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM relationships")
        return cursor.fetchone()[0]
    except Exception:
        return 0

def get_today_summary():
    today_str = datetime.now().strftime('%Y-%m-%d')
    report_path = Path(config.DAILY_REPORTS_PATH) / f"report_{today_str}.md"
    if report_path.exists():
        return report_path.read_text(encoding='utf-8')
    
    now = datetime.now()
    if now.hour < 22:
        return f"正在观察中... 今日报告将于 **22:00** 自动生成。"
    else:
        return "今日报告正在生成或生成失败，请稍后查看。"

# --- 问答功能 ---
def agent_answer_stream(question):
    if not MASTER_AGENT:
        yield "错误：后端Agent未初始化。"
        return
    
    # MasterAgent的execute_query本身就是生成器，直接返回
    return MASTER_AGENT.execute_query(question, lambda x: None)


# --- 知识图谱浏览器功能 ---
def generate_knowledge_graph_html(limit=150, focused_entity=None):
    if not MEMORY: return "<div>后端错误: 内存模块未初始化.</div>"
    relations = MEMORY.get_all_kg_data(limit=limit)
    if not relations: return "<div>暂无知识图谱数据。</div>"
    
    G = nx.DiGraph()
    for r in relations:
        G.add_node(r['source'], group=r['source_type'], title=f"{r['source_type']}: {r['source']}")
        G.add_node(r['target'], group=r['target_type'], title=f"{r['target_type']}: {r['target']}")
        G.add_edge(r['source'], r['target'], label=r['relation'], title=f"Event: {r['event_id']}")
    
    net = Network(height="700px", width="100%", notebook=False, directed=True, cdn_resources='remote')
    net.from_nx(G)
    
    if focused_entity and focused_entity in G.nodes:
        for node in net.nodes:
            if node['id'] == focused_entity:
                node['color'] = '#FF6347' # Tomato Red
                node['size'] = 40
    
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 16, "face": "tahoma", "color": "#f0f0f0" }, "borderWidth": 2, "shadow": true },
      "edges": { "color": { "inherit": "both" }, "smooth": { "type": "continuous" }, "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } } },
      "physics": { "forceAtlas2Based": { "gravitationalConstant": -100, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08 }, "maxVelocity": 50, "solver": "forceAtlas2Based", "timestep": 0.35, "stabilization": { "iterations": 150 } },
      "groups": {
          "Person": { "color": "#FFADAD", "shape": "dot", "size": 25 },
          "Object": { "color": "#A0C4FF", "shape": "box" },
          "Location": { "color": "#9BF699", "shape": "triangle" },
          "Activity": { "color": "#FDFFB6", "shape": "diamond" }
      },
      "interaction": { "navigationButtons": true, "keyboard": true }
    }
    """)
    return net.generate_html(name='temp_graph.html', local=False)


def generate_manual_report(period: str):
    """根据指定的周期（日报/周报/月报）生成报告"""
    today = datetime.now().date()
    if period == "日报":
        start_date = today
        end_date = today
    elif period == "周报":
        start_date = today - timedelta(days=today.weekday())
        end_date = today
    elif period == "月报":
        start_date = today.replace(day=1)
        end_date = today
    else:
        return "无效的报告周期"

    logging.info(f"--- 手动生成报告任务 ---")
    logging.info(f"周期: {period}, 时间范围: {start_date} to {end_date}")
    
    # 复用之前agent_tasks中的逻辑，这里简化调用
    from src.app.agent_tasks import DailyScribeAgent # 局部导入
    scribe = DailyScribeAgent()
    
    # 需要修改 DailyScribeAgent 来处理时间段
    report = scribe.generate_period_summary(start_date, end_date)
    logging.info("--- 报告生成完毕 ---")
    return report if report else f"在指定时间范围内没有找到活动记录。"


# --- 问答功能 (增加Debug模式) ---
def agent_answer_stream(query, debug_mode=False):
    if not MASTER_AGENT:
        yield "错误：后端Agent未初始化。"
        return

    # BINGO! Debug日志记录器
    log_stream = []
    def logger_callback(content):
        log_stream.append(content)
        logging.info(f"[Agent Debug] {content.strip()}")

    logging.info(f"--- 开始处理查询: '{query}' (Debug模式: {debug_mode}) ---")
    response_generator = MASTER_AGENT.execute_query(query, logger_callback)
    
    full_response = ""
    for chunk in response_generator:
        full_response += chunk
        
        # BINGO! 如果开启Debug模式，实时返回日志和回答
        if debug_mode:
            debug_info = "```log\n" + "".join(log_stream) + "\n```"
            yield f"**回答:**\n{full_response}▌\n\n---\n**思考过程:**\n{debug_info}"
        else:
            yield full_response + "▌"

    # 最终的完整返回
    logging.info(f"最终回答: {full_response.strip()}")
    logging.info("--- 查询处理完毕 ---")
    if debug_mode:
        debug_info = "```log\n" + "".join(log_stream) + "\n```"
        yield f"**回答:**\n{full_response}\n\n---\n**思考过程:**\n{debug_info}"
    else:
        yield full_response