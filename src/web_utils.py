# src/web_utils.py
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import json
import re
import networkx as nx
from pyvis.network import Network

# 路径设置
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.memory.long_term_memory import LongTermMemory
from src.agent.master_agent import MasterAgent

logger = logging.getLogger(__name__)

# --- 单例模式实例化 ---
_memory_instance = None
_master_agent_instance = None

def get_memory_instance():
    global _memory_instance
    if _memory_instance is None:
        try:
            logger.info("初始化 LongTermMemory...")
            _memory_instance = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        except Exception as e:
            logger.error(f"Memory Init Failed: {e}", exc_info=True)
    return _memory_instance

def get_master_agent():
    global _master_agent_instance
    if _master_agent_instance is None:
        mem = get_memory_instance()
        if mem:
            logger.info("初始化 MasterAgent...")
            _master_agent_instance = MasterAgent(mem)
    return _master_agent_instance

# 全局实例
MEMORY = get_memory_instance()
MASTER_AGENT = get_master_agent()

# --- 辅助函数 ---
def parse_summary(raw_summary):
    """解析摘要字符串，提取 Label 和 Score"""
    if not raw_summary: return "", "日常", 0
    parts = raw_summary.split("|||")
    text = parts[0]
    label = "日常"
    score = 0
    for p in parts:
        if p.startswith("LABEL:"): label = p.replace("LABEL:", "")
        if p.startswith("SCORE:"): 
            try: score = int(p.replace("SCORE:", ""))
            except: pass
    return text, label, score

# --- 核心数据统计 (Dashboard) ---
def get_dashboard_stats():
    """获取看板所需的 8 个核心指标"""
    if not MEMORY: return {}
    
    stats = {
        "event_count": 0, "risk_count": 0, "active_hours": 0.0, "rest_hours": 0.0,
        "max_inactive_min": 0, "social_count": 0, "family_count": 0, "new_knowledge": 0,
        "last_active": "--:--"
    }
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    
    try:
        with MEMORY.db_lock:
            cursor = MEMORY.sqlite_conn.cursor()
            
            # 1. 事件统计
            cursor.execute("SELECT start_time, end_time, summary FROM events WHERE start_time >= ? ORDER BY start_time", (today_start,))
            rows = cursor.fetchall()
            stats["event_count"] = len(rows)
            
            if rows:
                max_gap = 0
                last_end = rows[0][1]
                for i, r in enumerate(rows):
                    start, end, summary = r[0], r[1], r[2]
                    text, label, score = parse_summary(summary)
                    duration = end - start
                    
                    if "风险" in label or "跌倒" in label or score >= 8: stats["risk_count"] += 1
                    if "躺" in text or "睡" in text or "休息" in label: stats["rest_hours"] += duration
                    else: stats["active_hours"] += duration
                    if score >= 4: stats["social_count"] += 1
                        
                    if i > 0:
                        gap = start - last_end
                        if gap > max_gap: max_gap = gap
                    last_end = end
                
                curr_gap = datetime.now().timestamp() - rows[-1][1]
                if curr_gap > max_gap: max_gap = curr_gap
                
                stats["max_inactive_min"] = int(max_gap / 60)
                stats["active_hours"] = round(stats["active_hours"] / 3600, 1)
                stats["rest_hours"] = round(stats["rest_hours"] / 3600, 1)
                stats["last_active"] = datetime.fromtimestamp(rows[-1][0]).strftime("%H:%M")

            # 2. 家人统计 (非 Unknown)
            cursor.execute("""
                SELECT COUNT(DISTINCT e.name) FROM entities e
                JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
                JOIN events ev ON r.event_id = ev.event_id
                WHERE ev.start_time >= ? AND e.type = 'Person' AND e.name NOT LIKE '%Unknown%'
            """, (today_start,))
            stats["family_count"] = cursor.fetchone()[0]

            # 3. 新知统计
            cursor.execute("""
                SELECT COUNT(DISTINCT e.id) FROM entities e
                JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
                JOIN events ev ON r.event_id = ev.event_id
                WHERE ev.start_time >= ?
            """, (today_start,))
            stats["new_knowledge"] = cursor.fetchone()[0]
                
    except Exception as e:
        logger.error(f"Stats Error: {e}")
        
    return stats

def get_daily_insight_preview():
    """首页每日洞察逻辑 (注意：这里用 HTML 标签 <b> 实现加粗)"""
    now = datetime.now()
    if now.hour < 22:
        return {
            "ready": False,
            "title": "👁️ 空间态势观察中...",
            "content": f"AI 正在持续分析今日活动。完整日报将于今晚 <b>22:00</b> 自动生成。\n目前已记录 <b>{get_dashboard_stats().get('event_count', 0)}</b> 个事件片段。"
        }
    
    stats = get_dashboard_stats()
    summary = f"""
    <b>📅 今日日报已就绪</b><br>
    截止目前，记录了 {stats['event_count']} 个活动片段。风险告警 {stats['risk_count']} 次。<br>
    建议点击左侧 <b>[📝 报告生成]</b> 查看深度分析。
    """
    return {"ready": True, "title": "✅ 今日日报已就绪", "content": summary}

def get_interaction_trend():
    """交互热度数据"""
    if not MEMORY: return pd.DataFrame()
    today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
    with MEMORY.db_lock:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT start_time, summary FROM events WHERE start_time >= ? ORDER BY start_time", (today_start,))
        rows = cursor.fetchall()
    data = []
    for r in rows:
        _, _, score = parse_summary(r[1])
        data.append({"Time": datetime.fromtimestamp(r[0]).strftime("%H:%M"), "Score": score})
    return pd.DataFrame(data)

def get_scene_distribution():
    """场景分布数据"""
    if not MEMORY: return pd.DataFrame()
    today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
    with MEMORY.db_lock:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT summary FROM events WHERE start_time >= ?", (today_start,))
        rows = cursor.fetchall()
    labels = []
    for r in rows:
        _, label, _ = parse_summary(r[0])
        labels.append(label)
    if not labels: return pd.DataFrame()
    from collections import Counter
    return pd.DataFrame([{"Type": k, "Count": v} for k, v in Counter(labels).items()])

def agent_answer_stream(query):
    """流式问答透传"""
    if not MASTER_AGENT:
        yield {"status": "answer", "content": "⚠️ 系统未就绪"}
        return
    try:
        gen = MASTER_AGENT.execute_query_steps(query)
        for step in gen: yield step
    except Exception as e:
        yield {"status": "answer", "content": f"⚠️ 错误: {e}"}

def generate_daily_report_content(date_obj=None):
    """
    生成叙述性日报 (Prompt 升级版)
    """
    if not MEMORY: return "No Data"
    if not date_obj: date_obj = datetime.now()
    start_ts = datetime.combine(date_obj, datetime.min.time()).timestamp()
    end_ts = datetime.combine(date_obj, datetime.max.time()).timestamp()
    
    events = MEMORY.get_events_for_period(start_ts, end_ts)
    if not events: return f"# 📅 {date_obj.strftime('%Y-%m-%d')} 报告\n\n当日无记录。"
    
    # 构建流水
    context_lines = []
    for e in events:
        t = datetime.fromtimestamp(e['start_time']).strftime('%H:%M')
        txt = e['summary'].split('|||')[0]
        context_lines.append(f"- [{t}] {txt}")
    context_str = "\n".join(context_lines)
    
    # --- 升级后的 Prompt：强制要求综合叙述 ---
    prompt = f"""
    你是一位资深的家庭健康管理顾问。请阅读以下监控流水，为用户撰写一份【叙述性】的深度日报。
    
    【元数据】
    日期：{date_obj.strftime('%Y-%m-%d')}
    事件流：
    {context_str}
    
    【撰写要求】
    1. **不要**按时间流水账罗列（严禁出现“15:00... 16:00...”这种列表格式）。
    2. 请将一天划分为“上午”、“下午”、“晚间”等自然段落进行连贯的叙述。
    3. 报告结构如下：
       - **📝 今日生活画像**：用一段优美的文字概括老人今天的主要活动轨迹和精神状态。
       - **🩺 健康深度评估**：分析其运动量、休息规律、喝水频率、久坐情况等。
       - **⚠️ 异常风险检测**：明确指出是否存在跌倒风险、未关门窗、陌生人进入等安全隐患。
       - **💡 专属关怀建议**：给出暖心的、可执行的生活建议。
    4. 语气要温暖、专业、体现对长者的关怀。
    
    请直接输出 Markdown 内容。
    """
    
    try:
        if MASTER_AGENT:
            resp = MASTER_AGENT.llm_client.chat.completions.create(
                model=config.AI_THINKING_MODEL, 
                messages=[{"role": "user", "content": prompt}], 
                temperature=0.6 # 稍微提高温度，让文笔更好
            )
            return resp.choices[0].message.content
        return "Agent 未就绪。"
    except Exception as e:
        return f"生成出错: {e}"

def generate_kg_html():
    """生成带开场动画的知识图谱"""
    if not MEMORY: return "<div>No Data</div>"
    relations = MEMORY.get_all_kg_data(limit=500)
    if not relations: return "<div style='text-align:center;padding:50px;color:#666'>暂无知识图谱数据</div>"
    
    G = nx.DiGraph()
    # 配色优化
    color_map = {
        "Person": "#FF6B6B",   # 暖红
        "Object": "#4ECDC4",   # 青绿
        "Location": "#FFE66D", # 亮黄
        "Activity": "#1A535C"  # 深蓝
    }
    
    for r in relations:
        src, tgt = r.get('source', 'U'), r.get('target', 'U')
        G.add_node(src, label=src, color=color_map.get(r.get('source_type'), "#f7f1e3"), title=r.get('source_type'))
        G.add_node(tgt, label=tgt, color=color_map.get(r.get('target_type'), "#f7f1e3"), title=r.get('target_type'))
        G.add_edge(src, tgt, label=r.get('relation', '-'), color="#bdc3c7")
    
    net = Network(height="750px", width="100%", notebook=False, cdn_resources='remote', directed=True)
    net.from_nx(G)
    
    # --- 关键：物理引擎设置 ---
    # stabilization: false -> 意味着一开始不计算稳定状态，直接展示从混乱到有序的动画过程
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 16, "face": "tahoma" }, "borderWidth": 2, "shadow": true },
      "edges": { "width": 1, "smooth": { "type": "continuous" }, "arrows": { "to": { "scaleFactor": 0.5 } } },
      "physics": { 
          "enabled": true,
          "forceAtlas2Based": { "gravitationalConstant": -60, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08, "damping": 0.4 },
          "maxVelocity": 50,
          "solver": "forceAtlas2Based",
          "stabilization": { "enabled": false } 
      }
    }
    """)
    return net.generate_html(name='kg.html', local=False)