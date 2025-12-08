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

# Path setup...
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.memory.long_term_memory import LongTermMemory
from src.agent.master_agent import MasterAgent

logger = logging.getLogger(__name__)

# --- 单例 ---
_memory_instance = None
_master_agent_instance = None

def get_memory_instance():
    global _memory_instance
    if _memory_instance is None:
        try:
            _memory_instance = LongTermMemory(config.LANCEDB_PATH, config.SQLITE_DB_PATH)
        except Exception as e:
            logger.error(f"Init Memory Failed: {e}")
    return _memory_instance

def get_master_agent():
    global _master_agent_instance
    if _master_agent_instance is None:
        mem = get_memory_instance()
        if mem: _master_agent_instance = MasterAgent(mem)
    return _master_agent_instance

MEMORY = get_memory_instance()
MASTER_AGENT = get_master_agent()

# --- 辅助函数 ---
def parse_summary(raw_summary):
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

# --- 核心：8大指标获取 ---
def get_dashboard_stats():
    """获取看板所需的 8 个核心指标"""
    if not MEMORY: return {}
    
    # 初始化 8 个指标
    stats = {
        "event_count": 0,       # 1. 今日事件数
        "risk_count": 0,        # 2. 风险预警数
        "active_hours": 0.0,    # 3. 活跃时长
        "rest_hours": 0.0,      # 4. 休息时长
        "max_inactive_min": 0,  # 5. 最大静止间隔
        "social_count": 0,      # 6. 社交/高频互动数
        "family_count": 0,      # 7. 家人探访(去重人数)
        "new_knowledge": 0,     # 8. 今日新知(实体数)
        "last_active": "--:--"
    }
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    
    try:
        with MEMORY.db_lock:
            cursor = MEMORY.sqlite_conn.cursor()
            
            # --- 查询 A: 事件相关 ---
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
                    
                    # 统计逻辑
                    if "风险" in label or "跌倒" in label or score >= 8:
                        stats["risk_count"] += 1
                    
                    if "躺" in text or "睡" in text or "休息" in label:
                        stats["rest_hours"] += duration
                    else:
                        stats["active_hours"] += duration
                        
                    if score >= 4: # 假设评分>4算作有互动的社交/护理
                        stats["social_count"] += 1
                        
                    # 静止间隔
                    if i > 0:
                        gap = start - last_end
                        if gap > max_gap: max_gap = gap
                    last_end = end
                
                # 补算当前时刻的静止
                curr_gap = datetime.now().timestamp() - rows[-1][1]
                if curr_gap > max_gap: max_gap = curr_gap
                
                stats["max_inactive_min"] = int(max_gap / 60)
                stats["active_hours"] = round(stats["active_hours"] / 3600, 1)
                stats["rest_hours"] = round(stats["rest_hours"] / 3600, 1)
                stats["last_active"] = datetime.fromtimestamp(rows[-1][0]).strftime("%H:%M")

            # --- 查询 B: 家人/实体相关 ---
            # 统计今日涉及的非Unknown人物数量
            cursor.execute("""
                SELECT COUNT(DISTINCT e.name) 
                FROM entities e
                JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
                JOIN events ev ON r.event_id = ev.event_id
                WHERE ev.start_time >= ? AND e.type = 'Person' AND e.name NOT LIKE '%Unknown%'
            """, (today_start,))
            stats["family_count"] = cursor.fetchone()[0]

            # 统计今日新增实体(简单用关联事件时间估算)
            # 这里简化逻辑：统计今日事件关联的所有实体数
            cursor.execute("""
                SELECT COUNT(DISTINCT e.id)
                FROM entities e
                JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
                JOIN events ev ON r.event_id = ev.event_id
                WHERE ev.start_time >= ?
            """, (today_start,))
            stats["new_knowledge"] = cursor.fetchone()[0]
                
    except Exception as e:
        logger.error(f"Stats Error: {e}")
        
    return stats

def get_daily_insight_preview():
    """获取首页顶部的今日洞察摘要"""
    now = datetime.now()
    
    # 逻辑：22点前显示观察中
    if now.hour < 22:
        return {
            "ready": False,
            "title": "👁️ 空间态势观察中...",
            "content": f"AI 正在持续分析今日活动。完整日报将于今晚 **22:00** 自动生成。\n目前系统运行正常，已记录 {get_dashboard_stats().get('event_count', 0)} 个事件片段。"
        }
    
    # 22点后：如果没有生成过，现算一个简短的；如果想看详细的去报告页
    stats = get_dashboard_stats()
    summary = f"""
    **📅 今日概览 (自动生成)**
    
    截止目前，系统共记录了 **{stats['event_count']}** 个活动片段。
    重点数据如下：
    - 活跃时长: {stats['active_hours']} 小时
    - 休息时长: {stats['rest_hours']} 小时
    - 风险预警: {stats['risk_count']} 次
    
    建议您点击左侧 **[📝 报告生成]** 查看包含详细时间线的完整分析报告。
    """
    return {
        "ready": True,
        "title": "✅ 今日日报已就绪",
        "content": summary
    }

def get_interaction_trend():
    """获取交互热度曲线 (Area Chart)"""
    if not MEMORY: return pd.DataFrame()
    today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
    
    with MEMORY.db_lock:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT start_time, summary FROM events WHERE start_time >= ? ORDER BY start_time", (today_start,))
        rows = cursor.fetchall()
        
    data = []
    for r in rows:
        _, _, score = parse_summary(r[1])
        data.append({
            "Time": datetime.fromtimestamp(r[0]).strftime("%H:%M"),
            "Score": score
        })
    return pd.DataFrame(data)

def get_scene_distribution():
    """获取场景分布 (Pie Chart)"""
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
    
    # 统计并转为 DataFrame
    from collections import Counter
    counts = Counter(labels)
    return pd.DataFrame([{"Type": k, "Count": v} for k, v in counts.items()])


def get_person_frequency():
    """获取人员出现频率 (DataFrame)"""
    if not MEMORY: return pd.DataFrame()
    today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
    
    with MEMORY.db_lock:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("""
            SELECT e.name FROM entities e
            JOIN relationships r ON e.id = r.source_id OR e.id = r.target_id
            JOIN events ev ON r.event_id = ev.event_id
            WHERE ev.start_time >= ? AND e.type = 'Person' AND e.name != 'Unknown_Body'
        """, (today_start,))
        rows = cursor.fetchall()
    
    names = [r[0] for r in rows]
    if not names: return pd.DataFrame()
    
    df = pd.DataFrame(names, columns=["Name"])
    return df["Name"].value_counts().reset_index(name="Count").rename(columns={"index": "Name"})

def get_system_stats():
    """获取系统硬指标"""
    if not MEMORY: return {"memory": 0, "entities": 0, "care_hours": "0.0h"}
    
    with MEMORY.db_lock:
        cursor = MEMORY.sqlite_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        mem = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM entities")
        ent = cursor.fetchone()[0]
        cursor.execute("SELECT SUM(end_time - start_time) FROM events")
        row = cursor.fetchone()
        total_sec = row[0] if row[0] else 0
        
    return {
        "memory": mem,
        "entities": ent,
        "care_hours": f"{total_sec/3600:.1f}h"
    }

# --- 问答 & 报告 & 图谱 ---

def agent_answer_stream(query):
    """流式返回 Agent 回答"""
    if not MASTER_AGENT:
        yield "System Initializing..."
        return
    # 使用 execute_query_steps，而不是不存在的 execute_query
    gen = MASTER_AGENT.execute_query_steps(query, lambda x: None)
    for step in gen:
        if step['status'] in ['generating', 'done']:
            yield step['content']

def generate_daily_report_content(date_obj=None):
    """
    生成真正的 AI 叙述性日报
    """
    if not MEMORY: return "系统未初始化"
    if not date_obj: date_obj = datetime.now()
    
    start_ts = datetime.combine(date_obj, datetime.min.time()).timestamp()
    end_ts = datetime.combine(date_obj, datetime.max.time()).timestamp()
    
    # 1. 获取原始数据
    events = MEMORY.get_events_for_period(start_ts, end_ts)
    if not events: return f"# 📅 {date_obj.strftime('%Y-%m-%d')} 报告\n\n今日无活动记录。"

    # 2. 构建上下文给大模型
    context_lines = []
    for e in events:
        t = datetime.fromtimestamp(e['start_time']).strftime('%H:%M')
        txt, label, _ = parse_summary(e['summary'])
        context_lines.append(f"- [{t}] ({label}) {txt}")
    
    context_str = "\n".join(context_lines)
    
    # 3. 调用 MasterAgent 生成报告 (或者直接调 OpenAI)
    prompt = f"""
    你是一名专业的家庭看护助理。请根据以下今日的活动流水，写一份 Markdown 格式的日报。
    
    要求：
    1. 使用一级标题：📅 {date_obj.strftime('%Y-%m-%d')} 看护日报
    2. 包含“今日概览”、“详细活动”、“风险提示”三个板块。
    3. 语气温暖、客观。
    4. 重点关注老人的活动频率、休息情况和是否有异常。
    
    【活动流水】：
    {context_str}
    """
    
    try:
        if MASTER_AGENT:
            # 复用 MasterAgent 的 client
            resp = MASTER_AGENT.llm_client.chat.completions.create(
                model=config.AI_THINKING_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return resp.choices[0].message.content
        else:
            return "Agent 未就绪，无法生成报告。"
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        return f"报告生成出错: {e}\n\n原始数据:\n{context_str}"

def generate_kg_html():
    """生成增强版知识图谱 HTML"""
    if not MEMORY: return "<div>Memory Module Error</div>"
    
    # 获取数据
    relations = MEMORY.get_all_kg_data(limit=500)
    
    # 如果数据为空，返回提示
    if not relations:
        return """
        <div style='text-align:center; padding: 50px; color: #666;'>
            <h3>🕸️ 暂无知识图谱数据</h3>
            <p>请先运行系统并等待事件分析完成。</p>
        </div>
        """
    
    G = nx.DiGraph()
    for r in relations:
        src = r.get('source', '未知').strip()
        tgt = r.get('target', '未知').strip()
        rel = r.get('relation', '关联').strip()
        
        if not src or not tgt: continue
        
        # 根据类型设置颜色
        src_type = r.get('source_type', 'Object')
        tgt_type = r.get('target_type', 'Object')
        
        color_map = {
            "Person": "#ff7675", # 红
            "Object": "#74b9ff", # 蓝
            "Location": "#55efc4", # 绿
            "Activity": "#ffeaa7"  # 黄
        }
        
        G.add_node(src, label=src, color=color_map.get(src_type, "#dfe6e9"), title=src_type)
        G.add_node(tgt, label=tgt, color=color_map.get(tgt_type, "#dfe6e9"), title=tgt_type)
        G.add_edge(src, tgt, label=rel, color="#b2bec3")
    
    net = Network(height="650px", width="100%", notebook=False, cdn_resources='remote', directed=True)
    net.from_nx(G)
    
    # 物理引擎配置：防止爆炸和空白
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 14, "face": "arial" }, "borderWidth": 1 },
      "edges": { "font": { "size": 10, "align": "middle" }, "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } } },
      "physics": { 
          "enabled": true,
          "forceAtlas2Based": { "gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "damping": 0.4 },
          "solver": "forceAtlas2Based",
          "stabilization": { "enabled": true, "iterations": 200 } 
      }
    }
    """)
    
    return net.generate_html(name='kg.html', local=False)