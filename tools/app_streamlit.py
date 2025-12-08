# tools/app_streamlit.py
import streamlit as st
import sys
import os
import pandas as pd
import altair as alt
from datetime import datetime
import json

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import web_utils

# --- Page Config ---
st.set_page_config(page_title="HearthScribe", page_icon="🏠", layout="wide")

# --- Custom CSS (美化升级) ---
st.markdown("""
<style>
    .stApp { background-color: #f4f6f9; }
    
    /* 指标卡片 */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #eef2f6;
        text-align: center;
    }
    
    /* 洞察横幅 */
    .insight-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .insight-box.ready {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "view_mode" not in st.session_state: st.session_state.view_mode = "gallery"
if "selected_event_id" not in st.session_state: st.session_state.selected_event_id = None

# --- Sidebar ---
with st.sidebar:
    st.title("🏡 HearthScribe")
    st.caption("AI 全维空间感知系统 v2.2")
    
    nav = st.radio("系统导航", ["📊 态势看板", "🎞️ 影像回溯", "📝 报告生成", "🕸️ 认知图谱", "🤖 智能管家"])
    st.markdown("---")
    
    if st.button("🔄 刷新全站数据"):
        st.cache_data.clear()
        st.rerun()

# --- 1. 态势看板 (Dashboard V2) ---
if nav == "📊 态势看板":
    
    # === A. 每日洞察 (自动日报逻辑) ===
    insight = web_utils.get_daily_insight_preview()
    css_class = "ready" if insight['ready'] else ""
    
    st.markdown(f"""
    <div class="insight-box {css_class}">
        <h3>{insight['title']}</h3>
        <p style="white-space: pre-line;">{insight['content']}</p>
    </div>
    """, unsafe_allow_html=True)

    # === B. 核心指标 (8个数据) ===
    st.subheader("📡 核心监控指标")
    stats = web_utils.get_dashboard_stats()
    
    # 第一行：基础状态
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📸 今日事件", f"{stats.get('event_count', 0)}", help="今日检测到的有效活动片段总数")
    c2.metric("🚨 风险告警", f"{stats.get('risk_count', 0)}", delta_color="inverse", help="跌倒/求救/异常事件")
    c3.metric("💤 最大静止", f"{stats.get('max_inactive_min', 0)} min", help="最长连续无人/静止时间")
    c4.metric("👥 家人探访", f"{stats.get('family_count', 0)} 人", help="今日识别到的不同家庭成员/访客数量")
    
    # 第二行：深度分析
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🏃 活跃时长", f"{stats.get('active_hours', 0)} h", help="处于活动状态的总时长")
    c6.metric("🛌 休息时长", f"{stats.get('rest_hours', 0)} h", help="检测为睡觉/躺卧的总时长")
    c7.metric("🤝 高频互动", f"{stats.get('social_count', 0)} 次", help="评分>4的社交或护理事件")
    c8.metric("🧠 新知沉淀", f"{stats.get('new_knowledge', 0)} 条", help="今日新增的知识图谱实体数")
    
    st.markdown("---")
    
    # === C. 数据图表 ===
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown("##### 📈 24小时交互热度 (Time x Score)")
        df_trend = web_utils.get_interaction_trend()
        if not df_trend.empty:
            area = alt.Chart(df_trend).mark_area(
                color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#3498db', offset=1)], x1=1, x2=1, y1=1, y2=0),
                opacity=0.5
            ).encode(x=alt.X('Time', title='时间'), y=alt.Y('Score', title='活跃度'))
            line = alt.Chart(df_trend).mark_line(color='#2980b9').encode(x='Time', y='Score')
            st.altair_chart((area + line).interactive(), use_container_width=True)
        else:
            st.info("数据收集中...")
            
    with chart_col2:
        st.markdown("##### 🍰 场景类型分布")
        df_scene = web_utils.get_scene_distribution()
        if not df_scene.empty:
            base = alt.Chart(df_scene).encode(theta=alt.Theta("Count", stack=True))
            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("Type", scale={"scheme": "pastel1"}),
                order=alt.Order("Count", sort="descending"),
                tooltip=["Type", "Count"]
            )
            text = base.mark_text(radius=140).encode(
                text="Type",
                order=alt.Order("Count", sort="descending"),
                color=alt.value("black")
            )
            st.altair_chart(pie + text, use_container_width=True)
        else:
            st.caption("暂无分类数据")

# --- 2. 影像回溯 (Grid Gallery - 保持不变) ---
elif nav == "🎞️ 影像回溯":
    st.subheader("🎞️ 历史影像归档")
    # ... (保持上一个版本的 Grid 布局代码，不要改动) ...
    # 这里为了代码简洁，请复用上一次回答中的 "elif nav == '🎞️ 影像回溯':" 下面的代码
    # 只需要确保使用 st.columns(4) 或 st.columns(5) 即可
    if st.session_state.view_mode == "detail":
        if st.button("⬅️ 返回列表"):
            st.session_state.view_mode = "gallery"
            st.rerun()
        # 详情页逻辑...
        evt = web_utils.MEMORY.get_rich_event_details([st.session_state.selected_event_id])[0]
        st.info(f"**AI 摘要**: {web_utils.parse_summary(evt['summary'])[0]}")
        # 图片网格
        paths = json.loads(evt['image_paths'])
        if paths:
            cols = st.columns(5)
            for i, p in enumerate(paths):
                if os.path.exists(p): cols[i%5].image(p, caption=f"Frame {i+1}", use_container_width=True)
    else:
        events = web_utils.MEMORY.get_rich_event_details(limit=60)
        cols_count = 4
        for i in range(0, len(events), cols_count):
            cols = st.columns(cols_count)
            for j in range(cols_count):
                if i+j < len(events):
                    evt = events[i+j]
                    with cols[j], st.container(border=True):
                        if evt['preview_image_path']: st.image(evt['preview_image_path'])
                        st.caption(f"{datetime.fromtimestamp(evt['start_time']).strftime('%H:%M')} - {web_utils.parse_summary(evt['summary'])[1]}")
                        if st.button("查看", key=evt['event_id'], use_container_width=True):
                            st.session_state.selected_event_id = evt['event_id']
                            st.session_state.view_mode = "detail"
                            st.rerun()

# --- 3. 报告生成 (保持不变) ---
elif nav == "📝 报告生成":
    st.header("📋 智能日报")
    d = st.date_input("选择日期")
    if st.button("生成/刷新报告"):
        with st.spinner("AI 正在撰写..."):
            rpt = web_utils.generate_daily_report_content(d)
            st.session_state['report_md'] = rpt
    if 'report_md' in st.session_state:
        st.markdown(st.session_state['report_md'])

# --- 4. 认知图谱 (保持不变) ---
elif nav == "🕸️ 认知图谱":
    st.header("🧠 知识图谱")
    st.components.v1.html(web_utils.generate_kg_html(), height=700)

# --- 5. 智能管家 (保持不变) ---
elif nav == "🤖 智能管家":
    st.header("💬 家庭管家")
    # ... (保持上一个版本的 QA 代码) ...
    if q := st.chat_input("输入问题..."):
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"):
            st.write_stream(web_utils.agent_answer_stream(q)) # 简写版，或者用上个版本的复杂版