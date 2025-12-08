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

# --- 页面配置 ---
st.set_page_config(page_title="HearthScribe", page_icon="🏠", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .event-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        background: white;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .event-card:hover { transform: scale(1.02); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- Session State 初始化 ---
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "gallery" # 'gallery' or 'detail'
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None

# --- 侧边栏 ---
with st.sidebar:
    st.title("HearthScribe")
    # 如果在详情页，显示返回按钮
    if st.session_state.view_mode == "detail":
        if st.button("⬅️ 返回列表", use_container_width=True):
            st.session_state.view_mode = "gallery"
            st.session_state.selected_event = None
            st.rerun()
            
    nav = st.radio("导航", ["🏠 态势看板", "📽️ 影像回溯", "📝 报告生成", "🕸️ 认知图谱", "🤖 智能助手"])
    st.markdown("---")
    
    # 强制刷新报告按钮
    if nav == "📝 报告生成":
        if st.button("🔄 强制刷新数据"):
            st.cache_data.clear()
            st.rerun()

# --- 1. 态势看板 (保持不变) ---
if nav == "🏠 态势看板":
    # ... (你的原代码，不需要改动) ...
    st.header("☀️ 今日空间态势")
    data = web_utils.get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("⚠️ 风险预警", f"{data.get('risk_count',0)}", "次")
    with c2: st.metric("⏱️ 最大静止", f"{data.get('max_inactive_min',0)}", "分钟")
    with c3: st.metric("🛌 休息时长", f"{data.get('rest_hours',0)}", "小时")
    with c4: st.metric("📸 今日事件", f"{data.get('event_count',0)}", f"最新: {data.get('last_active','--')}")
    st.markdown("---")
    
    c_main, c_side = st.columns([2, 1])
    with c_main:
        st.subheader("📈 交互活跃度趋势")
        df_trend = web_utils.get_interaction_trend()
        if not df_trend.empty:
            chart = alt.Chart(df_trend).mark_area(
                line={'color':'#3498db'},
                color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#3498db', offset=1)], x1=1, x2=1, y1=1, y2=0)
            ).encode(x=alt.X('Time', title='时刻'), y=alt.Y('Score', title='活跃度'), tooltip=['Time', 'Score']).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
        else: st.info("等待数据积累...")
    with c_side:
        st.subheader("🍰 场景分布")
        df_scene = web_utils.get_scene_distribution()
        if not df_scene.empty:
            st.altair_chart(alt.Chart(df_scene).mark_arc(innerRadius=50).encode(theta="Count", color="Type", tooltip=["Type", "Count"]), use_container_width=True)
        else: st.caption("暂无数据")

# --- 2. 影像回溯 (重构为画廊模式) ---
elif nav == "📽️ 影像回溯":
    st.header("📅 历史影像归档")
    
    # === 详情视图 ===
    if st.session_state.view_mode == "detail" and st.session_state.selected_event:
        evt = st.session_state.selected_event
        txt, lbl, score = web_utils.parse_summary(evt['summary'])
        
        col_btn, _ = st.columns([1, 5])
        
        st.subheader(f"事件详情: {datetime.fromtimestamp(evt['start_time']).strftime('%H:%M:%S')}")
        
        c_info, c_imgs = st.columns([1, 2])
        with c_info:
            st.markdown(f"**场景标签**: `{lbl}`")
            st.markdown(f"**活跃评分**: `{score}`")
            st.info(f"**AI 摘要**: \n\n{txt}")
            st.divider()
            st.caption(f"Event ID: {evt['event_id']}")
            
        with c_imgs:
            try:
                paths = json.loads(evt['image_paths'])
                if paths:
                    st.write(f"共包含 {len(paths)} 帧画面：")
                    # 使用 expander 或者直接列出
                    for i, p in enumerate(paths):
                        if os.path.exists(p):
                            st.image(p, caption=f"Frame {i+1}", use_container_width=True)
                        else:
                            st.warning(f"图片丢失: {p}")
            except:
                st.error("图片数据解析失败")
                
    # === 画廊列表视图 ===
    else:
        events = web_utils.MEMORY.get_rich_event_details(limit=50)
        if not events:
            st.info("暂无记录")
        else:
            # 网格布局：每行 4 列
            cols_per_row = 4
            for i in range(0, len(events), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(events):
                        evt = events[i + j]
                        with cols[j]:
                            # 卡片容器
                            with st.container(border=True):
                                # 预览图
                                if evt['preview_image_path'] and os.path.exists(evt['preview_image_path']):
                                    st.image(evt['preview_image_path'], use_container_width=True)
                                else:
                                    st.markdown("🖼️ _No Image_")
                                
                                # 时间和摘要
                                t_str = datetime.fromtimestamp(evt['start_time']).strftime('%H:%M')
                                txt, _, _ = web_utils.parse_summary(evt['summary'])
                                st.markdown(f"**{t_str}**")
                                st.caption(f"{txt[:20]}...")
                                
                                # 详情按钮 (回调函数模式)
                                if st.button("查看", key=f"btn_{evt['event_id']}", use_container_width=True):
                                    st.session_state.selected_event = evt
                                    st.session_state.view_mode = "detail"
                                    st.rerun()

# --- 3. 报告生成 (修复) ---
elif nav == "📝 报告生成":
    st.header("📋 智能报告中心")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        d = st.date_input("选择日期", datetime.now())
        if st.button("🚀 生成 AI 分析报告", use_container_width=True):
            with st.spinner("正在请求大模型生成分析报告..."):
                report_md = web_utils.generate_daily_report_content(d)
                st.session_state['report_md'] = report_md
                
    with col2:
        if 'report_md' in st.session_state:
            st.markdown(st.session_state['report_md'])
            st.download_button("📥 下载 Markdown", st.session_state['report_md'], f"report_{d}.md")

# --- 4. 认知图谱 (修复) ---
elif nav == "🕸️ 认知图谱":
    st.header("🧠 空间认知网络")
    with st.spinner("正在构建图谱..."):
        html = web_utils.generate_kg_html()
        st.components.v1.html(html, height=700, scrolling=True)

# --- 5. 智能助手 (保持不变) ---
elif nav == "🤖 智能助手":
    st.header("💬 关怀问答")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    for role, text in st.session_state.chat_history:
        with st.chat_message(role): st.markdown(text)
    if q := st.chat_input("请输入问题..."):
        st.session_state.chat_history.append(("user", q))
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            ph = st.empty()
            full = ""
            for chunk in web_utils.agent_answer_stream(q):
                full = chunk
                ph.markdown(full + "▌")
            ph.markdown(full)
            st.session_state.chat_history.append(("assistant", full))