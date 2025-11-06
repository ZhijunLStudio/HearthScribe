# tools/app_streamlit.py (视图切换终极修正版)
import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import json

# --- 路径设置 & 页面配置 ---
st.set_page_config(page_title="HearthScribe", page_icon="🧠", layout="wide")

# 确保能找到 src 目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import web_utils

# --- 初始化 Session State ---
if 'view' not in st.session_state:
    st.session_state.view = "main"  # main, detail, kg_explorer
if 'selected_event_id' not in st.session_state:
    st.session_state.selected_event_id = None

# --- 主应用 ---
st.title("🧠 HearthScribe: 基于文心大模型的个性化家庭记忆助手")

# --- 侧边栏 ---
with st.sidebar:
    # BINGO! 添加一个返回主页的按钮
    if st.button("🏠 返回主页", use_container_width=True):
        st.session_state.view = "main"
        st.session_state.selected_event_id = None
        st.rerun()

    # BINGO! 添加一个跳转到知识图谱的按钮
    if st.button("🕸️ 探索知识图谱", use_container_width=True):
        st.session_state.view = "kg_explorer"
        st.rerun()

    st.divider()
    st.header("控制面板")
    debug_mode = st.toggle("开启Debug模式", help="开启后，问答区将显示Agent的详细思考过程。")
    st.divider()
    st.header("手动生成报告")
    report_period = st.selectbox("选择报告类型", ["日报", "周报", "月报"])
    if st.button("立即生成报告"):
        with st.spinner(f"正在生成{report_period}..."):
            report_content = web_utils.generate_manual_report(report_period)
            # 将报告直接显示在侧边栏的展开区域中
            with st.expander("查看报告", expanded=True):
                st.markdown(report_content)

# --- BINGO! 视图路由 ---
if st.session_state.view == "main":
    # --- 主视图 ---
    # TOP ROW: 今日洞察 & 核心指标
    with st.container():
        # ... (这部分代码保持不变) ...
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("今日自动总结")
            st.info(web_utils.get_today_summary())
        with col2:
            st.subheader("核心运行指标")
            stats = web_utils.get_dashboard_stats()
            st.metric("今日新记忆", f"{stats['new_memories']} 条")
            st.metric("知识实体总数", f"{web_utils.get_entity_count()} 个")
            st.metric("知识关系总数", f"{web_utils.get_relation_count()} 条")
    st.divider()

    # MIDDLE ROW: 记忆画廊
    st.subheader("记忆画廊 (点击“查看详情”深入探索)")
    recent_events = web_utils.MEMORY.get_rich_event_details(limit=10)
    if not recent_events:
        st.warning("暂无记忆事件。")
    else:
        # ... (画廊的网格布局代码保持不变) ...
        num_columns = 5
        for i in range(0, len(recent_events), num_columns):
            batch = recent_events[i : i + num_columns]
            cols = st.columns(num_columns)
            for j, event in enumerate(batch):
                with cols[j], st.container(border=True):
                    st.image(event['preview_image_path'], use_container_width=True)
                    st.caption(f"_{datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M')}_")
                    st.markdown(f"<p style='height: 60px; overflow: hidden; font-size: 14px;'>{event['summary']}</p>", unsafe_allow_html=True)
                    if st.button("查看详情", key=f"btn_{event['event_id']}", use_container_width=True):
                        st.session_state.selected_event_id = event['event_id']
                        st.session_state.view = "detail"
                        st.rerun()
    st.divider()

    # BOTTOM ROW: 智能问答
    st.subheader("💬 智能问答")
    # ... (问答区域的代码保持不变) ...
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"], unsafe_allow_html=True)
    if prompt := st.chat_input("向我提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for chunk in web_utils.agent_answer_stream(prompt, debug_mode=debug_mode):
                full_response = chunk
                placeholder.markdown(full_response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

elif st.session_state.view == "detail":
    # --- 事件详情视图 ---
    st.subheader("🔍 事件详情")
    # (这部分代码与之前在Tab里的完全一样)
    event_details = web_utils.MEMORY.get_rich_event_details([st.session_state.selected_event_id])[0]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**事件ID**: `{event_details['event_id']}`")
        st.markdown(f"**发生时间**: `{datetime.fromtimestamp(event_details['start_time']).strftime('%Y-%m-%d %H:%M:%S')}`")
        st.info(f"**AI 摘要**: {event_details['summary']}")
        st.subheader("知识图谱片段")
        kg_fragment = web_utils.MEMORY.get_kg_for_event(event_details['event_id'])
        st.dataframe(pd.DataFrame(kg_fragment), use_container_width=True)
    with col2:
        st.subheader("事件帧画廊")
        image_paths = json.loads(event_details.get('image_paths', '[]'))
        if image_paths: st.image(image_paths, width=150)
        else: st.warning("此事件没有关联的帧图像。")

elif st.session_state.view == "kg_explorer":
    # --- 知识图谱浏览器视图 ---
    st.subheader("🕸️ 知识图谱浏览器")
    # (这部分代码与之前在Tab里的完全一样)
    col1, col2 = st.columns([1, 3])
    with col1:
        limit = st.slider("加载关系数量", 50, 1000, 200)
        all_entities_df = pd.DataFrame(web_utils.MEMORY.get_all_kg_data(limit=1000))
        if not all_entities_df.empty:
            all_nodes = sorted(list(pd.concat([all_entities_df['source'], all_entities_df['target']]).unique()))
            focused_entity = st.selectbox("高亮实体", options=[""] + all_nodes)
        else:
            focused_entity = ""
    with col2:
        with st.spinner("正在构建神经网络..."):
            graph_html = web_utils.generate_knowledge_graph_html(limit=limit, focused_entity=focused_entity)
            st.components.v1.html(graph_html, height=750, scrolling=False)