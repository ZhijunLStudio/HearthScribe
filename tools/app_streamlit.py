import streamlit as st
import sys
import os
import pandas as pd
import streamlit.components.v1 as components
from datetime import datetime, timedelta

# --- 路径设置 & 页面配置 ---
# 确保无论从哪里运行 streamlit，都能找到 src 目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# BINGO! set_page_config() 必须是第一个 Streamlit 命令
st.set_page_config(page_title="HearthScribe", page_icon="🔥", layout="wide")

# 在页面配置之后，再安全地导入我们的后端工具
from src import web_utils

st.title("🔥 HearthScribe 智能中枢")

# --- 侧边栏 ---
with st.sidebar:
    st.header("系统状态")
    # 尝试获取实例来检查连接状态
    if web_utils.get_memory_instance() and web_utils.get_llm_client():
        st.success("后端服务已连接")
    else:
        st.error("后端服务连接失败")
    
    if st.button("🔄 刷新数据"):
        # 清除所有缓存，强制重新加载
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# --- 主界面 Tab ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 记忆问答", "🕸️ 知识图谱", "📋 事件流", "📊 总结报告"])

# === Tab 1: 问答 ===
with tab1:
    st.subheader("向记忆提问")
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("lizhijun 最近在喝水吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(web_utils.answer_question(prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})

# === Tab 2: 知识图谱 ===
with tab2:
    st.subheader("全域知识图谱 (可交互)")
    st.caption("这是 Agent“脑中”所有实体和关系的动态可视化。你可以拖动节点、缩放查看。")
    limit = st.slider("显示最新的关系数量", 50, 500, 150)
    
    with st.spinner("正在构建神经网络..."):
        graph_html = web_utils.generate_knowledge_graph_html(limit=limit)
        components.html(graph_html, height=650, scrolling=False)

# === Tab 3: 事件流 (修复图片破图) ===
with tab3:
    st.subheader("最近的事件记录")
    df = web_utils.get_recent_events_df()
    
    if not df.empty:
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Preview": st.column_config.ImageColumn(
                    "事件预览", help="事件的第一帧快照"
                ),
                "Summary": st.column_config.TextColumn(
                    "AI 摘要", width="large"
                )
            },
            # 动态调整高度以更好地显示图片
            height=(len(df) + 1) * 100 if len(df) < 8 else 800 
        )
    else:
        st.warning("暂无事件记录。请先运行 main_collector.py。")

# === Tab 4: 总结报告 (新增) ===
with tab4:
    st.subheader("生成生活洞察报告")
    st.caption("选择一个时间范围，让 AI 为你分析和总结这段时间的生活点滴。")

    # --- 时间范围选择 ---
    today = datetime.now().date()
    # 使用 session_state 来保存日期选择，避免快捷按钮后状态丢失
    if 'start_date' not in st.session_state:
        st.session_state.start_date = today - timedelta(days=6)
    if 'end_date' not in st.session_state:
        st.session_state.end_date = today

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.start_date = st.date_input("开始日期", st.session_state.start_date)
    with col2:
        st.session_state.end_date = st.date_input("结束日期", st.session_state.end_date)

    # --- 快捷按钮 ---
    st.write("快捷选择：")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    if b_col1.button("今日报告", use_container_width=True):
        st.session_state.start_date = today
        st.session_state.end_date = today
        st.rerun()

    if b_col2.button("昨日报告", use_container_width=True):
        st.session_state.start_date = today - timedelta(days=1)
        st.session_state.end_date = today - timedelta(days=1)
        st.rerun()

    if b_col3.button("本周报告", use_container_width=True):
        st.session_state.start_date = today - timedelta(days=today.weekday())
        st.session_state.end_date = today
        st.rerun()
        
    if b_col4.button("本月报告", use_container_width=True):
        st.session_state.start_date = today.replace(day=1)
        st.session_state.end_date = today
        st.rerun()

    # --- 生成按钮和报告显示区域 ---
    st.divider()
    if st.button("🚀 生成分析报告", type="primary", use_container_width=True):
        if st.session_state.start_date > st.session_state.end_date:
            st.error("错误：开始日期不能晚于结束日期。")
        else:
            with st.spinner(f"正在分析从 {st.session_state.start_date} 到 {st.session_state.end_date} 的记忆，请稍候..."):
                report = web_utils.generate_summary_report(st.session_state.start_date, st.session_state.end_date)
            
            st.markdown("---")
            st.subheader("你的专属生活报告")
            st.markdown(report, unsafe_allow_html=True) # 允许HTML以便Markdown表格等能正确渲染