import streamlit as st
import sys
import os
import pandas as pd
import streamlit.components.v1 as components

# --- 路径设置 ---
# 确保无论从哪里运行 streamlit，都能找到 src 目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# BINGO! 必须在 set_page_config() 之后再导入 web_utils
st.set_page_config(page_title="HearthScribe", page_icon="🔥", layout="wide")

# 现在导入是安全的
from src import web_utils

st.title("🔥 HearthScribe 智能中枢")

# --- 侧边栏状态 ---
with st.sidebar:
    st.header("系统状态")
    # 尝试获取实例来检查连接状态
    if web_utils.get_memory_instance() and web_utils.get_llm_client():
        st.success("后端服务已连接")
    else:
        st.error("后端服务连接失败")
    
    if st.button("🔄 刷新数据"):
        st.cache_data.clear()
        st.rerun()

# --- 主界面 Tab ---
tab1, tab2, tab3 = st.tabs(["💬 记忆问答", "🕸️ 知识图谱", "📋 事件流"])

# === Tab 1: 问答 ===
with tab1:
    st.subheader("向记忆提问")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    
    if prompt := st.chat_input("lizhijun 最近在喝水吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
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
        components.html(graph_html, height=650, scrolling=False) # scrolling=False 更好看

# === Tab 3: 事件流 ===
with tab3:
    st.subheader("最近的事件记录")
    df = web_utils.get_recent_events_df()
    
    if not df.empty:
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Preview": st.column_config.ImageColumn("事件预览", help="事件的第一帧快照"),
                "Summary": st.column_config.TextColumn("AI 摘要", width="large")
            }
        )
    else:
        st.warning("暂无事件记录。请先运行 main_collector.py。")