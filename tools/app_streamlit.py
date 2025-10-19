import streamlit as st
import sys
import os
import pandas as pd
import streamlit.components.v1 as components # 用于渲染 HTML

# --- 路径设置 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import web_utils

st.set_page_config(page_title="HearthScribe", page_icon="🔥", layout="wide")
st.title("🔥 HearthScribe 智能中枢")

# --- 侧边栏状态 ---
with st.sidebar:
    st.header("系统状态")
    if web_utils.MEMORY:
        st.success("后端服务已连接")
    else:
        st.error("后端服务连接失败")
    
    if st.button("刷新数据"):
        st.rerun()

# --- 主界面 Tab ---
tab1, tab2, tab3 = st.tabs(["💬 记忆问答", "🕸️ 知识图谱", "📋 事件流"])

# === Tab 1: 问答 (复用之前的逻辑，UI稍微美化) ===
with tab1:
    st.subheader("向记忆提问")
    # ... (标准的 Streamlit 聊天代码，和之前类似，调用 web_utils.answer_question) ...
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("最近发生了什么有趣的事？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            response = st.write_stream(web_utils.answer_question(prompt, []))
            st.session_state.messages.append({"role": "assistant", "content": response})

# === Tab 2: 酷炫的知识图谱可视化 ===
with tab2:
    st.subheader("全域知识图谱 (Interactive)")
    st.caption("这是 Agent“脑中”所有实体和关系的动态可视化。你可以拖动节点、缩放查看。")
    
    # 添加一个滑块来控制显示的节点数量，防止图太大卡顿
    limit = st.slider("显示最新的多少条关系?", 50, 500, 150)
    
    with st.spinner("正在构建神经元网络..."):
        # 调用后端生成 HTML
        graph_html = web_utils.generate_knowledge_graph_html(limit=limit)
        # 使用 components.html 渲染 Pyvis 生成的交互式图表
        components.html(graph_html, height=650, scrolling=True)

# === Tab 3: 简单的事件流表格 ===
with tab3:
    st.subheader("最近的事件记录")
    df = web_utils.get_recent_events_df()
    st.dataframe(df, use_container_width=True, hide_index=True)