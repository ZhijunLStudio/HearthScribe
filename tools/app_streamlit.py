# tools/app_streamlit.py

import streamlit as st
import sys
import os
import pandas as pd
import logging
import time

# --- 路径修正，确保能导入src下的模块 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 导入后端函数 ---
try:
    from src.web_utils import (
        get_all_memories_df,
        format_memories_for_display,
        answer_question,
        run_analysis,
        get_time_range
    )
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    st.error("请确保你是在项目的根目录 (RaspiAgent/) 下运行 `streamlit run tools/app_streamlit.py`")
    st.stop()

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="RaspiAgent UI",
    page_icon="🧠",
    layout="wide"
)

# --- 主标题 ---
st.header("🧠 RaspiAgent 智能记忆助手", divider="rainbow")


# --- 标签页 ---
tab1, tab2, tab3 = st.tabs(["[ 📷 记忆浏览器 ]", "[ 💬 问答助手 ]", "[ 📊 总结与分析 ]"])


# --- 1. 记忆浏览页面 ---
with tab1:
    # 使用 st.cache_data 缓存数据
    @st.cache_data(ttl=60) # 缓存1分钟
    def load_memory_data():
        df = get_all_memories_df()
        if df.empty:
            return None, "数据库中暂无记忆数据。", pd.DataFrame(columns=["时间", "摘要", "参与者"])
        images, status, dataframe_data = format_memories_for_display(df)
        return images, status, dataframe_data
    
    with st.expander("ℹ️ 查看状态与操作", expanded=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 刷新记忆", use_container_width=True):
                st.cache_data.clear()
                st.toast("记忆已刷新！", icon="✅")
        
        try:
            images, status, dataframe_data = load_memory_data()
            with col2:
                st.info(f"**当前状态:** {status}")
        except Exception as e:
            st.error(f"加载记忆时出错: {e}")
            images, dataframe_data = None, pd.DataFrame()

    if images is not None:
        st.subheader("🖼️ 记忆预览 (最新在前)", divider="gray")
        if images:
            cols = st.columns(5)
            for i, (image_path, caption) in enumerate(images):
                with cols[i % 5]:
                    st.image(image_path, caption=caption, use_column_width="auto")
        else:
            st.warning("没有找到可显示的记忆图片。")

        st.subheader("📋 详细记忆列表", divider="gray")
        st.dataframe(dataframe_data, use_container_width=True, hide_index=True)

# --- 2. 问答助手页面 ---
with tab2:
    st.subheader("💬 记忆问答助手", divider="gray")
    st.markdown("向记忆库提问，探索过去的点滴。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("例如：lizhijun在喝水吗？"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response_generator = answer_question(prompt, st.session_state.messages)
                full_response = st.write_stream(response_generator)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"问答时发生错误: {e}")
                logging.error("问答时发生错误", exc_info=True)

# --- 3. 总结分析页面 ---
with tab3:
    st.subheader("📊 总结与分析", divider="gray")
    st.markdown("对指定时间范围的记忆进行深度分析，发现行为模式。")
    
    try:
        min_date, max_date = get_time_range()
        col1, col2 = st.columns(2)
        with col1:
            start_date_input = st.date_input("起始日期", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date_input = st.date_input("终止日期", value=max_date, min_value=min_date, max_value=max_date)

        if st.button("🚀 生成分析报告"):
            if start_date_input > end_date_input:
                st.error("错误：起始日期不能晚于终止日期。")
            else:
                with st.spinner("正在进行深度分析，请稍候..."):
                    try:
                        report = run_analysis(start_date_input, end_date_input)
                        st.subheader("分析报告")
                        st.markdown(report)
                    except Exception as e:
                        st.error(f"生成报告时出错: {e}")
                        logging.error("生成报告时出错", exc_info=True)
    except Exception as e:
        st.error(f"加载时间范围时出错: {e}")
        logging.error("加载时间范围时出错", exc_info=True)