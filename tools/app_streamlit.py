import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime
import json

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import web_utils # 假设你保留了 web_utils 用于数据库读取

# --- 页面配置 ---
st.set_page_config(
    page_title="HearthScribe 长者守护系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS (让界面更温暖、专业) ---
st.markdown("""
<style>
    .reportview-container { background: #fdfcf0; }
    .main-header { font-family: 'Helvetica Neue', sans-serif; color: #2c3e50; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .css-1r6slb0 { background-color: #ffffff; border: 1px solid #eee; }
    .highlight-card { background-color: #e8f4f8; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db; }
    .alert-card { background-color: #fdecea; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/elderly-person.png", width=80)
    st.title("HearthScribe\n智慧守护")
    st.markdown("---")
    
    menu = st.radio("功能导航", ["🛡️ 实时看板", "📅 历史回溯", "🧠 认知图谱", "⚙️ 系统设置"])
    
    st.markdown("---")
    st.caption("系统状态: 🟢 在线监控中")
    st.caption(f"PaddleX 引擎: 🟢 {os.getenv('DET_MODEL_NAME', 'PicoDet')}")

# --- 1. 实时看板 (Dashboard) ---
if menu == "🛡️ 实时看板":
    st.markdown("<h1 class='main-header'>今日安康看板</h1>", unsafe_allow_html=True)
    st.caption(f"📅 {datetime.now().strftime('%Y年%m月%d日')} | 📍 客厅/卧室监控")

    # 顶部指标卡
    col1, col2, col3, col4 = st.columns(4)
    stats = web_utils.get_dashboard_stats() # 需要在 web_utils 适配返回 mock 或真实数据
    
    with col1: st.metric("今日活动事件", f"{stats.get('new_memories', 0)} 次", "+2")
    with col2: st.metric("识别到长者", "2 位", "Penny, Howard")
    with col3: st.metric("健康风险预警", "0 次", delta_color="normal") # 绿色表示无风险
    with col4: st.metric("环境安全指数", "98/100", "优")

    st.divider()

    # 左右布局
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.subheader("📹 最新动态摘要")
        # 获取最近一条事件
        recent_events = web_utils.MEMORY.get_rich_event_details(limit=1)
        if recent_events:
            evt = recent_events[0]
            # 判断是否有风险关键词
            is_risk = "跌倒" in evt['summary'] or "痛苦" in evt['summary']
            css_class = "alert-card" if is_risk else "highlight-card"
            
            st.markdown(f"""
            <div class='{css_class}'>
                <h3>{'⚠️ 异常监测' if is_risk else '✅ 正常活动'}</h3>
                <p><strong>时间:</strong> {datetime.fromtimestamp(evt['start_time']).strftime('%H:%M:%S')}</p>
                <p style='font-size:18px;'>{evt['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(evt['preview_image_path'], caption="现场快照", use_container_width=True)
        else:
            st.info("暂无今日活动记录，长者可能正在休息或不在监控区。")

    with c_right:
        st.subheader("🤖 守护助手")
        st.markdown("您可以询问关于长者的任何细节，例如：*“妈妈今天吃药了吗？”*")
        
        # 聊天窗口
        if "messages" not in st.session_state: st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])
            
        if prompt := st.chat_input("输入您的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            
            # 调用 Agent 回答
            with st.chat_message("assistant"):
                with st.spinner("回忆分析中..."):
                    # 这里复用原有的 agent 逻辑
                    full_response = ""
                    for chunk in web_utils.agent_answer_stream(prompt):
                        full_response = chunk # 这里简化处理，实际可以使用 st.write_stream
                    st.write(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- 2. 历史回溯 (Gallery) ---
elif menu == "📅 历史回溯":
    st.header("生活时光轴")
    date_filter = st.date_input("选择日期", datetime.now())
    
    events = web_utils.MEMORY.get_rich_event_details(limit=20) # 实际应传入 date_filter
    
    for evt in events:
        with st.expander(f"⏰ {datetime.fromtimestamp(evt['start_time']).strftime('%H:%M')} - {evt['summary'][:20]}...", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(evt['preview_image_path'], use_container_width=True)
            with c2:
                st.markdown(f"**完整摘要:** {evt['summary']}")
                # 显示提取出的风险/情绪 (如果在KG里存了)
                # st.tag("情绪: 平静") 

# --- 3. 认知图谱 (KG) ---
elif menu == "🧠 认知图谱":
    st.header("长者行为习惯图谱")
    st.caption("基于 ERNIE-Thinking 长期分析构建的健康与行为关联网络。")
    
    # 嵌入 PyVis HTML
    html_path = web_utils.generate_knowledge_graph_html() # 这里需要修改 web_utils 让它返回 path 或 string
    if isinstance(html_path, str) and html_path.startswith("<"):
        st.components.v1.html(html_path, height=600)
    else:
        st.info("图谱生成中...")