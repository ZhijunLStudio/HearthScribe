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
st.set_page_config(
    page_title="HearthScribe",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    /* 顶部卡片样式 */
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #6c757d;
    }
    /* 不同卡片不同色 */
    div[data-testid="stMetric"]:nth-of-type(1) { border-left-color: #e74c3c; } /* 红 */
    div[data-testid="stMetric"]:nth-of-type(2) { border-left-color: #f39c12; } /* 黄 */
    div[data-testid="stMetric"]:nth-of-type(3) { border-left-color: #3498db; } /* 蓝 */
    div[data-testid="stMetric"]:nth-of-type(4) { border-left-color: #2ecc71; } /* 绿 */
    
    /* 时间轴列表 */
    .timeline-item {
        background: white;
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 6px;
        border-left: 3px solid #ddd;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.title("HearthScribe")
    st.caption("AI 智能看护系统 v2.0")
    st.markdown("---")
    
    nav = st.radio("系统导航", ["🏠 态势看板", "📽️ 影像回溯", "📝 报告生成", "🕸️ 认知图谱", "🤖 智能助手"])
    
    st.markdown("---")
    # 系统硬指标 (实时查询)
    sys_stats = web_utils.get_system_stats()
    c1, c2 = st.columns(2)
    with c1: st.metric("记忆库", sys_stats['memory'])
    with c2: st.metric("实体数", sys_stats['entities'])
    st.metric("累计看护", sys_stats['care_hours'])

# --- 1. 态势看板 ---
if nav == "🏠 态势看板":
    st.header("☀️ 今日空间态势")
    
    # 1. 核心指标卡片
    data = web_utils.get_dashboard_stats()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("⚠️ 风险预警", f"{data.get('risk_count',0)}", "次")
    with c2: st.metric("⏱️ 最大静止", f"{data.get('max_inactive_min',0)}", "分钟")
    with c3: st.metric("🛌 休息时长", f"{data.get('rest_hours',0)}", "小时")
    with c4: st.metric("📸 今日事件", f"{data.get('event_count',0)}", f"最新: {data.get('last_active','--')}")
    
    st.markdown("---")
    
    # 2. 图表区
    # 左侧：趋势图 (宽)
    # 右侧：饼图 + 柱状图 (上下排布)
    c_main, c_side = st.columns([2, 1])
    
    with c_main:
        st.subheader("📈 交互活跃度趋势")
        df_trend = web_utils.get_interaction_trend()
        if not df_trend.empty:
            chart = alt.Chart(df_trend).mark_area(
                line={'color':'#3498db'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='#3498db', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('Time', title='时刻'),
                y=alt.Y('Score', title='活跃度 (0-10)'),
                tooltip=['Time', 'Score']
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("今日暂无交互数据，图表待生成...")

    with c_side:
        # 饼图：场景分布
        st.subheader("🍰 场景分布")
        df_scene = web_utils.get_scene_distribution()
        if not df_scene.empty:
            pie = alt.Chart(df_scene).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("Count", stack=True),
                color=alt.Color("Type", scale={"scheme": "pastel1"}),
                tooltip=["Type", "Count"]
            ).properties(height=200)
            st.altair_chart(pie, use_container_width=True)
        else:
            st.caption("暂无数据")
            
        # 柱状图：人员频率
        st.subheader("👥 人员频率")
        df_person = web_utils.get_person_frequency()
        if not df_person.empty:
            bar = alt.Chart(df_person).mark_bar().encode(
                x='Count',
                y=alt.Y('Name', sort='-x'),
                color=alt.Color("Name", scale={"scheme": "set2"})
            ).properties(height=200)
            st.altair_chart(bar, use_container_width=True)
        else:
            st.caption("暂无人员数据")

# --- 2. 影像回溯 (修复详情页) ---
elif nav == "📽️ 影像回溯":
    st.header("📅 历史影像归档")
    
    # 获取数据
    events = web_utils.MEMORY.get_rich_event_details(limit=50)
    
    if not events:
        st.info("暂无历史影像数据。")
    else:
        # 构造选择项
        evt_map = {f"{datetime.fromtimestamp(e['start_time']).strftime('%H:%M')} - {e['summary'][:30]}...": e for e in events}
        selected_label = st.selectbox("请选择一个事件查看详情:", list(evt_map.keys()))
        
        if selected_label:
            evt = evt_map[selected_label]
            txt, lbl, score = web_utils.parse_summary(evt['summary'])
            
            st.markdown("---")
            c_meta, c_imgs = st.columns([1, 2])
            
            with c_meta:
                st.info(f"**AI 观察**: {txt}")
                st.write(f"**发生时间**: {datetime.fromtimestamp(evt['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"**场景标签**: `{lbl}`")
                st.write(f"**活跃评分**: `{score}/10`")
                
            with c_imgs:
                st.subheader("📸 抓拍序列")
                try:
                    paths = json.loads(evt['image_paths'])
                    if paths:
                        # 使用 Tabs 展示多张图，避免刷屏
                        tabs = st.tabs([f"帧 {i+1}" for i in range(len(paths))])
                        for i, p in enumerate(paths):
                            if os.path.exists(p):
                                tabs[i].image(p, use_container_width=True)
                            else:
                                tabs[i].warning("图片文件丢失")
                except:
                    st.error("图片路径解析失败")

# --- 3. 报告生成 (回归) ---
elif nav == "📝 报告生成":
    st.header("📋 智能报告中心")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        d = st.date_input("选择日期", datetime.now())
        if st.button("🚀 生成报告", use_container_width=True):
            with st.spinner("正在生成 Markdown 报告..."):
                report_md = web_utils.generate_daily_report_content(d)
                st.session_state['report_md'] = report_md # 缓存结果
                
    with col2:
        if 'report_md' in st.session_state:
            st.markdown("### 预览")
            st.markdown(st.session_state['report_md'])
            st.download_button("📥 下载 .md 文件", st.session_state['report_md'], f"report_{d}.md")

# --- 4. 认知图谱 ---
elif nav == "🕸️ 认知图谱":
    st.header("🧠 空间认知网络")
    html = web_utils.generate_kg_html()
    # 使用 scrolling=True 允许图谱缩放
    st.components.v1.html(html, height=700, scrolling=True)

# --- 5. 智能助手 ---
elif nav == "🤖 智能助手":
    st.header("💬 关怀问答")
    
    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = []
    
    # 渲染历史
    for role, text in st.session_state.chat_history:
        with st.chat_message(role): st.markdown(text)
        
    # 输入处理
    if q := st.chat_input("您可以问：今天有人跌倒吗？爷爷什么时候吃的药？"):
        st.session_state.chat_history.append(("user", q))
        with st.chat_message("user"): st.markdown(q)
        
        with st.chat_message("assistant"):
            ph = st.empty()
            full_resp = ""
            for chunk in web_utils.agent_answer_stream(q):
                full_resp = chunk
                ph.markdown(full_resp + "▌") # 打字机效果
            ph.markdown(full_resp)
            st.session_state.chat_history.append(("assistant", full_resp))