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
st.set_page_config(
    page_title="HearthScribe - 文心驱动", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 深度美化 ---
st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #f4f7f6; }
    
    /* === 顶部通栏标题 (Hero Header) === */
    .main-header {
        background: white;
        padding: 25px 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        margin-bottom: 25px;
        border-bottom: 3px solid #1a73e8; /* 品牌底色条 */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .header-top-row {
        display: flex;
        align-items: baseline;
        gap: 15px;
    }
    
    .app-title {
        font-size: 32px;
        font-weight: 900;
        color: #2c3e50;
        letter-spacing: -0.5px;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 18px;
        font-weight: 500;
        color: #1a73e8; /* 文心蓝 */
        margin: 0;
    }
    
    .badge {
        background-color: #e8f0fe;
        color: #1a73e8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        transform: translateY(-5px);
    }
    
    .app-slogan {
        font-size: 14px;
        color: #7f8c8d;
        font-style: italic;
        margin-top: 8px;
        font-family: "Georgia", serif;
    }

    /* === 侧边栏优化 === */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }
    .stRadio label {
        font-size: 16px !important;
        padding: 10px 0;
        font-weight: 500;
    }

    /* === 指标卡片 === */
    div[data-testid="stMetric"] {
        background-color: white; 
        padding: 15px; 
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04); 
        border: 1px solid #f0f0f0; 
        border-left: 5px solid #ccc;
    }
    /* 颜色区分 */
    div[data-testid="stMetric"]:nth-of-type(1) { border-left-color: #3498db; }
    div[data-testid="stMetric"]:nth-of-type(2) { border-left-color: #e74c3c; }
    div[data-testid="stMetric"]:nth-of-type(3) { border-left-color: #f1c40f; }
    div[data-testid="stMetric"]:nth-of-type(4) { border-left-color: #2ecc71; }
    
    /* === 洞察横幅 === */
    .insight-box {
        background: linear-gradient(to right, #e3f2fd, #ffffff);
        border-left: 5px solid #2196f3;
        padding: 15px 20px; 
        border-radius: 8px; 
        margin-bottom: 25px;
    }
    .insight-box.ready { 
        background: linear-gradient(to right, #e8f5e9, #ffffff);
        border-left-color: #4caf50; 
    }
    
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "view_mode" not in st.session_state: st.session_state.view_mode = "gallery"
if "selected_event_id" not in st.session_state: st.session_state.selected_event_id = None

# --- 🟢 Sidebar (极简模式) ---
with st.sidebar:
    st.markdown("### ⚙️ 系统导航")
    nav = st.radio(
        "", # 隐藏标题，直接显示选项
        ["📊 态势看板", "🎞️ 影像回溯", "📝 报告生成", "🕸️ 认知图谱", "💬 智能管家"],
        index=0
    )
    
    st.markdown("---")
    
    # 底部放置操作按钮和版权
    col_btn, _ = st.columns([1, 0.1])
    with col_btn:
        if st.button("🔄 刷新全站数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
            
    st.markdown("""
        <div style='text-align: center; color: #999; font-size: 12px; margin-top: 20px;'>
            Powered by Baidu ERNIE 4.5<br>
            © 2025 HearthScribe
        </div>
    """, unsafe_allow_html=True)

# --- 🔵 Main Header (顶部通栏) ---
# 这段代码放在所有逻辑之前，作为页面的“页眉”
st.markdown("""
<div class="main-header">
    <div class="header-top-row">
        <div class="app-title">🏡 HearthScribe</div>
        <div class="app-subtitle">基于文心大模型的适老化智能看护系统</div>
        <div class="badge">ERNIE Inside</div>
    </div>
    <div class="app-slogan">
        —— 让爱跨越时空，为长者点亮 24 小时的 AI 守护灯
    </div>
</div>
""", unsafe_allow_html=True)


# --- 1. 态势看板 ---
if nav == "📊 态势看板":
    
    # 洞察横幅
    insight = web_utils.get_daily_insight_preview()
    css_class = "ready" if insight['ready'] else ""
    icon = "✅" if insight['ready'] else "👁️"
    
    st.markdown(f"""
    <div class="insight-box {css_class}">
        <h4 style="margin:0; color:#1565c0; display:flex; align-items:center; gap:8px;">
            {icon} {insight["title"]}
        </h4>
        <div style="margin-top:8px; color:#555; font-size:15px;">{insight["content"]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📡 核心监控指标")
    stats = web_utils.get_dashboard_stats()
    
    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📸 今日事件", stats.get('event_count', 0))
    c2.metric("🚨 风险告警", stats.get('risk_count', 0), delta_color="inverse")
    c3.metric("💤 最大静止", f"{stats.get('max_inactive_min', 0)} min")
    c4.metric("👥 家人探访", f"{stats.get('family_count', 0)} 人")
    
    st.write("")
    
    # Row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🏃 活跃时长", f"{stats.get('active_hours', 0)} h")
    c6.metric("🛌 休息时长", f"{stats.get('rest_hours', 0)} h")
    c7.metric("🤝 高频互动", f"{stats.get('social_count', 0)} 次")
    c8.metric("🧠 新知沉淀", f"{stats.get('new_knowledge', 0)} 条")
    
    st.divider()
    
    # Charts
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("📈 24小时交互热度")
        df_trend = web_utils.get_interaction_trend()
        if not df_trend.empty:
            # 标准折线图
            base = alt.Chart(df_trend).encode(
                x=alt.X('Time', title='时间轴', axis=alt.Axis(labelAngle=0)),
                # --- 关键修改在这里 ---
                # scale=alt.Scale(domain=[0, 10]) 强制将纵轴固定在 0-10
                y=alt.Y('Score', title='活跃评分 (0-10)', scale=alt.Scale(domain=[0, 10])),
                tooltip=['Time', 'Score']
            )
            
            line = base.mark_line(color='#1a73e8', strokeWidth=3)
            points = base.mark_circle(size=80, color='white', stroke='#1a73e8', strokeWidth=2)
            # 区域填充
            area = base.mark_area(opacity=0.1, color='#1a73e8') 
            
            st.altair_chart((area + line + points).interactive(), use_container_width=True)
        else:
            st.info("数据收集中...")
            
    with chart_col2:
        st.subheader("🍰 场景分布")
        df_scene = web_utils.get_scene_distribution()
        if not df_scene.empty:
            # 甜甜圈图
            base = alt.Chart(df_scene).encode(theta=alt.Theta("Count", stack=True))
            pie = base.mark_arc(outerRadius=120, innerRadius=70).encode(
                color=alt.Color("Type", scale=alt.Scale(scheme='set2')),
                order=alt.Order("Count", sort="descending"),
                tooltip=["Type", "Count"]
            )
            text = base.mark_text(radius=145).encode(
                text=alt.Text("Type"), 
                order=alt.Order("Count", sort="descending"), 
                color=alt.value("#333")
            )
            st.altair_chart(pie + text, use_container_width=True)
        else:
            st.caption("暂无数据")

# --- 2. 影像回溯 (Grid) ---
elif nav == "🎞️ 影像回溯":
    st.subheader("🎞️ 历史影像归档")
    if st.session_state.view_mode == "detail":
        if st.button("⬅️ 返回列表"):
            st.session_state.view_mode = "gallery"
            st.rerun()
        
        evt = web_utils.MEMORY.get_rich_event_details([st.session_state.selected_event_id])[0]
        txt, lbl, score = web_utils.parse_summary(evt['summary'])
        
        # 详情页顶部样式
        st.markdown(f"""
        <div style="background:white; padding:25px; border-radius:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom:25px;">
            <h3 style="margin-top:0; color:#2c3e50;">📝 AI 观察报告</h3>
            <div style="font-size:16px; line-height:1.6; color:#444;">{txt}</div>
            <hr style="margin: 20px 0; border: 0; border-top: 1px solid #eee;">
            <div style="display:flex; gap:30px; font-weight:500; color:#666;">
                <span>⏱️ {datetime.fromtimestamp(evt['start_time']).strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span style="background:#e3f2fd; color:#1565c0; padding:2px 10px; border-radius:12px; font-size:14px;">{lbl}</span>
                <span>⚡ 活跃评分: {score}/10</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
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
                        t_str = datetime.fromtimestamp(evt['start_time']).strftime('%H:%M')
                        txt, label, score = web_utils.parse_summary(evt['summary'])
                        st.markdown(f"**{t_str}** <span style='float:right; font-size:12px; background:#f0f0f0; padding:2px 6px; border-radius:4px;'>⭐ {score}</span>", unsafe_allow_html=True)
                        st.caption(f"{label} | {txt[:12]}...")
                        if st.button("查看", key=evt['event_id'], use_container_width=True):
                            st.session_state.selected_event_id = evt['event_id']
                            st.session_state.view_mode = "detail"
                            st.rerun()

# --- 3. 报告生成 ---
elif nav == "📝 报告生成":
    st.header("📋 智能日报生成")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("#### 📅 设定")
        d = st.date_input("选择日期")
        st.write("")
        if st.button("🚀 生成叙述性报告", type="primary", use_container_width=True):
            with st.spinner("文心大模型正在撰写..."):
                st.session_state['report_md'] = web_utils.generate_daily_report_content(d)
    with col2:
        if 'report_md' in st.session_state:
            st.markdown("#### 📄 报告预览")
            # 给报告加一个白底容器，像一张纸
            st.markdown(f"""
            <div style="background:white; padding:40px; border-radius:5px; box-shadow: 0 2px 15px rgba(0,0,0,0.08); min-height:600px;">
                {st.session_state['report_md']}
            </div>
            """, unsafe_allow_html=True)

# --- 4. 认知图谱 ---
elif nav == "🕸️ 认知图谱":
    st.header("🧠 空间认知网络")
    with st.spinner("正在绘制图谱..."):
        st.components.v1.html(web_utils.generate_kg_html(), height=750)

# --- 5. 智能管家 ---
elif nav == "💬 智能管家":
    st.header("💬 家庭管家 (ERNIE Bot)")
    if "messages" not in st.session_state: st.session_state.messages = []
    
    for role, content in st.session_state.messages:
        with st.chat_message(role): st.markdown(content)
        
    if prompt := st.chat_input("问：今天有人来过吗？"):
        st.session_state.messages.append(("user", prompt))
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            status_box = st.status("🧠 模型思考中...", expanded=True)
            ph = st.empty()
            full = ""
            try:
                for chunk in web_utils.agent_answer_stream(prompt):
                    if not chunk: continue
                    if isinstance(chunk, str):
                        full += chunk
                        ph.markdown(full + "▌")
                    else:
                        st_type = chunk.get("status")
                        if st_type == "thinking": status_box.write(chunk.get("content"))
                        elif st_type == "answer":
                            status_box.update(label="✅ 完成", state="complete", expanded=False)
                            full += chunk.get("content", "")
                            ph.markdown(full + "▌")
                ph.markdown(full)
                st.session_state.messages.append(("assistant", full))
            except Exception as e:
                st.error(f"Error: {e}")