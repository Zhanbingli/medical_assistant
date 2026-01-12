"""
AI 循证医学助手 - 主应用 (Premium Edition)
基于 Streamlit 的医学知识库问答系统 - 优化版UI
"""
import streamlit as st
import time
from typing import Dict, Any, List

from config import (
    APP_TITLE, PAGE_LAYOUT,
    DB_PATH, COLLECTION_NAME,
    RERANKER_MODEL, EMBEDDING_MODEL, LLM_MODEL,
    BATCH_SIZE
)
from src.rag.database import MedicalKnowledgeDB
from src.rag.loader import DocumentEmbedder
from src.rag.search import MedicalSearchEngine, Reranker, QueryExpander
from src.agent.tools import SearchTool
from src.agent.core import MedicalAgent
from src.utils.markdown_optimizer import optimize_markdown_for_rag

# === 页面配置 ===
st.set_page_config(
    page_title=APP_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# === 主题配置 ===
st.markdown("""
<style>
/* 主容器 */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 主内容区 */
.main-content {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
}

/* 聊天消息样式 */
.user-message {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 20px 20px 0 20px;
    padding: 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.assistant-message {
    background: white;
    border: 2px solid #e0e0e0;
    border-radius: 20px 20px 20px 0;
    padding: 1rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-right: auto;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
}

/* 医生头像 */
.doctor-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 1.2rem;
}

/* 诊断卡片 */
.diagnosis-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    border-left: 5px solid #fff;
}

/* 检查建议卡片 */
.checklist-card {
    background: white;
    border: 2px solid #34a853;
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* 上传区域 */
.upload-area {
    border: 2px dashed #667eea;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    background: rgba(102, 126, 234, 0.05);
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #764ba2;
    background: rgba(118, 75, 162, 0.1);
    transform: translateY(-2px);
}

/* 进度条 */
.medical-progress .stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 10px;
}

/* 按钮样式 */
.medical-button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.medical-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

/* 警告样式 */
.warning-container {
    background: linear-gradient(135deg, #ff9800, #f57c00);
    color: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
}

/* 错误样式 */
.error-container {
    background: linear-gradient(135deg, #f44336, #d32f2f);
    color: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
}

/* 成功样式 */
.success-container {
    background: linear-gradient(135deg, #4caf50, #388e3c);
    color: white;
    border-radius: 15px;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

/* 统计卡片 */
.stats-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.2);
    transition: transform 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-5px);
}

/* 侧边栏样式 */
.medical-sidebar {
    background: rgba(102, 126, 234, 0.05);
    border-radius: 0 20px 20px 0;
    padding: 2rem 1rem;
    height: 100%;
}

/* 文件列表 */
.file-item {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.3rem 0;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.file-item:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
    .main-content {
        margin: 0.5rem;
        padding: 1rem;
    }
    
    .user-message, .assistant-message {
        max-width: 90%;
    }
}
</style>
""", unsafe_allow_html=True)

# === 初始化组件（缓存） ===
@st.cache_resource
def init_system():
    """初始化系统组件"""
    # RAG 组件 (单例模式)
    db = MedicalKnowledgeDB(DB_PATH, COLLECTION_NAME)
    reranker = Reranker(RERANKER_MODEL)
    expander = QueryExpander(LLM_MODEL)
    search_engine = MedicalSearchEngine(db, reranker, expander)
    embedder = DocumentEmbedder(EMBEDDING_MODEL, BATCH_SIZE)

    # Agent 组件
    search_tool = SearchTool(search_engine)
    agent = MedicalAgent(search_tool)

    return db, embedder, agent

db, embedder, agent = init_system()

# === 侧边栏：知识库管理 ===
with st.sidebar:
    # 标题区域
    st.markdown("<div class='medical-sidebar'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <div class='doctor-avatar' style='margin: 0 auto 1rem;'>🩺</div>
        <h2 style='color: #667eea; margin: 0;'>MedAgent Pro</h2>
        <p style='color: #666; margin: 0;'>专业的 AI 循证医学助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)

    # 统计信息
    with st.expander("📊 系统统计", expanded=False):
        stats = db.get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class='stats-card' style='padding: 1rem; text-align: center;'>
                <h3 style='color: #667eea; margin: 0; font-size: 2rem;'>{}</h3>
                <p style='margin: 0; color: #666;'>知识片段</p>
            </div>
            """.format(stats.get('total_chunks', 0)), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='stats-card' style='padding: 1rem; text-align: center;'>
                <h3 style='color: #667eea; margin: 0; font-size: 2rem;'>{}</h3>
                <p style='margin: 0; color: #666;'>已学书籍</p>
            </div>
            """.format(stats.get('total_files', 0)), unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)

    # 知识库管理
    st.subheader("📚 知识库管理")

    # 已学习的书籍
    with st.expander("已学习的书籍", expanded=True):
        current_files = db.get_existing_files()
        if not current_files:
            st.info("暂无数据，请上传教材")
        else:
            for f in sorted(current_files):
                st.markdown(f"""
                <div class='file-item'>
                    📖 <strong>{f}</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)

    # 模型信息
    with st.expander("🤖 模型信息", expanded=False):
        st.markdown(f"""
        - **嵌入模型**: {EMBEDDING_MODEL}
        - **推理模型**: {LLM_MODEL}
        - **重排序**: {RERANKER_MODEL}
        - **批大小**: {BATCH_SIZE}
        """)
    
    st.markdown("<hr style='border-color: rgba(102, 126, 234, 0.2);'>", unsafe_allow_html=True)
    
    # 调试模式
    debug_mode = st.toggle('🛠️ 调试模式', value=False)
    
    # 上传新文件
    st.subheader("📤 上传新书")
    
    # 提供优化选项
    optimize_md = st.toggle("🔧 自动优化Markdown", value=True, help="启用后自动清理页码、页眉页脚并优化结构")
    
    # 拖拽上传区域
    st.markdown("""
    <div class='upload-area'>
        <p style='color: #667eea; font-size: 1.2rem; margin-bottom: 1rem;'>📚 拖拽文件到此处</p>
        <p style='color: #666;'>支持 Markdown 格式</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        # 统计文件大小
        total_size = sum(file.size for file in uploaded_files) / 1024 / 1024  # MB
        st.info(f"共选择了 {len(uploaded_files)} 个文件，总计 {total_size:.2f} MB")
        
        if st.button("🚀 开始学习", type="primary", use_container_width=True):
            success_count = 0
            for file in uploaded_files:
                with st.status(f"📖 正在学习 {file.name}...", expanded=True) as status:
                    try:
                        content = file.read().decode("utf-8", errors='ignore')
                        
                        progress_text = status.empty()
                        
                        def update_progress(progress, text):
                            progress_text.progress(progress, text=f"{file.name}: {text}")
                        
                        start_time = time.time()
                        success, info = embedder.process_file(
                            content, file.name, db, update_progress
                        )
                        end_time = time.time()
                        
                        if success:
                            success_count += 1
                            status.update(
                                label=f"✅ 学习成功！添加了 {info} 条知识",
                                state="complete"
                            )
                            st.toast(f"📚 成功学习《{file.name}》", icon="🎉")
                        elif info == "EXIST":
                            status.update(
                                label=f"⚠️ 《{file.name}》 已存在",
                                state="complete"
                            )
                        else:
                            status.update(
                                label=f"❌ 学习失败: {info}",
                                state="error"
                            )
                    except Exception as e:
                        status.update(
                            label=f"❌ 处理失败: {str(e)}",
                            state="error"
                        )
            
            if success_count > 0:
                st.success(f"🎉 成功学习 {success_count} 本新书！")
                time.sleep(2)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# === 主聊天界面 ===
# Header
def render_header():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 2rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>👨‍⚕️ AI 循证医学助手</h1>
        <p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;'>专业医学知识库 · 循证诊断建议</p>
    </div>
    """, unsafe_allow_html=True)

render_header()

# 使用说明卡片
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    <div style='background: rgba(102, 126, 234, 0.05); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea;'>
        <h4 style='color: #667eea; margin-top: 0;'>如何获得最佳诊断建议？</h4>
        <ul style='margin-bottom: 0;'>
            <li>描述患者的<b>主要症状</b>（如：发热、咳嗽、胸痛）</li>
            <li>说明症状的<b>持续时间</b>（如：3天、1周）</li>
            <li>提供<b>关键体征</b>（如：体温38.5°C、血压120/80）</li>
            <li>提及<b>既往病史</b>（如：高血压病史5年）</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 症状模板卡片
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🤒 发热相关", use_container_width=True):
        st.session_state.suggested_input = "患者发热3天，体温最高38.5°C，伴有寒战和头痛，需要注意哪些诊断？"

with col2:
    if st.button("🫁 呼吸系统", use_container_width=True):
        st.session_state.suggested_input = "患者咳嗽1周，伴有胸闷和呼吸困难，无发热，需要注意哪些诊断？"

with col3:
    if st.button("❤️ 心血管", use_container_width=True):
        st.session_state.suggested_input = "患者胸痛2小时，伴有呼吸困难和出汗，需要注意哪些诊断？"

with col4:
    if st.button("🤕 急腹症", use_container_width=True):
        st.session_state.suggested_input = "患者腹痛6小时，伴有恶心和呕吐，右下腹压痛明显，需要注意哪些诊断？"

# 初始化或获取建议输入
if "suggested_input" in st.session_state:
    suggested_text = st.session_state.suggested_input
    del st.session_state.suggested_input
else:
    suggested_text = ""

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "🩺 你好，我是你的专业医学助手。请描述患者的症状、体征或检查结果，我会基于知识库为你提供循证诊断建议。",
        "type": "welcome"
    }]

# 显示对话历史
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "assistant":
        if msg.get("type") == "welcome":
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0;'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div class='doctor-avatar'>👨‍⚕️</div>
                    <strong style='margin-left: 1rem; font-size: 1.2rem;'>AI 医学助手</strong>
                </div>
                <div>{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        elif "diagnosis" in msg.get("type", ""):
            # 诊断结果卡片
            st.markdown(f"""
            <div class='diagnosis-card'>
                <h4 style='margin-top: 0; color: white;'>📊 诊断建议</h4>
                <div style='color: white; line-height: 1.8;'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        elif "checklist" in msg.get("type", ""):
            # 检查建议卡片
            st.markdown(f"""
            <div class='checklist-card'>
                <h4 style='margin-top: 0; color: #34a853;'>🔬 检查建议</h4>
                <div style='line-height: 1.8;'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display: flex; align-items: flex-start; margin: 1rem 0;'>
                <div class='doctor-avatar'>👨‍⚕️</div>
                <div class='assistant-message'>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin: 1rem 0; justify-content: flex-end;'>
            <div style='background: #667eea; color: white; border-radius: 20px 20px 0 20px; padding: 1rem; max-width: 80%;'>{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

# 用户输入
# Note: st.chat_input doesn't support value parameter, so we handle template click separately
if suggested_text:
    # If a template was clicked, use it directly as input
    user_input = suggested_text
else:
    # Otherwise wait for user input
    user_input = st.chat_input("请详细描述患者症状（如：发热伴咳嗽3天，体温38.5°C）...")

if user_input:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 重新渲染用户消息
    st.markdown(f"""
    <div style='display: flex; align-items: flex-start; margin: 1rem 0; justify-content: flex-end;'>
        <div style='background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 20px 20px 0 20px; padding: 1rem; max-width: 80%;'>{user_input}</div>
    </div>
    """, unsafe_allow_html=True)

    # AI 回复
    with st.container():
        st.markdown("<div style='margin: 2rem 0;'>", unsafe_allow_html=True)
        response_container = st.empty()
        
        # 创建状态指示器
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            st.markdown("""
            <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea;'>
                <div style='display: flex; align-items: center;'>
                    <div class='doctor-avatar'>👨‍⚕️</div>
                    <strong style='margin-left: 1rem; font-size: 1.1rem;'>正在分析中...</strong>
                </div>
                <div style='margin-top: 1rem;'>
                    <div style='background: rgba(102, 126, 234, 0.1); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;'>
                        🔍 检索相关知识中...
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        try:
            final_answer_text = ""
            agent_generator = agent.run(user_input, st.session_state.messages)

            for event_type, data in agent_generator:
                if event_type == "THOUGHT":
                    # 思考过程可以显示在调试模式中
                    if debug_mode:
                        st.markdown(f"<div style='background: rgba(255, 152, 0, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 3px solid #ff9800;'><strong>🤔 思考过程:</strong><br>{data}</div>", unsafe_allow_html=True)
                
                elif event_type == "ACTION_START":
                    # 更新状态为检索中
                    status_placeholder.empty()
                    with status_placeholder.container():
                        st.markdown(f"""
                        <div style='background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #667eea;'>
                            <div style='display: flex; align-items: center;'>
                                <div class='doctor-avatar'>👨‍⚕️</div>
                                <strong style='margin-left: 1rem; font-size: 1.1rem;'>正在分析中...</strong>
                            </div>
                            <div style='margin-top: 1rem;'>
                                <div style='background: rgba(76, 175, 80, 0.1); padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0;'>
                                    🔍 正在检索: <strong>{data}</strong>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                elif event_type == "OBSERVATION":
                    # 显示检索结果（调试模式）
                    if debug_mode and agent.search_tool.last_logs:
                        with st.expander("📊 检索详情", expanded=False):
                            for log in agent.search_tool.last_logs:
                                st.markdown(f"<div style='font-family: monospace; padding: 0.5rem; background: rgba(0,0,0,0.05); border-radius: 5px; margin: 0.2rem 0; font-size: 0.9rem;'>{log}</div>", unsafe_allow_html=True)
                
                elif event_type == "FINAL_ANSWER":
                    final_answer_text = data
                    final_answer_text = data
                    
                    # 判断内容类型
                    if "诊断" in data and ("1." in data or "①" in data):
                        msg_type = "diagnosis"
                    elif "检查" in data or "实验室" in data:
                        msg_type = "checklist"
                    else:
                        msg_type = "general"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_answer_text,
                        "type": msg_type
                    })
                    
                    # 清空状态指示器，显示结果
                    status_placeholder.empty()
                    
                    if msg_type == "diagnosis":
                        st.markdown(f"""
                        <div class='diagnosis-card'>
                            <h4 style='margin-top: 0; color: white;'>📋 诊断分析</h4>
                            <div style='color: white; line-height: 1.8; white-space: pre-wrap;'>{final_answer_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif msg_type == "checklist":
                        st.markdown(f"""
                        <div class='checklist-card'>
                            <h4 style='margin-top: 0; color: #34a853;'>🔬 检查建议</h4>
                            <div style='line-height: 1.8; white-space: pre-wrap;'>{final_answer_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='display: flex; align-items: flex-start; margin: 1rem 0;'>
                            <div class='doctor-avatar' style='margin-right: 1rem;'>👨‍⚕️</div>
                            <div class='assistant-message' style='flex: 1;'>{final_answer_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
        except Exception as e:
            status_placeholder.empty()
            st.error(f"🚨 分析过程中出现错误: {str(e)}")
            st.exception(e)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # 重新运行以更新UI
        time.sleep(1)
        st.rerun()