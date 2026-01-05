"""
AI 循证医学助手 - 主应用 (Premium Edition)
基于 Streamlit 的医学知识库问答系统
"""
import streamlit as st
import time

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


# === 页面配置 ===
st.set_page_config(
    page_title=APP_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# === 自定义 CSS ===
st.markdown("""
<style>
    /* 全局字体优化 */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 聊天气泡优化 */
    .stChatMessage {
        background-color: transparent;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .stChatMessage:hover {
        background-color: rgba(240, 242, 246, 0.5);
    }

    /* 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }

    /* 按钮样式 */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* 标题样式 */
    h1 {
        color: #1a73e8;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* 状态指示器 */
    .stStatusWidget {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
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
    st.image("https://img.icons8.com/color/96/doctor-male--v1.png", width=80)
    st.title("MedAgent Pro")
    st.caption("专业的 AI 循证医学助手")
    st.divider()

    st.subheader("📚 知识库管家")

    # 显示已存储的文件
    with st.expander("已学习的书籍", expanded=True):
        current_files = db.get_existing_files()
        if not current_files:
            st.info("暂无数据，请上传教材。")
        else:
            for f in current_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"📖 {f}")
                with col2:
                    if st.button("🗑️", key=f"del_{f}", help=f"删除《{f}》"):
                        success, error = db.delete_file(f)
                        if success:
                            st.toast(f"已删除《{f}》")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"失败: {error}")

    st.divider()

    # 调试模式开关
    debug_mode = st.toggle('🛠️ 调试模式 (Debug)', value=False)

    # 上传新文件
    st.subheader("📤 上传新书")
    uploaded_files = st.file_uploader(
        "支持 PDF(转MD) / Markdown",
        type=["md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("开始学习", type="primary", use_container_width=True):
            for file in uploaded_files:
                with st.status(f"正在处理 {file.name}...", expanded=True) as status:
                    # 读取文件内容
                    content = file.read().decode("utf-8")

                    # 定义进度回调
                    progress_bar = status.empty()

                    def update_progress(progress, text):
                        progress_bar.progress(progress, text=text)

                    # 处理文件
                    start_time = time.time()
                    success, info = embedder.process_file(
                        content, file.name, db, update_progress
                    )
                    end_time = time.time()

                    # 显示结果
                    if success:
                        status.update(label=f"✅ 《{file.name}》学习完成！(耗时 {end_time - start_time:.1f}s)", state="complete", expanded=False)
                        st.toast(f"成功存入 {info} 条知识片段。")
                        time.sleep(1)
                        st.rerun()
                    elif info == "EXIST":
                        status.update(label=f"⚠️ 《{file.name}》已存在", state="complete", expanded=False)
                        st.warning(f"《{file.name}》已经学过了，跳过。")
                    else:
                        status.update(label="❌ 处理失败", state="error")
                        st.error(f"失败: {info}")


# === 主聊天界面 ===
st.title(f"👨‍⚕️ {APP_TITLE}")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "你好，我是你的专业医学助手。请描述患者的症状、体征或上传的检查结果，我会基于知识库为你提供循证建议。"
    }]

# 显示对话历史
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        with st.chat_message(msg["role"], avatar="👨‍⚕️"):
            st.markdown(msg["content"])
    else:
        with st.chat_message(msg["role"], avatar="🧑‍🎓"):
            st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入病例描述 (例如: 患者男，45岁，发热伴咳嗽3天...)"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍🎓").write(prompt)

    # AI 回复
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        response_container = st.empty()
        final_answer_text = ""

        with st.status("正在分析病例...", expanded=True) as status:

            # 调用 Agent
            agent_generator = agent.run(prompt, st.session_state.messages)

            for event_type, data in agent_generator:

                if event_type == "THOUGHT":
                    # 动态更新思考状态
                    status.update(label="🤔 正在思考...", state="running")
                    st.markdown(f"_{data}_")
                    st.divider()

                elif event_type == "ACTION_START":
                    status.update(label=f"🔍 正在检索: {data}", state="running")
                    st.toast(f"正在查阅: {data}")

                elif event_type == "OBSERVATION":
                    # 显示调试信息 (如果开启)
                    if debug_mode and agent.search_tool.last_logs:
                        with st.expander("📊 Rerank 打分详情", expanded=False):
                            for log in agent.search_tool.last_logs:
                                try:
                                    if log.strip().startswith("["):
                                        score = float(log.split(']')[0].replace('[', ''))
                                        if score > -10:
                                            st.markdown(f":green[{log}]")
                                        else:
                                            st.markdown(f":grey[{log}]")
                                    else:
                                        st.info(log)
                                except Exception:
                                    st.text(log)

                elif event_type == "FINAL_ANSWER":
                    final_answer_text = data
                     # 不再清除思考过程，改为收起状态
                    status.update(label="✅ 分析完成", state="complete", expanded=False)

        # 流式输出最终结果
        if final_answer_text:
            def stream_text():
                for word in final_answer_text.split():
                    yield word + " "
                    time.sleep(0.01)

            response_container.write_stream(stream_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer_text
            })
