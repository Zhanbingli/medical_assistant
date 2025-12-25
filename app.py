"""
AI 循证医学助手 - 主应用
基于 Streamlit 的医学知识库问答系统
"""
import streamlit as st
import ollama
import time

from config import (
    APP_TITLE, PAGE_LAYOUT,
    DB_PATH, COLLECTION_NAME,
    RERANKER_MODEL, EMBEDDING_MODEL, LLM_MODEL,
    BATCH_SIZE, SYSTEM_PROMPT,
    MAX_REASONING_STEPS, CONTEXT_HISTORY_TURNS,
    LLM_TEMPERATURE_STRICT
)
from database import MedicalKnowledgeDB
from document_processor import DocumentEmbedder
from search import MedicalSearchEngine, Reranker, QueryExpander


# === 页面配置 ===
st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)


# === 初始化组件（缓存） ===
@st.cache_resource
def init_components():
    """初始化所有组件并缓存"""
    db = MedicalKnowledgeDB(DB_PATH, COLLECTION_NAME)
    reranker = Reranker(RERANKER_MODEL)
    expander = QueryExpander(LLM_MODEL)
    search_engine = MedicalSearchEngine(db, reranker, expander)
    embedder = DocumentEmbedder(EMBEDDING_MODEL, BATCH_SIZE)
    return db, search_engine, embedder


db, search_engine, embedder = init_components()


# === 侧边栏：知识库管理 ===
with st.sidebar:
    st.header("📚 知识库管家")
    st.subheader("已学习的书籍")

    # 显示已存储的文件
    current_files = db.get_existing_files()

    if not current_files:
        st.caption("暂无数据，请上传。")
    else:
        for f in current_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📖 {f}")
            with col2:
                if st.button("删", key=f"del_{f}", help=f"删除《{f}》"):
                    success, error = db.delete_file(f)
                    if success:
                        st.success(f"已删除")
                        st.rerun()
                    else:
                        st.error(f"失败: {error}")

    st.divider()

    # 调试模式开关
    debug_mode = st.toggle('开启调试模式 (Debug)', value=False)

    # 上传新文件
    st.subheader("上传新书")
    uploaded_files = st.file_uploader(
        "支持 Markdown",
        type=["md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("开始学习"):
            for file in uploaded_files:
                with st.spinner(f"正在处理 {file.name}..."):
                    # 读取文件内容
                    content = file.read().decode("utf-8")

                    # 定义进度回调
                    progress_bar = st.progress(0, text=f"正在学习新书: {file.name}...")

                    def update_progress(progress, text):
                        progress_bar.progress(progress, text=text)

                    # 处理文件
                    success, info = embedder.process_file(
                        content, file.name, db, update_progress
                    )

                    progress_bar.empty()

                    # 显示结果
                    if success:
                        st.balloons()
                        st.success(f"存入 {info} 条知识片段。")
                        st.rerun()
                    elif info == "EXIST":
                        st.warning(f"《{file.name}》已经学过了，跳过。")
                    else:
                        st.error(f"失败: {info}")


# === 主聊天界面 ===
st.title(f"👨‍⚕️ {APP_TITLE}")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "你好，我是你的医学助手。已学知识请看左侧列表。"
    }]

# 显示对话历史
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👨‍⚕️").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🧑‍🎓").write(msg["content"])

# 用户输入
prompt = st.chat_input("请输入病例...")

if prompt:
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍🎓").write(prompt)

    # AI 回复
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        response_container = st.empty()

        with st.status("正在推理...", expanded=True) as status:
            # 构建对话上下文
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            # 添加历史对话（最近 N 轮）
            history_start = max(0, len(st.session_state.messages) - CONTEXT_HISTORY_TURNS * 2)
            for msg in st.session_state.messages[history_start:]:
                messages.append(msg)

            messages.append({"role": "user", "content": prompt})

            # ReAct 推理循环
            final_answer = ""
            last_action = ""

            for step in range(MAX_REASONING_STEPS):
                # 调用 LLM
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages,
                    options={'temperature': LLM_TEMPERATURE_STRICT}
                )
                ai_content = response['message']['content']
                st.markdown(f"*{ai_content}*")
                messages.append(response['message'])

                # 检测检索动作
                if "检索:" in ai_content or "检索：" in ai_content:
                    splitter = "检索:" if "检索:" in ai_content else "检索："
                    keyword = ai_content.split(splitter)[-1].split("\n")[0].strip()

                    # 避免重复检索
                    if keyword == last_action:
                        obs = "Observation: 已搜索过该词，无新信息。请尝试总结。"
                    else:
                        st.info(f"查阅: {keyword}")

                        # 执行搜索
                        res, logs = search_engine.search(keyword, debug=debug_mode)

                        # 显示调试信息
                        if debug_mode:
                            with st.expander("📊 Rerank 打分详情", expanded=True):
                                for log in logs:
                                    try:
                                        # 只有以 "[" 开头的日志才可能是打分日志
                                        if log.strip().startswith("["):
                                            # 提取分数
                                            score_str = log.split(']')[0].replace('[', '')
                                            score = float(score_str)

                                            # 根据分数显示颜色
                                            if score > -10:
                                                st.success(log)  # 高分绿底
                                            else:
                                                st.text(log)  # 低分灰底
                                        else:
                                            # 其他类型的日志（如查询词扩展），用蓝色显示
                                            st.info(log)
                                    except Exception:
                                        # 万一解析还是崩了，兜底显示纯文本
                                        st.text(log)

                        # 将检索结果传给 LLM
                        obs = f"Observation: {res}"
                        last_action = keyword

                    messages.append({"role": "user", "content": obs})

                # 检测最终答案
                if "Final Answer" in ai_content:
                    final_answer = ai_content.split("Final Answer")[-1].lstrip(":").lstrip("：").strip()
                    status.update(label="✅ 完成", state="complete", expanded=False)
                    break

                # 直接回答（未使用检索）
                if "检索" not in ai_content and len(ai_content) > 20:
                    final_answer = ai_content
                    status.update(label="✅ 完成（直接回答）", state='complete', expanded=False)
                    break

            # 兜底处理
            if not final_answer:
                if len(ai_content) > 10:
                    final_answer = ai_content
                    status.update(label="⚠️ 强制结束（取最后回复）", state="complete", expanded=False)
                else:
                    final_answer = "抱歉，我未查到相关资料，未能得出明确结论。"
                    status.update(label="❌ 无结论", state="error", expanded=False)

        # 流式输出结果
        if final_answer:
            def stream_text():
                for word in final_answer.split():
                    yield word + " "
                    time.sleep(0.02)

            response_container.write_stream(stream_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })
