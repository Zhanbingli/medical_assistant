import streamlit as st
import ollama
import chromadb
import uuid
import os
from sentence_transformers import CrossEncoder
import time

# === 1. 基础配置与数据库初始化 ===
st.set_page_config(page_title="AI 循证医学助手", layout="wide")

@st.cache_resource
def init_memory():
    # 数据持久化存储
    client = chromadb.PersistentClient(path="./medical_db")
    collection = client.get_or_create_collection(name="medical_knowledge")
    return collection

memory_collection = init_memory()

@st.cache_resource
def init_reranker():
    print("Loading Rerank model...")
    return CrossEncoder('BAAI/bge-reranker-base')
reranker = init_reranker()


# === 2. 核心功能函数 ===

def get_existing_files():
    """获取数据库中已存储的所有文件名"""
    try:
        data = memory_collection.get(include=['metadatas'])
        if not data['metadatas']:
            return set()
        files = set([m.get('source') for m in data['metadatas'] if m])
        return files
    except Exception:
        return set()

def delete_file_from_db(filename):
    """从数据库中删除指定文件的所有片段"""
    try:
        memory_collection.delete(where={"source": filename})
        return True
    except Exception as e:
        return str(e)

def split_markdown_smart(text, chunk_size=600): # 修正函数名拼写
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    current_headers = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            level = stripped.count('#')
            title = stripped.strip('#').strip()
            if len(current_headers) >= level:
                current_headers = current_headers[:level-1]
            current_headers.append(title)

            # 标题也作为正文的一部分，保证上下文连贯
            current_chunk.append(line)
            current_length += len(line)
            continue

        current_chunk.append(line)
        current_length += len(line)

        if current_length > chunk_size:
            header_context = " > ".join(current_headers)
            full_text = f"【章节：{header_context}】\n" + "\n".join(current_chunk)
            chunks.append(full_text)

            # 简单的重叠策略：保留最后3行
            current_chunk = current_chunk[-3:]
            current_length = sum(len(l) for l in current_chunk)

    if current_chunk:
        header_context = " > ".join(current_headers) # 统一分隔符格式
        full_text = f"【章节：{header_context}】\n" + "\n".join(current_chunk)
        chunks.append(full_text)
    return chunks


def save_uploaded_file(uploaded_file):
    """保存文件"""
    existing_files = get_existing_files()
    if uploaded_file.name in existing_files:
        return False, "EXIST"

    try:
        content = uploaded_file.read().decode("utf-8")
        raw_chunks = split_markdown_smart(content, chunk_size=600)
        total_chunks = len(raw_chunks)

        if total_chunks == 0: return False, "EMPTY"

        progress_bar = st.progress(0, text=f"正在学习新书: {uploaded_file.name}...")

        ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
        BATCH_SIZE = 20

        for i, chunk in enumerate(raw_chunks):
            if len(chunk) < 10: continue

            try:
                response = ollama.embeddings(model='bge-m3', prompt=chunk)
                ids_batch.append(str(uuid.uuid4()))
                embeddings_batch.append(response['embedding'])
                documents_batch.append(chunk)
                metadatas_batch.append({"source": uploaded_file.name, "chunk_index": i})

                if len(ids_batch) >= BATCH_SIZE:
                    memory_collection.add(ids=ids_batch, embeddings=embeddings_batch, documents=documents_batch, metadatas=metadatas_batch)
                    ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
            except Exception as e:
                return False, str(e)

            progress_bar.progress((i + 1) / total_chunks)

        if ids_batch:
            memory_collection.add(ids=ids_batch, embeddings=embeddings_batch, documents=documents_batch, metadatas=metadatas_batch)

        progress_bar.empty()
        return True, total_chunks
    except Exception as e:
        return False, str(e)

def generate_search_queries(original_query):
    """生成扩展查询词"""
    prompt = f"""
    你是一个医学搜索优化专家。
    请根据用户的口语化描述，生成 3 个用于检索医学教材的专业关键词或短语。
    用户问题: "{original_query}"
    要求:
    1. 包含医学术语。
    2. 包含可能的关联疾病。
    3. 只输出 3 行关键词，不要有序号。
    """
    try:
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.7}
        )
        queries = response['message']['content'].strip().split('\n')
        clean_queries = [q.split('.')[-1].strip() for q in queries if q.strip()]
        return [original_query] + clean_queries[:3]
    except:
        return [original_query]

def search_memory(query, debug=False):
    """多路召回 + Rerank 核心函数"""
    debug_logs = []
    try:
        debug_logs.append(f"🔍 原始查询: {query}")

        # 1. 扩展查询
        expanded_queries = generate_search_queries(query) # 修正拼写
        if debug:
            debug_logs.append(f"🧠 扩展关键词: {expanded_queries}")

        all_documents = []
        all_metadatas = []
        seen_docs = set()

        # 2. 多路召回
        for q in expanded_queries:
            try:
                response = ollama.embeddings(model='bge-m3', prompt=q)
                results = memory_collection.query(query_embeddings=[response['embedding']], n_results=5)

                if results['documents'] and results["documents"][0]:
                    docs = results["documents"][0]
                    # 容错处理：如果没有 metadata，填充空字典
                    metas = results['metadatas'][0] if results['metadatas'] else [{}] * len(docs)

                    for doc, meta in zip(docs, metas):
                        if doc not in seen_docs:
                            all_documents.append(doc)
                            all_metadatas.append(meta)
                            seen_docs.add(doc)
            except Exception as e:
                debug_logs.append(f"⚠️ 检索关键词 '{q}' 时出错: {e}")

        if not all_documents:
            return "未找到相关资料。", debug_logs

        debug_logs.append(f"∑ 共召回 {len(all_documents)} 条不重复片段，开始 Rerank...")

        # 3. 重排 (Rerank)
        pairs = [[query, doc] for doc in all_documents]
        scores = reranker.predict(pairs)
        scored_docs = sorted(zip(all_documents, scores, all_metadatas), key=lambda x: x[1], reverse=True)

        top_k_docs = []

        # 4. 筛选与日志
        for doc, score, meta in scored_docs:
            source_name = meta.get('source', '未知来源') if meta else '未知来源'

            # 记录详细日志用于 UI 展示
            if debug:
                preview = doc[:20].replace('\n', ' ')
                log_str = f"[{score:.2f}] {source_name}: {preview}..."
                debug_logs.append(log_str)

            # 筛选逻辑：阈值 -10，最多取 3 条
            if len(top_k_docs) < 3 and score > -10:
                doc_with_source = f"{doc}\n[来源: {source_name}]"
                top_k_docs.append(doc_with_source)

        if not top_k_docs:
            return "资料相关度较低，建议补充细节。", debug_logs

        return "\n---\n".join(top_k_docs), debug_logs

    except Exception as e:
        return f"检索过程发生错误: {str(e)}", [str(e)]


# === 3. UI 界面布局 ===

with st.sidebar:
    st.header("📚 知识库管家")
    st.subheader("已学习的书籍")
    current_files = get_existing_files()

    if not current_files:
        st.caption("暂无数据，请上传。")
    else:
        for f in current_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📖 {f}")
            with col2:
                if st.button("删", key=f"del_{f}", help=f"删除《{f}》"):
                    res = delete_file_from_db(f)
                    if res is True:
                        st.success(f"已删除")
                        st.rerun()
                    else:
                        st.error(f"失败: {res}")

    st.divider()
    debug_mode = st.toggle('开启调试模式 (Debug)', value=False) # 修正文案

    st.subheader("上传新书")
    uploaded_files = st.file_uploader("支持 Markdown", type=["md"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("开始学习"):
            for file in uploaded_files:
                with st.spinner(f"正在处理 {file.name}..."):
                    success, info = save_uploaded_file(file)
                    if success:
                        st.balloons()
                        st.success(f"存入 {info} 条知识片段。")
                        st.rerun()
                    elif info == "EXIST":
                        st.warning(f"《{file.name}》已经学过了，跳过。")
                    else:
                        st.error(f"失败: {info}")

# === 4. 主聊天界面 ===

st.title("👨‍⚕️ AI 循证医学助手")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，我是你的医学助手。已学知识请看左侧列表。"}]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"], avatar="👨‍⚕️").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🧑‍🎓").write(msg["content"])

prompt = st.chat_input("请输入病例...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍🎓").write(prompt)

    with st.chat_message("assistant", avatar="👨‍⚕️"):
        response_container = st.empty()
        with st.status("正在推理...", expanded=True) as status:

            system_prompt = """
            你是一个必须查阅知识库的医学AI助手。

            【铁律 - 必须遵守】：
            1. **第一步必须是检索**：无论用户问什么（只要和医学有关），你输出的第一句话必须是 "Action: 检索: [关键词]"。
            2. **禁止裸答**：在没有看到 Observation (检索结果) 之前，禁止给出任何建议，禁止反问用户。
            3. **强制关联**：如果用户问“怎么治”，而你不知道病因，先检索症状（如 "Action: 检索: 发热寒战"）来看看可能是什么病。

            【标准工作流】：
            User: 发热伴寒战
            Assistant: Thought: 用户提到症状，我必须先查库。
            Action: 检索: 发热伴寒战
            Observation: (系统返回知识)
            Final Answer: 根据资料，这可能是...
            """

            messages = [{"role": "system", "content": system_prompt}]
            # 上下文记忆：取最后2轮对话
            for msg in st.session_state.messages[-4:]:
                messages.append(msg)
            messages.append({"role": "user", "content": prompt})

            final_answer = ""
            last_action = ""

            for step in range(5):
                # 调用 LLM，Temperature=0 保证严谨
                response = ollama.chat(model='qwen2.5:7b', messages=messages, options={'temperature': 0})
                ai_content = response['message']['content']
                st.markdown(f"*{ai_content}*")
                messages.append(response['message'])

                if "检索:" in ai_content or "检索：" in ai_content:
                    splitter = "检索:" if "检索:" in ai_content else "检索："
                    keyword = ai_content.split(splitter)[-1].split("\n")[0].strip()

                    if keyword == last_action:
                        obs = "Observation: 已搜索过该词，无新信息。请尝试总结。"
                    else:
                        st.info(f"查阅: {keyword}")

                        # === 关键修正：直接使用 search_memory 的结果，不要重复检索 ===
                        res, logs = search_memory(keyword, debug=debug_mode)

                        # 调试信息展示
                        if debug_mode:
                            with st.expander("📊 Rerank 打分详情", expanded=True):
                                for log in logs:
                                    # === 修复开始：更健壮的日志解析 ===
                                    try:
                                        # 1. 只有以 "[" 开头的日志才可能是打分日志 (过滤掉 "🧠 扩展关键词" 这种)
                                        if log.strip().startswith("["):
                                            # 提取分数
                                            score_str = log.split(']')[0].replace('[', '')
                                            score = float(score_str)

                                            # 根据分数显示颜色
                                            if score > -10:
                                                st.success(log) # 高分绿底
                                            else:
                                                st.text(log)    # 低分灰底
                                        else:
                                            # 2. 其他类型的日志（如查询词扩展），用蓝色显示
                                            st.info(log)
                                    except Exception:
                                        # 3. 万一解析还是崩了，兜底显示纯文本，不让程序崩溃
                                        st.text(log)
                        # 将 Rerank 后的高质量内容传给 LLM
                        obs = f"Observation: {res}"
                        last_action = keyword

                    messages.append({"role": "user", "content": obs})

                if "Final Answer" in ai_content:
                    final_answer = ai_content.split("Final Answer")[-1].lstrip(":").lstrip("：").strip()
                    status.update(label="✅ 完成", state="complete", expanded=False)
                    break

                if "检索" not in ai_content and len(ai_content) > 20:
                    final_answer = ai_content
                    status.update(label="✅ 完成（直接回答）", state='complete', expanded=False)
                    break

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
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
