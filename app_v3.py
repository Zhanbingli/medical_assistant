"""
AI 循证医学助手 - 微信风格专业版
- 微信式对话界面
- 并行检索：知识库 + PubMed + 模型知识
- 置信度显示
"""

import streamlit as st
import logging
import re
from typing import Dict, List, Any

from config import (
    DB_PATH, COLLECTION_NAME,
    RERANKER_MODEL, EMBEDDING_MODEL,
    BATCH_SIZE, MEDGEMMA_MODEL,
    RECALL_N_RESULTS
)
from src.rag.database import MedicalKnowledgeDB
from src.rag.loader import DocumentEmbedder
from src.rag.search import MedicalSearchEngine, Reranker, QueryExpander
from src.agent.tools import SearchTool
from src.agent.core import MedicalAgent
from src.utils.safety import SafetyEnhancer
from src.utils.web_search import PubMedSearch
from src.retrieval import ParallelRetriever, ResultFuser, FusedResult, FusionStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI 循证医学助手",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --primary-light: #dbeafe;
    --success: #16a34a;
    --text: #1f2937;
    --text-light: #6b7280;
    --bg: #f0f2f5;
    --card-bg: #ffffff;
    --border: #e5e7eb;
}

* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--card-bg); border-right: 1px solid var(--border); }

.wechat-message { margin: 0.75rem 0; }
.wechat-user { display: flex; justify-content: flex-end; align-items: flex-start; }
.wechat-ai { display: flex; justify-content: flex-start; align-items: flex-start; }

.wechat-avatar {
    width: 36px; height: 36px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
}
.wechat-user .wechat-avatar { display: none; }

.wechat-user .wechat-bubble {
    background: var(--primary); color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem; max-width: 70%;
    line-height: 1.6;
}

.wechat-ai .wechat-bubble {
    background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.25rem; max-width: 75%;
    line-height: 1.7; margin-left: 0.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

.thinking {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
    border: 1px solid #93c5fd; border-radius: 12px;
    padding: 1rem 1.25rem; margin: 1rem 0 1rem 2.5rem;
    display: flex; align-items: center; gap: 0.75rem;
}

.thinking-icon { font-size: 1.25rem; animation: pulse 2s infinite; }
.thinking-text { color: #1e40af; font-weight: 500; }

.welcome-card {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
}

.stat-card { background: var(--primary-light); border-radius: 8px; padding: 0.75rem; text-align: center; }

.confidence-badge {
    display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px;
    font-size: 0.75rem; font-weight: 600; margin-left: 0.5rem;
}
.confidence-high { background: #dcfce7; color: #16a34a; }
.confidence-medium { background: #fef9c3; color: #ca8a04; }
.confidence-low { background: #fee2e2; color: #dc2626; }

.source-badge {
    display: inline-block; padding: 0.2rem 0.4rem; border-radius: 4px;
    font-size: 0.7rem; margin-right: 0.25rem;
}
.source-knowledge { background: #e0e7ff; color: #4338ca; }
.source-pubmed { background: #dcfce7; color: #16a34a; }
.source-model { background: #f3f4f6; color: #6b7280; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []


def init_components():
    @st.cache_resource
    def _init():
        db = MedicalKnowledgeDB(DB_PATH, COLLECTION_NAME)
        reranker = Reranker(RERANKER_MODEL)
        expander = QueryExpander()
        search_engine = MedicalSearchEngine(db, reranker, expander)
        embedder = DocumentEmbedder(EMBEDDING_MODEL, BATCH_SIZE)
        
        search_tool = SearchTool(search_engine)
        agent = MedicalAgent(search_tool)
        safety_enhancer = SafetyEnhancer()
        pubmed_search = PubMedSearch()
        
        parallel_retriever = ParallelRetriever(
            db=db,
            pubmed_search=pubmed_search,
            embed_model=EMBEDDING_MODEL,
            recall_count=RECALL_N_RESULTS
        )
        
        result_fuser = ResultFuser(
            similarity_threshold=0.75,
            max_results=5
        )
        
        return {
            "db": db,
            "embedder": embedder,
            "agent": agent,
            "safety_enhancer": safety_enhancer,
            "pubmed_search": pubmed_search,
            "parallel_retriever": parallel_retriever,
            "result_fuser": result_fuser
        }
    return _init()


def render_sidebar(components):
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🩺</span>
            <h2 style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">AI 循证医学助手</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📚 知识库")
        stats = components["db"].get_collection_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class="stat-card"><div class="stat-value" style="font-size: 1.25rem;">{stats['total_chunks']}</div><div class="stat-label">知识片段</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="stat-card" style="background: #dcfce7;"><div class="stat-value" style="color: var(--success); font-size: 1.25rem;">{stats['total_files']}</div><div class="stat-label">医学教材</div></div>""", unsafe_allow_html=True)
        
        if stats['files']:
            with st.expander("📖 已学习教材", expanded=False):
                for f in sorted(stats['files']):
                    st.markdown(f"<div style='padding: 0.25rem 0; color: var(--text-light); font-size: 0.85rem;'>📄 {f}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.expander("🔬 联网检索 (PubMed)", expanded=False):
            st.markdown("**仅检索英文医学权威期刊**")
            st.markdown("• 并行检索知识库、PubMed、模型知识")
            st.markdown("• 显示置信度评分")
        
        st.markdown("---")
        
        st.subheader("📤 上传教材")
        uploaded_files = st.file_uploader("选择 Markdown 文件", type=["md"], accept_multiple_files=True)
        
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files) / 1024 / 1024
            st.info(f"已选择 {len(uploaded_files)} 个 ({total_size:.2f} MB)")
            if st.button("🚀 开始学习", type="primary", use_container_width=True):
                process_files(uploaded_files, components)
        
        return {
            "web_search_enabled": st.toggle("🌐 启用 PubMed 检索", value=True, help="启用后同时检索 PubMed 英文医学文献")
        }


def process_files(uploaded_files, components):
    db, embedder = components["db"], components["embedder"]
    success_count = 0
    
    for file in uploaded_files:
        with st.status(f"处理 {file.name}...", expanded=True) as status:
            try:
                content = file.read().decode("utf-8", errors='ignore')
                progress = status.empty()
                def update_progress(p, t):
                    progress.progress(p, text=t)
                success, info = embedder.process_file(content, file.name, db, update_progress)
                if success:
                    success_count += 1
                    status.update(label=f"✅ {file.name}", state="complete")
                elif info == "EXIST":
                    status.update(label=f"⚠️ {file.name} 已存在", state="complete")
                else:
                    status.update(label=f"❌ {file.name} 失败", state="error")
            except:
                status.update(label=f"❌ 处理失败", state="error")
    
    if success_count > 0:
        st.success(f"🎉 成功学习 {success_count} 本教材")
        st.rerun()


def render_thinking(message="正在分析"):
    """渲染思考动画"""
    st.markdown(f"""
    <div class="thinking">
        <span class="thinking-icon">🧠</span>
        <span class="thinking-text">{message}</span>
        <span style="margin-left: auto; color: #3b82f6;">
            <span style="animation: pulse 1.5s infinite;">·</span>
            <span style="animation: pulse 1.5s infinite; animation-delay: 0.2s;">·</span>
            <span style="animation: pulse 1.5s infinite; animation-delay: 0.4s;">·</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_parallel_searching(query):
    """渲染并行检索动画"""
    st.markdown(f"""
    <div class="thinking" style="background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%); border-color: #93c5fd;">
        <span style="font-size: 1.25rem;">🔍</span>
        <span style="color: #1e40af; font-weight: 500;">并行检索中</span>
        <span style="color: #6b7280; font-size: 0.85rem;">知识库 · PubMed · 模型知识</span>
        <span style="margin-left: auto; font-family: monospace; color: #3b82f6; font-size: 0.85rem;">
            「{query[:20]}...」
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_thought_card(steps, fused_results: List[FusedResult], fusion_stats: FusionStats):
    """渲染思考过程卡片（统一卡片）"""
    with st.expander("🤔 推理过程与来源", expanded=True):
        st.markdown("### 📊 检索统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("知识库", fusion_stats.knowledge_base_count)
        col2.metric("PubMed", fusion_stats.pubmed_count)
        col3.metric("模型知识", fusion_stats.model_count)
        col4.metric("融合结果", fusion_stats.final_count)
        
        st.divider()
        
        st.markdown("### 📚 融合结果")
        for i, result in enumerate(fused_results, 1):
            conf_percent = int(result.confidence * 100)
            if conf_percent >= 70:
                conf_class = "confidence-high"
            elif conf_percent >= 40:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            source_badges = []
            for source in result.sources:
                if source == "knowledge_base":
                    source_badges.append('<span class="source-badge source-knowledge">📖 知识库</span>')
                elif source == "pubmed":
                    source_badges.append('<span class="source-badge source-pubmed">🔬 PubMed</span>')
                elif source == "model":
                    source_badges.append('<span class="source-badge source-model">🧠 模型</span>')
            
            st.markdown(f"""
            <div style="background: #f8fafc; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; margin-right: 0.5rem;">结果 {i}</span>
                    <span class="confidence-badge {conf_class}">{conf_percent}% 置信度</span>
                    <span style="margin-left: auto;">{"".join(source_badges)}</span>
                </div>
                <div style="color: var(--text); line-height: 1.6;">{result.content[:300]}{"..." if len(result.content) > 300 else ""}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### 🔍 思考步骤")
        for i, step in enumerate(steps):
            st.write(f"**步骤 {i+1}: {step['title']}**")
            if "content" in step:
                st.text(step["content"])
            elif "desc" in step:
                st.write(step["desc"])
            if i < len(steps) - 1:
                st.divider()


def render_user_message(content):
    st.markdown(f"""<div class="wechat-message wechat-user"><div class="wechat-bubble">{content}</div></div>""", unsafe_allow_html=True)


def render_ai_message(content, msg_type="general"):
    bubble_content = content
    if msg_type == "diagnosis":
        bubble_content = f"""<div style="margin-bottom: 0.5rem;">📊 <strong>诊断建议</strong></div>{content}"""
    elif msg_type == "checklist":
        bubble_content = f"""<div style="margin-bottom: 0.5rem;">🔬 <strong>检查建议</strong></div>{content}"""
    
    st.markdown(f"""
    <div class="wechat-message wechat-ai">
        <div class="wechat-avatar" style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);">🩺</div>
        <div class="wechat-bubble">{bubble_content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome():
    st.markdown(f"""
    <div class="welcome-card">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🩺</span>
            <div>
                <div style="font-weight: 600; font-size: 1.1rem;">AI 循证医学助手</div>
                <div style="opacity: 0.9; font-size: 0.85rem;">专业医学知识库问答系统</div>
            </div>
        </div>
        <div style="line-height: 1.8; font-size: 0.95rem;">
            您好！我可以帮助您：
            <br>🔍 基于医学知识库回答临床问题
            <br>📊 提供诊断建议和鉴别诊断
            <br>🔬 检索 PubMed 英文医学权威期刊
            <br>📈 显示置信度评分
            <br><br>
            请描述患者症状，我将为您提供专业的医学建议。
        </div>
    </div>
    """, unsafe_allow_html=True)


def clean_response(text: str) -> str:
    """清理响应中的HTML标签和系统信息"""
    # 移除 <thought>...</thought> 标签
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
    # 移除 <unusedXX> 标签
    text = re.sub(r'<unused\d+>', '', text)
    # 移除 MedGemma 的思考过程标记
    text = re.sub(r'Here\'s a thinking process.*?:\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>.*', '', text)
    text = re.sub(r'^\s*1\.\s*\*\*.*?\.\.\.\s*', '', text)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余的空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def render_error():
    st.markdown(f"""
    <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 12px; margin: 1rem 0 1rem 2.5rem; padding: 1rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <span>⚠️</span>
            <span style="font-weight: 600; color: #dc2626;">抱歉</span>
        </div>
        <div style="color: #7f1d1d;">
            我遇到了一些问题，请稍后重试或重新描述症状。
        </div>
    </div>
    """, unsafe_allow_html=True)


def generate_answer_with_context(query: str, context: str, history: List[Dict]) -> str:
    """使用 MedGemma adapter 直接生成回答"""
    from src.llm import get_adapter
    
    adapter = get_adapter()
    
    messages = []
    if history:
        for msg in history[-4:]:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    
    # 简化上下文，只保留主要内容
    context_lines = []
    for line in context.split('\n'):
        line = line.strip()
        if line and not line.startswith('[') and len(line) > 10:
            context_lines.append(line)
    simplified_context = '\n'.join(context_lines[:15])  # 只保留前15行
    
    prompt = f"""基于以下医学信息，用中文回答问题。直接给出简洁的答案，不要添加思考过程、编号列表或"以下是"等开场白。

问题: {query}

信息:
{simplified_context}

请直接回答（用1-2个段落）："""

    messages.append({"role": "user", "content": prompt})
    
    try:
        response = adapter.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        
        content = response.get('message', {}).get('content', '')
        
        # 清理 MedGemma 的思考标签和格式
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        content = re.sub(r'<unused\d+>', '', content)
        content = re.sub(r'Here\'s a thinking process.*?:\s*', '', content, flags=re.DOTALL)
        content = re.sub(r'<\|im_end\|>.*', '', content)
        content = re.sub(r'^\s*\d+\.\s+\*\*.*?\*\*\s*', '', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'^以下是.*?：\s*', '', content)
        content = content.strip()
        
        return content if content else ""
        
    except Exception as e:
        logger.error(f"生成回答失败: {e}")
        return ""


def process_query_parallel(
    components,
    user_text: str,
    history: List[Dict],
    web_search_enabled: bool
) -> Dict[str, Any]:
    """并行处理查询"""
    steps = []
    fused_results = []
    fusion_stats = FusionStats()
    final_answer = ""
    
    steps.append({
        "title": "🧠 分析问题",
        "desc": f"理解用户描述: {user_text[:30]}..."
    })
    
    parallel_retriever = components["parallel_retriever"]
    result_fuser = components["result_fuser"]
    
    steps.append({
        "title": "🔍 并行检索",
        "desc": "同时搜索知识库、PubMed、模型知识"
    })
    
    retrieval_results = parallel_retriever.retrieve_all(user_text, history)
    
    steps.append({
        "title": "🔗 融合结果",
        "desc": "合并去重，按优先级排序"
    })
    
    fused_results, fusion_stats = result_fuser.fuse(retrieval_results)
    
    if fused_results:
        context = result_fuser.build_fused_context(fused_results, user_text)
        
        steps.append({
            "title": "✨ 生成回答",
            "desc": f"基于 {len(fused_results)} 条检索结果生成专业建议"
        })
        
        final_answer = generate_answer_with_context(
            user_text, context, history
        )
    else:
        steps.append({
            "title": "⚠️ 无检索结果",
            "desc": "尝试使用模型知识直接回答"
        })
        
        agent = components["agent"]
        for event_type, data in agent.run(user_text, history):
            if event_type == "FINAL_ANSWER":
                final_answer = data
                break
    
    if not final_answer:
        final_answer = "抱歉，我未能找到相关的医学信息。\n\n**建议：**\n1. 补充更多症状描述\n2. 使用更专业的医学术语\n3. 上传相关医学教材"
    
    return {
        "steps": steps,
        "fused_results": fused_results,
        "fusion_stats": fusion_stats,
        "final_answer": final_answer
    }


def render_chat(components, settings):
    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 1rem 1.5rem; 
                margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 1.25rem; display: flex; align-items: center; gap: 0.5rem;">
            💬 对话咨询
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    templates = [
        ("🤒 发热", "患者发热3天，体温最高38.5°C，伴有寒战和头痛"),
        ("🫁 咳嗽", "患者咳嗽1周，伴有胸闷和呼吸困难，无发热"),
        ("❤️ 胸痛", "患者胸痛2小时，伴有呼吸困难和出汗"),
        ("🤕 腹痛", "患者腹痛6小时，伴有恶心和呕吐，右下腹压痛明显")
    ]
    
    for col, (icon, template) in zip([col1, col2, col3, col4], templates):
        if col.button(icon, use_container_width=True, key=f"tmpl_{icon}"):
            st.session_state.suggested_input = template
    
    if "suggested_input" in st.session_state:
        user_text = st.session_state.suggested_input
        del st.session_state.suggested_input
    else:
        user_text = None
    
    render_welcome()
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_ai_message(msg["content"], msg.get("type", "general"))
            if msg.get("fused_results"):
                render_thought_card(
                    msg.get("thought_steps", []),
                    msg.get("fused_results", []),
                    msg.get("fusion_stats", FusionStats())
                )
    
    prompt = st.chat_input("描述患者症状（如：发热伴咳嗽3天，体温38.5°C）...")
    if user_text or prompt:
        user_text = user_text or prompt
        
        st.session_state.messages.append({"role": "user", "content": user_text})
        render_user_message(user_text)
        
        with st.container():
            thinking_placeholder = st.empty()
            render_parallel_searching(user_text[:20] + "...")
            
            try:
                result = process_query_parallel(
                    components,
                    user_text,
                    st.session_state.messages[:-1],
                    settings["web_search_enabled"]
                )
                
                full_response = clean_response(result["final_answer"])
                
                thinking_placeholder.markdown("")
                
                render_thought_card(
                    result["steps"],
                    result["fused_results"],
                    result["fusion_stats"]
                )
                
                msg_type = "general"
                if "诊断" in full_response and ("1." in full_response or "可能" in full_response):
                    msg_type = "diagnosis"
                elif "检查" in full_response or "实验室" in full_response:
                    msg_type = "checklist"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "type": msg_type,
                    "thought_steps": result["steps"],
                    "fused_results": result["fused_results"],
                    "fusion_stats": result["fusion_stats"]
                })
                
                render_ai_message(full_response, msg_type)
            
            except Exception as e:
                thinking_placeholder.markdown("")
                render_error()
                logger.error(f"处理出错: {e}")


def main():
    init_session_state()
    components = init_components()
    settings = render_sidebar(components)
    render_chat(components, settings)


if __name__ == "__main__":
    main()
