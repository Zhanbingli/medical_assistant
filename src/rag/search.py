"""
搜索模块 - 实现多路召回和 Rerank 功能
"""
import ollama
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Any
import logging

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    MULTI_QUERY_COUNT,
    RECALL_N_RESULTS,
    RERANK_TOP_K,
    RERANK_THRESHOLD,
    LLM_TEMPERATURE_CREATIVE,
    QUERY_EXPANSION_PROMPT
)
from .database import MedicalKnowledgeDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryExpander:
    """查询扩展器 - 生成相关的医学关键词 (Singleton)"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QueryExpander, cls).__new__(cls)
        return cls._instance

    def __init__(self, llm_model: str = LLM_MODEL):
        """
        初始化查询扩展器
        """
        if hasattr(self, 'llm_model'):
           return

        self.llm_model = llm_model
        logger.info(f"查询扩展器已初始化: 模型={llm_model}")

    def expand(self, query: str, count: int = MULTI_QUERY_COUNT) -> List[str]:
        """
        扩展查询词，生成多个相关关键词

        Args:
            query: 原始查询
            count: 生成关键词数量

        Returns:
            包含原始查询和扩展查询的列表
        """
        prompt = QUERY_EXPANSION_PROMPT.format(query=query, count=count)

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': LLM_TEMPERATURE_CREATIVE}
            )

            queries = response['message']['content'].strip().split('\n')
            # 清理序号和空白
            clean_queries = [
                q.split('.')[-1].strip()
                for q in queries
                if q.strip()
            ]

            # 原始查询 + 扩展查询
            result = [query] + clean_queries[:count]
            logger.info(f"查询扩展: {query} -> {result}")
            return result

        except Exception as e:
            logger.error(f"查询扩展失败: {e}")
            return [query]


class Reranker:
    """重排序器 - 使用 CrossEncoder 对检索结果打分 (Singleton)"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = RERANKER_MODEL):
        """
        初始化 Reranker
        """
        if hasattr(self, 'model'):
            return

        self.model_name = model_name
        logger.info(f"正在加载 Rerank 模型: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Rerank 模型加载完成")

    def rerank(
        self,
        query: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            metadatas: 元数据列表

        Returns:
            (文档, 分数, 元数据) 的列表，按分数降序排列
        """
        if not documents:
            return []

        # 构造查询-文档对
        pairs = [[query, doc] for doc in documents]

        # 预测分数
        scores = self.model.predict(pairs)

        # 组合并排序
        scored_docs = list(zip(documents, scores, metadatas))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Rerank 完成: {len(scored_docs)} 条结果")
        return scored_docs


class MedicalSearchEngine:
    """医学搜索引擎 - 整合多路召回和 Rerank"""

    def __init__(
        self,
        db: MedicalKnowledgeDB,
        reranker: Reranker,
        expander: QueryExpander
    ):
        """
        初始化搜索引擎

        Args:
            db: 数据库实例
            reranker: 重排序器实例
            expander: 查询扩展器实例
        """
        self.db = db
        self.reranker = reranker
        self.expander = expander
        logger.info("医学搜索引擎已初始化")

    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        try:
            response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            return []

    def _multi_recall(
        self,
        queries: List[str]
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        多路召回：对多个查询分别检索并去重

        Args:
            queries: 查询列表

        Returns:
            (文档列表, 元数据列表, 调试日志列表)
        """
        all_documents = []
        all_metadatas = []
        seen_docs = set()
        debug_logs = []

        for q in queries:
            try:
                # 生成嵌入
                embedding = self._generate_embedding(q)
                if not embedding:
                    continue

                # 检索
                results = self.db.query(embedding, n_results=RECALL_N_RESULTS)

                # 提取结果
                if results['documents'] and results['documents'][0]:
                    docs = results['documents'][0]
                    # 容错处理：如果没有 metadata，填充空字典
                    metas = results.get('metadatas', [[]])[0]
                    if not metas:
                        metas = [{}] * len(docs)

                    # 去重
                    for doc, meta in zip(docs, metas):
                        if doc not in seen_docs:
                            all_documents.append(doc)
                            all_metadatas.append(meta)
                            seen_docs.add(doc)

            except Exception as e:
                debug_logs.append(f"⚠️ 检索关键词 '{q}' 时出错: {e}")

        return all_documents, all_metadatas, debug_logs

    def search(
        self,
        query: str,
        debug: bool = False
    ) -> Tuple[str, List[str]]:
        """
        执行完整的搜索流程：查询扩展 -> 多路召回 -> Rerank

        Args:
            query: 用户查询
            debug: 是否返回调试信息

        Returns:
            (检索结果文本, 调试日志列表)
        """
        debug_logs = []

        try:
            debug_logs.append(f"🔍 原始查询: {query}")

            # 1. 查询扩展
            expanded_queries = self.expander.expand(query, count=MULTI_QUERY_COUNT)
            if debug:
                debug_logs.append(f"🧠 扩展关键词: {expanded_queries}")

            # 2. 多路召回
            all_documents, all_metadatas, recall_logs = self._multi_recall(expanded_queries)
            debug_logs.extend(recall_logs)

            if not all_documents:
                logger.info("未找到相关资料")
                return "未找到相关资料。", debug_logs

            debug_logs.append(f"∑ 共召回 {len(all_documents)} 条不重复片段，开始 Rerank...")

            # 3. Rerank 重排序
            scored_docs = self.reranker.rerank(query, all_documents, all_metadatas)

            # 4. 筛选高质量结果
            top_k_docs = []

            for doc, score, meta in scored_docs:
                source_name = meta.get('source', '未知来源') if meta else '未知来源'

                # 记录详细日志
                if debug:
                    preview = doc[:20].replace('\n', ' ')
                    log_str = f"[{score:.2f}] {source_name}: {preview}..."
                    debug_logs.append(log_str)

                # 筛选逻辑：阈值过滤 + Top-K
                if len(top_k_docs) < RERANK_TOP_K and score > RERANK_THRESHOLD:
                    doc_with_source = f"{doc}\n[来源: {source_name}]"
                    top_k_docs.append(doc_with_source)

            if not top_k_docs:
                logger.info("资料相关度较低")
                return "资料相关度较低，建议补充细节。", debug_logs

            result_text = "\n---\n".join(top_k_docs)
            logger.info(f"搜索完成: 返回 {len(top_k_docs)} 条高质量结果")
            return result_text, debug_logs

        except Exception as e:
            error_msg = f"检索过程发生错误: {str(e)}"
            logger.error(error_msg)
            return error_msg, [str(e)]
