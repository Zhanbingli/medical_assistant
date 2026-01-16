"""
并行检索器 - 同时检索多个来源
"""

import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from config import RECALL_N_RESULTS, EMBEDDING_MODEL
from src.llm import get_adapter
from src.rag.search import QueryExpander

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    source: str  # 来源: "knowledge_base", "pubmed", "model"
    content: Any  # 内容
    relevance_score: float = 0.0  # 相关性分数
    metadata: Dict = None  # 元数据


class ParallelRetriever:
    """
    并行检索器
    
    同时从三个来源检索：
    1. 知识库 (本地 ChromaDB)
    2. PubMed (网络医学文献)
    3. 模型自身知识 (MedGemma)
    """
    
    def __init__(
        self,
        db,
        pubmed_search,
        embed_model: str = EMBEDDING_MODEL,
        recall_count: int = RECALL_N_RESULTS
    ):
        """
        初始化并行检索器
        
        Args:
            db: ChromaDB 数据库实例
            pubmed_search: PubMed 搜索实例
            embed_model: 嵌入模型名称
            recall_count: 知识库检索数量
        """
        self.db = db
        self.pubmed_search = pubmed_search
        self.embed_model = embed_model
        self.recall_count = recall_count
        self.adapter = get_adapter()
        self.expander = QueryExpander()
    
    def retrieve_all(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[RetrievalResult]]:
        """
        并行检索所有来源
        
        Args:
            query: 查询文本
            history: 对话历史
            
        Returns:
            Dict[str, List[RetrievalResult]]: 各来源的检索结果
        """
        results = {
            "knowledge_base": [],
            "pubmed": [],
            "model": []
        }
        
        # 并行执行三个检索任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_source = {
                "knowledge_base": executor.submit(
                    self._search_knowledge_base, query
                ),
                "pubmed": executor.submit(
                    self._search_pubmed, query
                ),
                "model": executor.submit(
                    self._query_model_knowledge, query, history
                )
            }
            
            for source, future in future_to_source.items():
                try:
                    result = future.result(timeout=30)
                    results[source] = result
                    logger.info(f"{source} 检索完成: {len(result)} 条结果")
                except Exception as e:
                    logger.error(f"{source} 检索失败: {e}")
                    results[source] = []
        
        return results
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询词"""
        expanded = self.expander.expand(query, count=3)
        if not expanded:
            return [query]
        return expanded[:4]  # 最多4个查询变体
    
    def _search_knowledge_base(self, query: str) -> List[RetrievalResult]:
        """
        检索知识库
        
        Args:
            query: 查询文本
            
        Returns:
            List[RetrievalResult]: 知识库检索结果
        """
        import ollama
        
        results = []
        
        try:
            # 查询扩展
            expanded_queries = self._expand_query(query)
            logger.info(f"查询扩展: {expanded_queries}")
            
            all_documents = []
            seen_contents = set()
            
            for q in expanded_queries:
                # 生成查询嵌入
                response = ollama.embeddings(
                    model=self.embed_model,
                    prompt=q
                )
                embedding = response.get('embedding')
                
                if not embedding:
                    continue
                
                # 查询 ChromaDB
                db_results = self.db.query(
                    query_embedding=embedding,
                    n_results=self.recall_count // len(expanded_queries) + 1
                )
                
                documents = db_results.get('documents', [[]])[0]
                metadatas = db_results.get('metadatas', [[]])[0]
                distances = db_results.get('distances', [[]])[0]
                
                for doc, meta, dist in zip(documents, metadatas, distances):
                    # 去重
                    content_hash = hash(doc[:200])
                    if content_hash in seen_contents:
                        continue
                    seen_contents.add(content_hash)
                    
                    all_documents.append((doc, meta, float(dist) if dist else 0.5))
            
            # 按距离排序并返回
            all_documents.sort(key=lambda x: x[2] if x[2] is not None else float('inf'))
            
            if not all_documents:
                return results
            
            min_dist = min(d[2] for d in all_documents if d[2] is not None)
            max_dist = max(d[2] for d in all_documents if d[2] is not None)
            dist_range = max_dist - min_dist if max_dist != min_dist else 1.0
            
            for i, (doc, meta, dist) in enumerate(all_documents[:self.recall_count]):
                if dist is not None and dist_range > 0:
                    score = 1.0 - (dist - min_dist) / dist_range
                    score = max(0, min(1, score))
                else:
                    score = 0.5
                
                results.append(RetrievalResult(
                    source="knowledge_base",
                    content=doc,
                    relevance_score=score,
                    metadata=meta or {}
                ))
            
            logger.info(f"知识库检索完成: {len(results)} 条结果")
            
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
        
        return results
    
    def _search_pubmed(self, query: str) -> List[RetrievalResult]:
        """
        检索 PubMed
        
        Args:
            query: 查询文本
            
        Returns:
            List[RetrievalResult]: PubMed 检索结果
        """
        results = []
        
        try:
            pubmed_results = self.pubmed_search.search(query, num_results=3)
            
            for r in pubmed_results:
                results.append(RetrievalResult(
                    source="pubmed",
                    content=f"**{r.title}**\n{r.snippet}",
                    relevance_score=0.8,  # PubMed 权威性高
                    metadata={
                        "url": r.url,
                        "source": "PubMed"
                    }
                ))
            
            logger.info(f"PubMed 检索完成: {len(results)} 条结果")
            
        except Exception as e:
            logger.error(f"PubMed 检索失败: {e}")
        
        return results
    
    def _query_model_knowledge(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[RetrievalResult]:
        """
        查询模型自身知识
        
        Args:
            query: 查询文本
            history: 对话历史
            
        Returns:
            List[RetrievalResult]: 模型知识检索结果
        """
        results = []
        
        try:
            # 构建提示
            prompt = f"""Based on your medical knowledge, provide relevant information about:

{query}

Please answer concisely (2-3 sentences)."""
            
            # 添加历史上下文
            messages = []
            if history:
                # 添加最近2轮历史
                for msg in history[-4:]:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            messages.append({"role": "user", "content": prompt})
            
            # 调用模型
            response = self.adapter.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.get('message', {}).get('content', '')
            
            if content:
                results.append(RetrievalResult(
                    source="model",
                    content=content,
                    relevance_score=0.5,  # 模型知识置信度中等
                    metadata={
                        "source": "MedGemma Model"
                    }
                ))
            
            logger.info(f"模型知识查询完成: {len(results)} 条结果")
            
        except Exception as e:
            logger.error(f"模型知识查询失败: {e}")
        
        return results
