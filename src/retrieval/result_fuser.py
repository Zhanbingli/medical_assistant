"""
结果融合器 - 合并多个检索来源的结果
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """融合后的结果"""
    content: str
    sources: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    confidence: float = 0.0
    source_priority: int = 0


@dataclass
class FusionStats:
    """融合统计信息"""
    total_sources: int = 0
    knowledge_base_count: int = 0
    pubmed_count: int = 0
    model_count: int = 0
    duplicates_removed: int = 0
    final_count: int = 0


class ResultFuser:
    """
    结果融合器
    
    功能：
    1. 去重 - 移除相似内容
    2. 优先级排序 - 按来源可靠性排序
    3. 置信度计算 - 计算最终置信度
    4. 内容提取 - 提取关键信息
    """
    
    SOURCE_PRIORITIES = {
        "pubmed": 2,      # 最高优先级 - 权威医学文献
        "knowledge_base": 1,  # 中等优先级 - 本地知识库
        "model": 0        # 较低优先级 - 模型自身知识
    }
    
    SOURCE_WEIGHTS = {
        "pubmed": 0.9,
        "knowledge_base": 0.7,
        "model": 0.5
    }
    
    SIMILARITY_THRESHOLD = 0.75
    
    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        max_results: int = 5
    ):
        """
        初始化结果融合器
        
        Args:
            similarity_threshold: 相似度阈值 (默认 0.75)
            max_results: 最大结果数量
        """
        self.similarity_threshold = similarity_threshold or self.SIMILARITY_THRESHOLD
        self.max_results = max_results
    
    def fuse(
        self,
        retrieval_results: Dict[str, List]
    ) -> Tuple[List[FusedResult], FusionStats]:
        """
        融合多个检索来源的结果
        
        Args:
            retrieval_results: 各来源的检索结果
            
        Returns:
            Tuple[List[FusedResult], FusionStats]: 融合后的结果和统计信息
        """
        stats = FusionStats()
        all_results = []
        
        for source, results in retrieval_results.items():
            stats.total_sources += len(results)
            
            if source == "knowledge_base":
                stats.knowledge_base_count = len(results)
            elif source == "pubmed":
                stats.pubmed_count = len(results)
            elif source == "model":
                stats.model_count = len(results)
            
            for r in results:
                all_results.append(self._convert_to_fused(r, source))
        
        all_results = self._deduplicate(all_results)
        stats.duplicates_removed = stats.total_sources - len(all_results)
        
        all_results = self._prioritize(all_results)
        
        all_results = self._calculate_confidence(all_results)
        
        all_results = all_results[:self.max_results]
        stats.final_count = len(all_results)
        
        logger.info(f"结果融合完成: {stats.total_sources} -> {stats.final_count} 条")
        
        return all_results, stats
    
    def _convert_to_fused(self, result, source: str) -> FusedResult:
        """将检索结果转换为融合结果"""
        content = ""
        if hasattr(result, 'content'):
            content = str(result.content)
        else:
            content = str(result.get('content', str(result)))
        
        relevance = 0.5
        if hasattr(result, 'relevance_score'):
            relevance = float(result.relevance_score) if result.relevance_score else 0.5
        elif isinstance(result, dict) and 'relevance_score' in result:
            relevance = float(result.get('relevance_score', 0.5)) if result.get('relevance_score') else 0.5
        
        return FusedResult(
            content=content,
            sources=[source],
            relevance_score=relevance,
            source_priority=self.SOURCE_PRIORITIES.get(source, 0)
        )
    
    def _deduplicate(self, results: List[FusedResult]) -> List[FusedResult]:
        """去重 - 移除相似内容"""
        if not results:
            return results
        
        unique_results = []
        
        for current in results:
            is_duplicate = False
            
            for existing in unique_results:
                similarity = self._calculate_similarity(
                    current.content, 
                    existing.content
                )
                
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    if len(current.sources) > len(existing.sources):
                        existing.sources.extend(current.sources)
                    break
            
            if not is_duplicate:
                unique_results.append(current)
        
        return unique_results
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0
        
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        import re
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _prioritize(self, results: List[FusedResult]) -> List[FusedResult]:
        """按优先级排序"""
        return sorted(
            results,
            key=lambda x: (
                x.source_priority,
                x.relevance_score
            ),
            reverse=True
        )
    
    def _calculate_confidence(self, results: List[FusedResult]) -> List[FusedResult]:
        """计算置信度"""
        for r in results:
            if not r.sources:
                r.confidence = 0.0
                continue
            
            source_weights = [self.SOURCE_WEIGHTS.get(s, 0.5) for s in r.sources]
            avg_weight = sum(source_weights) / len(source_weights)
            
            r.confidence = avg_weight * r.relevance_score
        
        return results
    
    def build_fused_context(
        self,
        fused_results: List[FusedResult],
        query: str,
        include_confidence: bool = True
    ) -> str:
        """
        构建融合后的上下文
        
        Args:
            fused_results: 融合后的结果
            query: 查询文本
            include_confidence: 是否包含置信度
            
        Returns:
            str: 构建的上下文文本
        """
        if not fused_results:
            return ""
        
        context_parts = []
        
        context_parts.append(f"Query: {query}\n")
        context_parts.append("Relevant Information:\n")
        
        for i, result in enumerate(fused_results, 1):
            source_info = ", ".join(result.sources)
            
            if include_confidence:
                conf_percent = int(result.confidence * 100)
                context_parts.append(f"[Source: {source_info} | Confidence: {conf_percent}%] ")
            else:
                context_parts.append(f"[Source: {source_info}] ")
            
            content = result.content.strip()
            if content:
                context_parts.append(f"{content}\n")
                context_parts.append("-" * 40 + "\n")
        
        return "".join(context_parts)
    
    def extract_key_findings(
        self,
        fused_results: List[FusedResult],
        max_findings: int = 5
    ) -> List[Dict[str, Any]]:
        """
        提取关键发现
        
        Args:
            fused_results: 融合后的结果
            max_findings: 最大发现数量
            
        Returns:
            List[Dict]: 关键发现列表
        """
        findings = []
        
        for result in fused_results[:max_findings]:
            finding = {
                "content": result.content,
                "sources": result.sources,
                "confidence": result.confidence,
                "priority": result.source_priority
            }
            findings.append(finding)
        
        return findings
