"""
医疗安全增强模块
提供置信度评估、知识验证和责任追溯功能
"""
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class EvidenceLevel(Enum):
    """循证医学证据等级"""
    A = "A级：系统评价/Meta分析"
    B = "B级：RCT随机对照试验"
    C = "C级：观察性研究/专家意见"
    D = "D级：经验性建议"
    E = "E级：无证据支持"

class MedicalSafetyChecker:
    """医疗安全检查器"""
    
    def __init__(self):
        """初始化安全检查器"""
        # 高风险关键词（需要警告）
        self.high_risk_keywords = [
            "死亡", "致命", "猝死", "危及生命",
            "肿瘤", "癌症", "白血病",
            "手术", "开颅", "开胸",
            "急救", "抢救", "ICU"
        ]
        
        # 需要立即就医的症状
        self.emergency_symptoms = [
            "胸痛", "呼吸困难", "意识模糊", "高烧不退",
            "严重头痛", "肢体无力", "大量出血"
        ]
    
    def check_answer_safety(
        self, 
        answer: str, 
        query: str,
        retrieved_sources: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        检查回答的安全性
        
        Args:
            answer: AI生成的回答
            query: 用户查询
            retrieved_sources: 检索到的源文本
            
        Returns:
            (是否安全, 警告列表)
        """
        warnings = []
        
        # 1. 检查高风险内容
        for keyword in self.high_risk_keywords:
            if keyword in answer:
                warnings.append(f"⚠️ 回答涉及高风险关键词：{keyword}，请谨慎对待")
        
        # 2. 检查急救症状
        for symptom in self.emergency_symptoms:
            if symptom in query and "立即就医" not in answer:
                warnings.append(f"🚨 用户描述{symptom}，回答未包含'立即就医'建议")
        
        # 3. 检查是否有"不确定"标记
        if "不确定" in query and "建议" not in answer:
            warnings.append("⚠️ 用户表示不确定，但回答未明确建议咨询医生")
        
        # 4. 检查检索结果质量
        if len(retrieved_sources) == 0:
            warnings.append("🔍 检索结果为空，回答可能不可靠")
        elif len(retrieved_sources) < 3:
            warnings.append("⚠️ 检索结果少于3条，证据有限")
        
        is_safe = len(warnings) == 0
        return is_safe, warnings

class ConfidenceScorer:
    """置信度评分器"""
    
    def __init__(self):
        """初始化评分器"""
        pass
    
    def calculate_confidence(
        self,
        retrieved_sources: List[str],
        rerank_scores: List[float],
        answer_length: int,
        answer_structure: str
    ) -> Dict[str, Any]:
        """
        计算回答的置信度

        Args:
            retrieved_sources: 检索到的源文本
            rerank_scores: Rerank打分
            answer_length: 回答长度
            answer_structure: 回答结构类型
            
        Returns:
            置信度详情字典
        """
        # 1. 基于Rerank分数的置信度
        if rerank_scores:
            avg_score = sum(rerank_scores) / len(rerank_scores)
            max_score = max(rerank_scores)
            
            # 归一化到0-100
            score_confidence = (avg_score + 10) / 10 * 50  # -10到0的范围
            score_confidence = min(100, max(0, score_confidence))
        else:
            score_confidence = 0
            avg_score = 0
            max_score = 0
        
        # 2. 基于检索数量的置信度
        source_count = len(retrieved_sources)
        count_confidence = min(100, source_count * 20)  # 每条加20分
        
        # 3. 基于回答完整性的置信度
        if answer_structure == "诊断":
            # 诊断结构应该有较高的完整性评分
            structure_score = 80.0
        else:
            structure_score = 70.0  # 其他结构默认分
        
        # 4. 综合置信度
        total_confidence = (
            score_confidence * 0.5 +    # Rerank分数权重最高
            count_confidence * 0.35 +    # 检索数量
            structure_score * 0.15       # 回答完整性
        )
        
        # 5. 证据等级: 0-40=D, 40-70=C, 70-85=B, 85-100=A
        if total_confidence >= 85:
            evidence_level = EvidenceLevel.A  # 优秀证据
        elif total_confidence >= 70:
            evidence_level = EvidenceLevel.B  # 良好证据
        elif total_confidence >= 60:
            evidence_level = EvidenceLevel.C  # 中等证据
        else:
            evidence_level = EvidenceLevel.D  # 证据有限
        
        return {
            "total_confidence": round(total_confidence, 1),
            "score_confidence": round(score_confidence, 1),
            "count_confidence": round(count_confidence, 1),
            "structure_confidence": round(structure_score, 1),
            "avg_rerank_score": round(avg_score, 3),
            "max_rerank_score": round(max_score, 3),
            "source_count": source_count,
            "evidence_level": evidence_level,
            "confidence_color": self._get_confidence_color(total_confidence)
        }
    
    def _get_confidence_color(self, confidence: float) -> str:
        """根据置信度返回颜色"""
        if confidence >= 80:
            return "#4caf50"  # 绿色
        elif confidence >= 60:
            return "#ff9800"  # 橙色
        else:
            return "#f44336"  # 红色

class SourceAttributor:
    """来源追溯器"""
    
    def __init__(self):
        """初始化追溯器"""
        pass
    
    def extract_sources(
        self,
        answer: str,
        retrieved_docs: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        从回答中提取引用来源
        
        Args:
            answer: AI回答
            retrieved_docs: 检索到的文档
            metadatas: 文档元数据
            
        Returns:
            来源列表
        """
        sources = []
        
        # 提取文件名
        files_used = set()
        for meta in metadatas:
            if meta and 'source' in meta:
                files_used.add(meta['source'])
        
        # 提取可能的章节（如果有【章节：标记）
        chapters = re.findall(r'【章节：(.*?)】', answer)
        
        for i, (doc, meta) in enumerate(zip(retrieved_docs, metadatas)):
            if not doc:
                continue
                
            source_info = {
                "index": i + 1,
                "filename": meta.get('source', '未知来源') if meta else '未知来源',
                "chapter": meta.get('chapter', '未知章节') if meta else self._extract_chapter(doc),
                "preview": doc[:100] + "..." if len(doc) > 100 else doc,
                "chunk_length": len(doc),
                "relevance": self._calculate_relevance(answer, doc)
            }
            sources.append(source_info)
        
        # 按相关性排序
        sources.sort(key=lambda x: x['relevance'], reverse=True)
        
        return sources[:5]  # 返回最相关的5个
    
    def _extract_chapter(self, doc: str) -> str:
        """从文档中提取章节信息"""
        match = re.search(r'【章节：(.*?)】', doc)
        return match.group(1) if match else "未知章节"
    
    def _calculate_relevance(self, answer: str, doc: str) -> float:
        """计算回答与文档的相关性（简化版）"""
        # 实际项目中应该用更复杂的算法
        answer_words = set(answer.split())
        doc_words = set(doc.split())
        
        intersection = answer_words & doc_words
        union = answer_words | doc_words
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union) * 100

class SafetyEnhancer:
    """安全增强器 - 主入口"""
    
    def __init__(self):
        """初始化安全增强器"""
        self.safety_checker = MedicalSafetyChecker()
        self.confidence_scorer = ConfidenceScorer()
        self.source_attributor = SourceAttributor()
    
    def enhance_answer(
        self,
        answer: str,
        query: str,
        retrieved_docs: List[str],
        metadatas: List[Dict[str, Any]],
        rerank_scores: List[float]
    ) -> Dict[str, Any]:
        """
        增强回答的安全性和可信度
        
        Args:
            answer: 原始AI回答
            query: 用户查询
            retrieved_docs: 检索到的文档
            metadatas: 文档元数据
            rerank_scores: Rerank打分
            
        Returns:
            增强后的回答信息
        """
        # 1. 安全检查
        is_safe, warnings = self.safety_checker.check_answer_safety(
            answer, query, retrieved_docs
        )
        
        # 2. 置信度评分
        # 判断回答结构
        if "诊断" in query or ("可能" in answer and "鉴别" in answer):
            answer_structure = "诊断"
        elif "治疗" in query or "怎么治" in query:
            answer_structure = "治疗"
        elif "检查" in query or "化验" in query:
            answer_structure = "检查"
        else:
            answer_structure = "通用"
        
        confidence = self.confidence_scorer.calculate_confidence(
            retrieved_docs,
            rerank_scores,
            len(answer),
            answer_structure
        )
        
        # 3. 来源追溯
        sources = self.source_attributor.extract_sources(
            answer, retrieved_docs, metadatas
        )
        
        # 4. 构建增强信息
        enhanced_answer = {
            "original_answer": answer,
            "is_safe": is_safe,
            "warnings": warnings,
            "confidence": confidence,
            "sources": sources,
            "enhanced_answer": self._build_enhanced_answer(
                answer, warnings, confidence, sources
            ),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "source_count": len(retrieved_docs),
                "rerank_scores": rerank_scores
            }
        }
        
        return enhanced_answer
    
    def _build_enhanced_answer(
        self,
        answer: str,
        warnings: List[str],
        confidence: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> str:
        """构建增强后的回答"""
        parts = []
        
        # 1. 置信度标签
        color = confidence['confidence_color']
        parts.append(f"""
<div style='background: {color}; color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <strong>📊 置信度: {confidence['total_confidence']}% | 证据等级: {confidence['evidence_level'].value}</strong>
    <div style='margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;'>
        • Rerank分数: {confidence['avg_rerank_score']} | 检索数量: {confidence['source_count']}条
    </div>
</div>
""")
        
        # 2. 警告信息
        if warnings:
            parts.append("""
<div style='background: rgba(255, 152, 0, 0.1); border-left: 4px solid #ff9800; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <strong>⚠️ 安全提示:</strong>
    <ul style='margin: 0.5rem 0;'>
""")
            for warning in warnings:
                parts.append(f"        <li>{warning}</li>")
            parts.append("""
    </ul>
</div>
""")
        
        # 3. 原始回答
        parts.append(f"<div style='margin-bottom: 1.5rem;'>{answer}</div>")
        
        # 4. 来源列表
        if sources:
            parts.append("""
<div style='background: rgba(102, 126, 234, 0.05); padding: 1.5rem; border-radius: 10px; border-left: 5px solid #667eea;'>
    <h4 style='margin-top: 0; color: #667eea;'>📚 知识来源</h4>
    <ul style='margin: 0; padding-left: 1.5rem;'>
""")
            for source in sources:
                parts.append(f"""
        <li>
            <strong>{source['filename']}</strong> 
            <span style='color: #666; font-size: 0.9rem;'>({source['chapter']})</span>
            <div style='font-size: 0.85rem; color: #666; margin-top: 0.3rem;'>
                {source['preview']}
            </div>
        </li>
""")
            parts.append("""
    </ul>
</div>
""")
        
        # 5. 免责声明
        parts.append("""
<div style='background: rgba(244, 67, 54, 0.05); border-left: 4px solid #f44336; padding: 1rem; border-radius: 10px; margin-top: 1.5rem;'>
    <strong style='color: #f44336;'>⚠️ 免责声明</strong>
    <div style='margin-top: 0.5rem; color: #666; font-size: 0.9rem;'>
        本回答仅供参考，不能替代专业医疗建议。如有症状，请及时就医或咨询专业医生。
    </div>
</div>
""")
        
        return ''.join(parts)

# 使用示例
if __name__ == "__main__":
    enhancer = SafetyEnhancer()
    
    # 模拟数据
    answer = "根据检索结果，发热伴咳嗽的可能诊断包括：1.急性上呼吸道感染，2.急性支气管炎，3.肺炎。"
    query = "发热伴咳嗽3天，需要注意哪些诊断"
    retrieved_docs = [
        "【章节：第一章 > 第一节】\n发热是指体温升高超过正常范围。"
    ]
    metadatas = [
        {"source": "诊断学_cleaned.md", "chapter": "第一章 > 第一节"}
    ]
    rerank_scores = [-5.2, -6.8, -7.5]
    
    # 增强回答
    enhanced = enhancer.enhance_answer(
        answer, query, retrieved_docs, metadatas, rerank_scores
    )
    
    print("安全性:", enhanced["is_safe"])
    print("警告:", enhanced["warnings"])
    print("置信度:", enhanced["confidence"])
    print("来源:", enhanced["sources"])
