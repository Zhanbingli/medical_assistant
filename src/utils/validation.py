"""输入验证模块"""
from typing import Any, Optional, List
import re
from src.utils.exceptions import MedicalAIException

def validate_query(query: str, min_length: int = 3, max_length: int = 1000) -> str:
    """
    验证查询文本
    
    Args:
        query: 查询文本
        min_length: 最小长度
        max_length: 最大长度
        
    Returns:
        清洗后的查询文本
        
    Raises:
        MedicalAIException: 验证失败
    """
    if not query or not query.strip():
        raise MedicalAIException("查询不能为空")
    
    query = query.strip()
    
    if len(query) < min_length:
        raise MedicalAIException(f"查询长度太短（最少{min_length}个字符）")
    
    if len(query) > max_length:
        raise MedicalAIException(f"查询长度太长（最多{max_length}个字符）")
    
    # 移除特殊字符
    query = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', query)
    
    return query

def validate_document(document: str, min_length: int = 50) -> str:
    """
    验证文档内容
    
    Args:
        document: 文档内容
        min_length: 最小长度
        
    Returns:
        清洗后的文档内容
        
    Raises:
        MedicalAIException: 验证失败
    """
    if not document or not document.strip():
        raise MedicalAIException("文档内容不能为空")
    
    document = document.strip()
    
    if len(document) < min_length:
        raise MedicalAIException(f"文档内容太短（最少{min_length}个字符）")
    
    # 移除控制字符
    document = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', document)
    
    return document

def validate_embedding(embedding: List[float], expected_dim: int = 1024) -> List[float]:
    """
    验证嵌入向量
    
    Args:
        embedding: 嵌入向量
        expected_dim: 期望的维度
        
    Returns:
        验证后的嵌入向量
        
    Raises:
        MedicalAIException: 验证失败
    """
    if not embedding:
        raise MedicalAIException("嵌入向量为空")
    
    if not isinstance(embedding, list):
        raise MedicalAIException("嵌入向量必须是列表类型")
    
    if len(embedding) != expected_dim:
        raise MedicalAIException(f"嵌入向量维度不匹配（期望{expected_dim}，实际{len(embedding)}）")
    
    # 检查是否包含NaN或Inf
    import math
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise MedicalAIException(f"嵌入向量第{i}个元素不是数值类型")
        if math.isnan(val) or math.isinf(val):
            raise MedicalAIException(f"嵌入向量第{i}个元素为无效值")
    
    return embedding
