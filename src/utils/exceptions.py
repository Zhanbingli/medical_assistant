"""自定义异常模块"""

class MedicalAIException(Exception):
    """医学AI系统基础异常"""
    pass

class DocumentProcessingError(MedicalAIException):
    """文档处理错误"""
    pass

class EmbeddingError(MedicalAIException):
    """嵌入生成错误"""
    pass

class SearchError(MedicalAIException):
    """搜索错误"""
    pass

class DatabaseError(MedicalAIException):
    """数据库操作错误"""
    pass

class AgentError(MedicalAIException):
    """Agent推理错误"""
    pass

class ConfigurationError(MedicalAIException):
    """配置错误"""
    pass
