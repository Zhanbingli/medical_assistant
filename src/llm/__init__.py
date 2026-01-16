"""
LLM 适配器层
支持切换不同的本地模型
"""

from .medgemma_adapter import MedGemmaAdapter, get_adapter, MedGemmaConfig

__all__ = ['MedGemmaAdapter', 'get_adapter', 'MedGemmaConfig']
