"""
MedGemma 1.5 4B 专用适配器
基于 Gemma 3 架构，针对医学场景优化
"""

import ollama
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from config import MEDGEMMA_MODEL, MedGemmaConfig as Config


@dataclass
class MedGemmaConfig:
    """MedGemma 配置参数"""
    model_name: str = MEDGEMMA_MODEL
    temperature_strict: float = Config.TEMPERATURE_STRICT
    temperature_creative: float = Config.TEMPERATURE_CREATIVE
    max_tokens: int = Config.MAX_TOKENS
    top_k: int = Config.TOP_K
    top_p: float = Config.TOP_P
    context_length: int = Config.CONTEXT_LENGTH
    stop_tokens: List[str] = None
    
    def __post_init__(self):
        if self.stop_tokens is None:
            self.stop_tokens = Config.STOP_TOKENS


class MedGemmaAdapter:
    """
    MedGemma 1.5 4B 统一适配器
    
    特点:
    - 自动处理 Gemma 3 chat template
    - 针对医学场景优化参数
    - 支持流式输出
    - 单用户场景优化
    """
    
    def __init__(self, config: Optional[MedGemmaConfig] = None):
        self.config = config or MedGemmaConfig()
        self._warmup_done = False
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        统一的聊天接口
        
        Args:
            messages: 消息列表 [{'role': 'user', 'content': '...'}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            stream: 是否流式输出
        
        Returns:
            Ollama 响应字典
        """
        if not self._warmup_done:
            self._warmup()
        
        temp = temperature if temperature is not None else self.config.temperature_strict
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        response = ollama.chat(
            model=self.config.model_name,
            messages=messages,
            options={
                'temperature': temp,
                'num_predict': tokens,
                'top_k': self.config.top_k,
                'top_p': self.config.top_p,
                'stop': self.config.stop_tokens,
                'num_ctx': self.config.context_length,
            },
            stream=stream
        )
        
        return response
    
    def stream_chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        """
        流式聊天接口
        
        Yields:
            生成的文本片段
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        for chunk in response:
            if chunk.get('message', {}).get('content'):
                yield chunk['message']['content']
    
    def _warmup(self):
        """预热模型，减少冷启动延迟"""
        try:
            ollama.chat(
                model=self.config.model_name,
                messages=[{'role': 'user', 'content': 'Hi'}],
                options={'temperature': 0.1, 'num_predict': 10}
            )
            self._warmup_done = True
        except Exception as e:
            print(f"Warning: Model warmup failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            info = ollama.show(self.config.model_name)
            return {
                'name': self.config.model_name,
                'size': info.get('size', 'unknown'),
                'parameters': info.get('parameters', 'unknown'),
                'template': info.get('template', 'unknown'),
            }
        except Exception as e:
            return {'error': str(e)}


_medgemma_adapter: Optional[MedGemmaAdapter] = None


def get_adapter() -> MedGemmaAdapter:
    """获取全局适配器实例 (单例模式)"""
    global _medgemma_adapter
    if _medgemma_adapter is None:
        _medgemma_adapter = MedGemmaAdapter()
    return _medgemma_adapter
