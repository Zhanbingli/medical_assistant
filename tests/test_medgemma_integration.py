"""
MedGemma 集成测试
重点验证医学准确性
"""

import pytest
import time
from src.llm import get_adapter, MedGemmaAdapter
from config import MEDGEMMA_SYSTEM_PROMPT


class TestMedGemmaIntegration:
    """MedGemma 集成测试类"""
    
    @pytest.fixture
    def adapter(self):
        """获取适配器实例"""
        return get_adapter()
    
    def test_basic_chat(self, adapter):
        """基础对话测试"""
        response = adapter.chat(
            messages=[{'role': 'user', 'content': '你好，请自我介绍'}]
        )
        assert response['message']['content'] is not None
        assert len(response['message']['content']) > 10
    
    def test_medical_qa_accuracy(self, adapter):
        """医学问答准确性测试"""
        test_cases = [
            {
                'question': '什么是高血压的诊断标准？',
                'keywords': ['血压', '140', '90', 'mmHg']
            },
            {
                'question': '糖尿病的典型症状有哪些？',
                'keywords': ['多饮', '多尿', '多食', '体重下降']
            },
            {
                'question': '冠心病的危险因素有哪些？',
                'keywords': ['高血压', '糖尿病', '吸烟', '血脂']
            }
        ]
        
        for case in test_cases:
            response = adapter.chat(
                messages=[{'role': 'user', 'content': case['question']}]
            )
            content = response['message']['content']
            
            # 检查是否包含关键医学术语
            found_keywords = [kw for kw in case['keywords'] if kw in content]
            assert len(found_keywords) >= 2, f"回答中应包含至少2个关键词, 找到: {found_keywords}"
    
    def test_reasoning_chain(self, adapter):
        """推理链测试"""
        messages = [
            {'role': 'system', 'content': MEDGEMMA_SYSTEM_PROMPT},
            {'role': 'user', 'content': '阿司匹林的主要适应症和禁忌症是什么？'}
        ]
        
        response = adapter.chat(
            messages=messages,
            temperature=0.1
        )
        
        assert response['message']['content'] is not None
        content = response['message']['content']
        # 回答应该包含适应症和禁忌症相关内容
        assert len(content) > 50
    
    def test_response_time(self, adapter):
        """响应时间测试 (M2 性能基准)"""
        questions = [
            '什么是高血压？',
            '糖尿病的诊断标准是什么？',
            '简述冠心病的治疗原则'
        ]
        
        times = []
        for q in questions:
            start = time.time()
            adapter.chat(messages=[{'role': 'user', 'content': q}])
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        
        # M2 预期: 平均响应时间 < 5秒
        assert avg_time < 5.0, f"平均响应时间 {avg_time:.2f}秒 超过预期"
        print(f"平均响应时间: {avg_time:.2f}秒")


class TestMedGemmaEdgeCases:
    """边缘情况测试"""
    
    def test_empty_message(self):
        """空消息处理"""
        adapter = get_adapter()
        with pytest.raises(Exception):
            adapter.chat(messages=[])
    
    def test_very_long_query(self):
        """超长查询测试"""
        adapter = get_adapter()
        long_query = "高血压" * 1000
        
        response = adapter.chat(
            messages=[{'role': 'user', 'content': long_query}]
        )
        assert response is not None


class TestMedGemmaConfig:
    """配置测试"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        config = MedGemmaConfig()
        
        assert config.model_name == "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
        assert config.temperature_strict == 0.1
        assert config.temperature_creative == 0.5
        assert config.max_tokens == 2048
        assert len(config.stop_tokens) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
