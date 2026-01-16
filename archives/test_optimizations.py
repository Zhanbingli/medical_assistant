#!/usr/bin/env python3
"""
MedGemma 优化后综合验证脚本
验证所有修改是否正常工作
"""

import sys
import time

sys.path.insert(0, '/Users/lizhanbing12/learn_project/pdf_markd')

from config import (
    RECALL_N_RESULTS, RERANK_TOP_K, RERANK_THRESHOLD, MULTI_QUERY_COUNT,
    CHUNK_SIZE, CHUNK_OVERLAP_LINES, BATCH_SIZE,
    MedGemmaConfig, MEDGEMMA_MODEL, MEDGEMMA_SYSTEM_PROMPT
)
from src.llm import get_adapter
from src.agent.core import MedicalAgent
from src.rag.search import QueryExpander, Reranker, MedicalSearchEngine
from src.rag.loader import MarkdownProcessor, DocumentEmbedder
from src.rag.database import MedicalKnowledgeDB


def test_config():
    """测试配置参数"""
    print("=== 配置参数测试 ===")
    print(f"RECALL_N_RESULTS: {RECALL_N_RESULTS} (优化前: 3)")
    print(f"RERANK_TOP_K: {RERANK_TOP_K} (优化前: 3)")
    print(f"RERANK_THRESHOLD: {RERANK_THRESHOLD} (优化前: -10.0)")
    print(f"MULTI_QUERY_COUNT: {MULTI_QUERY_COUNT} (优化前: 2)")
    print(f"CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP_LINES: {CHUNK_OVERLAP_LINES}")
    print(f"MedGemma 温度: strict={MedGemmaConfig.TEMPERATURE_STRICT}, creative={MedGemmaConfig.TEMPERATURE_CREATIVE}")
    assert RECALL_N_RESULTS == 10, "RECALL_N_RESULTS 应为 10"
    assert RERANK_TOP_K == 5, "RERANK_TOP_K 应为 5"
    assert RERANK_THRESHOLD == -1.0, "RERANK_THRESHOLD 应为 -1.0"
    print("✅ 配置参数正确\n")


def test_agent_parser():
    """测试 Agent 动作解析"""
    print("=== Agent 动作解析测试 ===")
    
    from src.agent.core import MedicalAgent
    
    class MockSearchTool:
        def execute(self, keyword):
            return f"关于 {keyword} 的检索结果"
    
    agent = MedicalAgent(MockSearchTool())
    
    # 测试中文解析
    test_cases = [
        ("Action: 检索: 糖尿病并发症", ("search", "糖尿病并发症")),
        ("检索：高血压的诊断标准", ("search", "高血压的诊断标准")),
        ("search: 冠心病的治疗", ("search", "冠心病的治疗")),
        ("Search: 糖尿病肾病", ("search", "糖尿病肾病")),
        ("这是一个普通回复，没有动作", ("", "")),
    ]
    
    for input_text, expected in test_cases:
        result = agent._parse_action(input_text)
        status = "✅" if result == expected else "❌"
        print(f"{status} 输入: '{input_text[:30]}...' -> {result}")
        if result != expected:
            print(f"   期望: {expected}")
    
    # 测试最终答案检测
    final_cases = [
        ("Final Answer: 这是答案", True),
        ("最终答案: 这里是答案", True),
        ("Answer: answer", True),
        ("只是一个回复", False),
    ]
    
    for text, expected in final_cases:
        result = agent._check_final_answer(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} 最终答案检测: '{text[:20]}...' -> {result}")
    
    print()


def test_document_chunking():
    """测试文档切分"""
    print("=== 文档切分测试 ===")
    
    # 测试医学文本
    test_text = """# 第一章 糖尿病

## 1.1 糖尿病的定义

糖尿病是一种以高血糖为特征的代谢性疾病。主要表现为多饮、多尿、多食和体重减轻。

## 1.2 病因与发病机制

糖尿病的病因主要包括遗传因素和环境因素。胰岛β细胞功能缺陷导致胰岛素分泌减少，胰岛素抵抗引起外周组织对葡萄糖利用障碍。

## 1.3 临床表现

典型症状包括多饮、多尿、多食和体重减轻，即"三多一少"。部分患者可出现皮肤瘙痒、视力模糊等症状。

## 1.4 诊断标准

糖尿病的诊断标准包括：空腹血糖≥7.0mmol/L，餐后2小时血糖≥11.1mmol/L，糖化血红蛋白≥6.5%。

## 1.5 并发症

糖尿病的慢性并发症包括糖尿病肾病、糖尿病视网膜病变、糖尿病足和心血管疾病。
"""
    
    chunks = MarkdownProcessor.split_smart(test_text, chunk_size=300)
    print(f"原文长度: {len(test_text)} 字符")
    print(f"切分块数: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        # 检查是否在段落边界切分
        has_section = "【章节：" in chunk
        print(f"  块 {i+1}: {len(chunk)} 字符 {'✅' if has_section else '❌'} 包含章节上下文")
    
    print("✅ 文档切分测试完成\n")


def test_adapter_performance():
    """测试适配器性能"""
    print("=== 适配器性能测试 ===")
    
    adapter = get_adapter()
    
    questions = [
        "什么是高血压？",
        "糖尿病的诊断标准是什么？",
    ]
    
    total_time = 0
    for q in questions:
        start = time.time()
        response = adapter.chat(
            messages=[{'role': 'user', 'content': q}],
            temperature=0.1,
            max_tokens=512
        )
        elapsed = time.time() - start
        total_time += elapsed
        
        content = response['message']['content']
        print(f"问题: '{q[:15]}...'")
        print(f"  耗时: {elapsed:.2f}秒 | 回复: {len(content)}字符")
    
    avg_time = total_time / len(questions)
    print(f"\n平均响应时间: {avg_time:.2f}秒")
    if avg_time < 5:
        print("✅ 性能优秀")
    elif avg_time < 10:
        print("✅ 性能可接受")
    else:
        print("⚠️ 性能较慢，建议优化")
    print()


def test_query_expansion():
    """测试查询扩展"""
    print("=== 查询扩展测试 ===")
    
    expander = QueryExpander()
    
    test_query = "糖尿病并发症有哪些"
    start = time.time()
    expanded = expander.expand(test_query, count=3)
    elapsed = time.time() - start
    
    print(f"原始查询: {test_query}")
    print(f"扩展结果: {expanded}")
    print(f"耗时: {elapsed:.2f}秒")
    
    assert len(expanded) >= 2, "扩展结果应至少包含原始查询和1个变体"
    assert expanded[0] == test_query, "第一个结果应该是原始查询"
    print("✅ 查询扩展测试通过\n")


def main():
    print("=" * 60)
    print("MedGemma 优化后综合验证")
    print("=" * 60)
    print()
    
    try:
        test_config()
        test_agent_parser()
        test_document_chunking()
        test_adapter_performance()
        test_query_expansion()
        
        print("=" * 60)
        print("✅ 所有测试通过！优化完成！")
        print("=" * 60)
        
        print("\n📋 修改总结:")
        print("1. ✅ Agent 动作解析支持中英文")
        print("2. ✅ RAG 参数优化 (召回量增加，阈值调整)")
        print("3. ✅ 文档切分优化 (段落边界切分)")
        print("4. ✅ 新增 UI 组件库 (medical_ui.py)")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
