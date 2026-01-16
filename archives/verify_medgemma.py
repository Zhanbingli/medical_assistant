#!/usr/bin/env python3
"""
MedGemma 1.5 4B 快速验证脚本
用于验证模型替换是否成功
"""

import sys
import time

# 添加项目路径
sys.path.insert(0, '/Users/lizhanbing12/learn_project/pdf_markd')

from src.llm import get_adapter
from config import MEDGEMMA_MODEL, MEDGEMMA_SYSTEM_PROMPT, MedGemmaConfig


def test_adapter():
    """测试适配器初始化"""
    print("1. 测试适配器初始化...")
    adapter = get_adapter()
    print(f"   ✅ 适配器初始化成功")
    print(f"   模型: {MEDGEMMA_MODEL}")
    return adapter


def test_basic_chat(adapter):
    """测试基础对话"""
    print("\n2. 测试基础对话...")
    response = adapter.chat(
        messages=[{'role': 'user', 'content': '你好'}],
        temperature=0.1
    )
    content = response['message']['content']
    assert len(content) > 10, "回复太短"
    print(f"   ✅ 基础对话正常 (回复长度: {len(content)}字符)")


def test_medical_qa(adapter):
    """测试医学问答"""
    print("\n3. 测试医学问答...")
    response = adapter.chat(
        messages=[
            {'role': 'system', 'content': MEDGEMMA_SYSTEM_PROMPT},
            {'role': 'user', 'content': '什么是高血压的诊断标准？'}
        ],
        temperature=0.1
    )
    content = response['message']['content']
    
    # 检查医学关键词
    keywords = ['mmHg', '血压', '140', '90']
    found = [kw for kw in keywords if kw in content]
    
    print(f"   回复长度: {len(content)}字符")
    print(f"   包含关键词: {found}")
    assert len(found) >= 2, f"应包含至少2个医学关键词"
    print(f"   ✅ 医学问答正常")


def test_performance(adapter):
    """测试响应时间"""
    print("\n4. 测试响应时间...")
    
    questions = [
        '什么是高血压？',
        '糖尿病的诊断标准是什么？'
    ]
    
    times = []
    for q in questions:
        start = time.time()
        adapter.chat(messages=[{'role': 'user', 'content': q}])
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   问题: {q[:15]}... 耗时: {elapsed:.2f}秒")
    
    avg_time = sum(times) / len(times)
    print(f"\n   平均响应时间: {avg_time:.2f}秒")
    
    if avg_time < 5:
        print("   ✅ 性能优秀")
    elif avg_time < 10:
        print("   ✅ 性能可接受")
    else:
        print("   ⚠️ 性能较慢 (M2 预期)")


def main():
    print("=" * 60)
    print("MedGemma 1.5 4B 验证脚本")
    print("=" * 60)
    print(f"模型: {MEDGEMMA_MODEL}")
    print(f"温度: 严格={MedGemmaConfig.TEMPERATURE_STRICT}, 创意={MedGemmaConfig.TEMPERATURE_CREATIVE}")
    print("=" * 60)
    
    try:
        adapter = test_adapter()
        test_basic_chat(adapter)
        test_medical_qa(adapter)
        test_performance(adapter)
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过! MedGemma 替换成功!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
