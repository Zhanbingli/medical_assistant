"""
性能基准测试
针对 M2 芯片优化
"""

import time
from src.llm import get_adapter


def benchmark_medical_qa():
    """医学问答性能基准"""
    adapter = get_adapter()
    
    questions = [
        "什么是高血压？",
        "糖尿病的慢性并发症有哪些？",
        "比较阿司匹林和华法林在抗凝治疗中的优缺点"
    ]
    
    results = []
    
    for i, q in enumerate(questions):
        times = []
        for _ in range(3):
            start = time.time()
            response = adapter.chat(messages=[{'role': 'user', 'content': q}])
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        response_length = len(response['message']['content'])
        
        results.append({
            'question_id': i + 1,
            'question': q[:30] + '...' if len(q) > 30 else q,
            'avg_time': avg_time,
            'response_length': response_length,
            'tokens_per_second': response_length / avg_time if avg_time > 0 else 0
        })
        
        print(f"问题 {i+1}: {avg_time:.2f}秒 | {response_length}字符 | {response_length/avg_time:.1f}字符/秒")
    
    avg_time_total = sum(r['avg_time'] for r in results) / len(results)
    print(f"\n总体平均响应时间: {avg_time_total:.2f}秒")
    
    if avg_time_total < 2.0:
        print("性能优秀 (M2)")
    elif avg_time_total < 4.0:
        print("性能一般 (M2)")
    else:
        print("性能较差，需要优化")


def benchmark_concurrent_requests():
    """并发请求测试"""
    import concurrent.futures
    
    adapter = get_adapter()
    questions = ["什么是高血压？"] * 5
    
    def make_request(q):
        start = time.time()
        adapter.chat(messages=[{'role': 'user', 'content': q}])
        return time.time() - start
    
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        times = list(executor.map(make_request, questions))
    total_time = time.time() - start
    
    print(f"\n并发测试 (5个请求):")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个请求: {total_time/5:.2f}秒")


if __name__ == "__main__":
    print("=" * 60)
    print("MedGemma 1.5 4B 性能基准测试")
    print("硬件: Mac M2")
    print("=" * 60)
    
    print("\n--- 单请求测试 ---")
    benchmark_medical_qa()
    
    print("\n--- 并发测试 ---")
    benchmark_concurrent_requests()
