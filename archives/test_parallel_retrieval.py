#!/usr/bin/env python
"""
测试并行检索架构
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, RECALL_N_RESULTS
from src.rag.database import MedicalKnowledgeDB
from src.utils.web_search import PubMedSearch
from src.retrieval import ParallelRetriever, ResultFuser


def test_parallel_retriever():
    """测试并行检索器"""
    print("=" * 60)
    print("测试并行检索架构")
    print("=" * 60)
    
    db = MedicalKnowledgeDB(DB_PATH, COLLECTION_NAME)
    pubmed_search = PubMedSearch()
    
    retriever = ParallelRetriever(
        db=db,
        pubmed_search=pubmed_search,
        embed_model=EMBEDDING_MODEL,
        recall_count=RECALL_N_RESULTS
    )
    
    fuser = ResultFuser(
        similarity_threshold=0.75,
        max_results=5
    )
    
    test_queries = [
        "什么是高血压？",
        "糖尿病的并发症有哪些？",
        "冠心病的诊断标准"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"测试查询: {query}")
        print(f"{'='*60}")
        
        print("1. 并行检索...")
        retrieval_results = retriever.retrieve_all(query, [])
        
        print(f"   - 知识库: {len(retrieval_results['knowledge_base'])} 条")
        print(f"   - PubMed: {len(retrieval_results['pubmed'])} 条")
        print(f"   - 模型知识: {len(retrieval_results['model'])} 条")
        
        print("\n2. 融合结果...")
        fused_results, stats = fuser.fuse(retrieval_results)
        
        print(f"   - 融合后: {len(fused_results)} 条")
        print(f"   - 去重: {stats.duplicates_removed} 条")
        
        print("\n3. 构建上下文...")
        context = fuser.build_fused_context(fused_results, query)
        print(f"   - 上下文长度: {len(context)} 字符")
        
        print("\n4. 置信度分布:")
        for i, result in enumerate(fused_results[:3], 1):
            conf_percent = int(result.confidence * 100)
            sources = ", ".join(result.sources)
            print(f"   [{i}] 置信度: {conf_percent}% | 来源: {sources}")
            print(f"       内容: {result.content[:80]}...")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_parallel_retriever()
