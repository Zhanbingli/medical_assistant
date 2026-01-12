"""
测试search.py模块
"""
import pytest
from src.rag.search import QueryExpander, Reranker, MedicalSearchEngine
from src.rag.database import MedicalKnowledgeDB
from src.utils.exceptions import SearchError

class TestQueryExpander:
    """测试查询扩展器"""
    
    def test_expand_success(self, mock_ollama_chat):
        """测试查询扩展成功"""
        expander = QueryExpander(llm_model="qwen2.5:7b")
        result = expander.expand("发热症状")
        assert len(result) > 0
        assert "发热症状" in result
    
    def test_expand_empty_query(self):
        """测试空查询"""
        expander = QueryExpander(llm_model="qwen2.5:7b")
        result = expander.expand("")
        assert result == []
    
    def test_expand_api_failure(self, mock_ollama_chat):
        """测试API失败"""
        mock_ollama_chat.side_effect = Exception("API Error")
        expander = QueryExpander(llm_model="qwen2.5:7b")
        result = expander.expand("发热")
        assert result == ["发热"]

class TestReranker:
    """测试重排序器"""
    
    def test_rerank_empty(self):
        """测试空文档列表"""
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        result = reranker.rerank("发热", [], [])
        assert result == []
    
    def test_rerank_batching(self):
        """测试批处理功能"""
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        # 创建大量文档测试批处理
        docs = [f"Document {i} about fever" for i in range(100)]
        metas = [{"source": f"doc_{i}.pdf"} for i in range(100)]
        
        # 应该成功处理，不会OOM
        result = reranker.rerank("发热", docs, metas)
        assert len(result) == 100
        # 按分数降序排列
        scores = [score for _, score, _ in result]
        assert scores == sorted(scores, reverse=True)

class TestMedicalSearchEngine:
    """测试医学搜索引擎"""
    
    def test_search_empty_query(self, temp_db_dir):
        """测试空查询"""
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        expander = QueryExpander(llm_model="qwen2.5:7b")
        engine = MedicalSearchEngine(db, reranker, expander)
        
        result, logs = engine.search("")
        assert "查询不能为空" in result
        assert len(logs) == 0
    
    def test_search_api_failure(self, mock_ollama_chat, temp_db_dir):
        """测试搜索API失败"""
        mock_ollama_chat.side_effect = Exception("API Error")
        
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        expander = QueryExpander(llm_model="qwen2.5:7b")
        engine = MedicalSearchEngine(db, reranker, expander)
        
        # 不应该抛出异常
        result, logs = engine.search("发热")
        assert isinstance(result, str)
        assert isinstance(logs, list)
    
    def test_search_no_results(self, temp_db_dir):
        """测试无结果情况"""
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        expander = QueryExpander(llm_model="qwen2.5:7b")
        engine = MedicalSearchEngine(db, reranker, expander)
        
        result, logs = engine.search("不存在的症状xyz123")
        assert "未找到相关资料" in result