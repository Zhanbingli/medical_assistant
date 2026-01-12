"""
测试loader.py模块
"""
import pytest
from src.rag.loader import MarkdownProcessor, DocumentEmbedder
from src.utils.exceptions import DocumentProcessingError, EmbeddingError

class TestMarkdownProcessor:
    """测试Markdown处理器"""
    
    def test_split_smart_basic(self, sample_document):
        """测试基本分块功能"""
        chunks = MarkdownProcessor.split_smart(sample_document, chunk_size=100)
        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)
        assert all('【章节：' in chunk for chunk in chunks)
    
    def test_split_smart_empty(self):
        """测试空文档"""
        chunks = MarkdownProcessor.split_smart("")
        assert len(chunks) == 0
    
    def test_split_smart_whitespace(self):
        """测试空白文档"""
        chunks = MarkdownProcessor.split_smart("   \n  \t  ")
        assert len(chunks) == 0
    
    def test_split_smart_preserves_headers(self, sample_document):
        """测试保留章节标题"""
        chunks = MarkdownProcessor.split_smart(sample_document, chunk_size=50)
        # 至少有一个块包含章节信息
        assert any('第一章' in chunk for chunk in chunks)

class TestDocumentEmbedder:
    """测试文档嵌入器"""
    
    def test_embed_text_success(self, mock_ollama_embeddings):
        """测试嵌入生成成功"""
        embedder = DocumentEmbedder(model_name="bge-m3")
        result = embedder.embed_text("发热症状")
        assert result is not None
        assert len(result) == 1024
        mock_ollama_embeddings.assert_called_once()
    
    def test_embed_text_empty(self):
        """测试嵌入空文本"""
        embedder = DocumentEmbedder(model_name="bge-m3")
        result = embedder.embed_text("")
        assert result is None
    
    def test_embed_text_too_short(self):
        """测试嵌入文本过短"""
        embedder = DocumentEmbedder(model_name="bge-m3")
        result = embedder.embed_text("ab")
        assert result is None
    
    def test_embed_text_api_failure(self, mock_ollama_embeddings):
        """测试API失败"""
        mock_ollama_embeddings.side_effect = Exception("API Error")
        embedder = DocumentEmbedder(model_name="bge-m3")
        result = embedder.embed_text("发热")
        assert result is None
    
    def test_process_file_success(
        self, mock_ollama_embeddings, sample_document, temp_db_dir
    ):
        """测试完整文件处理流程"""
        from src.rag.database import MedicalKnowledgeDB
        
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        embedder = DocumentEmbedder(model_name="bge-m3")
        
        success, info = embedder.process_file(
            sample_document, 
            "test.md", 
            db
        )
        assert success is True
        assert isinstance(info, int)
        assert info > 0
    
    def test_process_file_empty(self, temp_db_dir):
        """测试处理空文件"""
        from src.rag.database import MedicalKnowledgeDB
        
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        embedder = DocumentEmbedder(model_name="bge-m3")
        
        success, info = embedder.process_file("", "empty.md", db)
        assert success is False
        assert info == "EMPTY"
    
    def test_process_file_duplicate(
        self, mock_ollama_embeddings, sample_document, temp_db_dir
    ):
        """测试处理重复文件"""
        from src.rag.database import MedicalKnowledgeDB
        
        db = MedicalKnowledgeDB(temp_db_dir, "test_collection")
        embedder = DocumentEmbedder(model_name="bge-m3")
        
        # 第一次处理
        embedder.process_file(sample_document, "test.md", db)
        
        # 第二次处理（重复）
        success, info = embedder.process_file(sample_document, "test.md", db)
        assert success is False
        assert info == "EXIST"