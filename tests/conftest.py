"""
测试框架配置和共享固件
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def temp_db_dir():
    """创建临时数据库目录"""
    temp_dir = tempfile.mkdtemp(prefix="test_medical_db_")
    yield temp_dir
    # 清理
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture
def mock_ollama_embeddings():
    """Mock Ollama嵌入API"""
    with patch('ollama.embeddings') as mock:
        mock.return_value = {'embedding': [0.1] * 1024}
        yield mock

@pytest.fixture
def mock_ollama_chat():
    """Mock Ollama聊天API"""
    with patch('ollama.chat') as mock:
        mock.return_value = {
            'message': {
                'content': 'Thought: 用户有发热症状\nAction: 检索: 发热'
            }
        }
        yield mock

@pytest.fixture
def sample_document():
    """示例文档"""
    return """
# 第一章 发热

## 第一节 发热的定义

发热是指体温升高超过正常范围。

## 第二节 发热的病因

感染性发热、非感染性发热。
"""

@pytest.fixture
def sample_queries():
    """示例查询"""
    return {
        'simple': '发热',
        'complex': '患者男性，30岁，发热3天，伴有咳嗽和胸痛',
        'empty': '',
        'too_short': 'ab'
    }

@pytest.fixture
def mock_search_engine():
    """Mock搜索引擎"""
    engine = Mock()
    engine.search.return_value = (
        "【章节：第一篇 > 第一章】\n发热的定义和分类\n[来源: 诊断学.pdf]",
        []
    )
    yield engine
