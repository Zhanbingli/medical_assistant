# AI Medical Knowledge Base Assistant

AI医学知识库助手 - 基于RAG的循证医学问答系统

## 项目简介

这是一个专业的AI医学知识库助手，采用RAG（检索增强生成）架构，以医学教材作为知识来源，实现类似实习医生的医学问答能力。

### 核心特性
- **循证医学**：强制基于知识库检索，确保回答有医学依据
- **中文优化**：针对中文医学术语和教材优化
- **ReAct架构**：采用推理-行动循环，模拟医生诊疗思维
- **多路召回**：查询扩展 + 向量检索 + 重排序，提升检索质量
- **Streamlit界面**：友好的交互式Web界面

## 技术栈

- **后端框架**: Python 3.x
- **Web界面**: Streamlit
- **AI引擎**:
  - Ollama (本地LLM推理)
  - bge-m3 (中文嵌入模型)
  - qwen2.5:7b (推理模型)
  - BAAI/bge-reranker-base (重排序模型)
- **向量数据库**: ChromaDB
- **文档处理**: PyMuPDF4LLM

## 项目结构

```
├── app.py                      # 主应用 (Streamlit)
├── config.py                   # 配置管理
├── convert.py                  # PDF转Markdown工具
├── requirements.txt            # 依赖管理
├── medical_db/                 # ChromaDB向量存储
├── scripts/
│   └── clean_md.py            # Markdown预处理
├── src/                       # 核心模块
│   ├── agent/                 # ReAct Agent
│   │   ├── core.py           # 医学Agent
│   │   └── tools.py          # 搜索工具
│   ├── rag/                  # RAG核心
│   │   ├── database.py       # 数据库接口
│   │   ├── loader.py         # 文档加载器
│   │   └── search.py         # 搜索引擎
│   └── utils/
└── data/                     # 数据文件
    ├── *.pdf                 # 原始PDF教材
    └── *.md                  # 处理后的Markdown
```

## 快速开始

### 环境要求

- Python 3.8+
- Ollama (需提前安装并下载模型)
- 至少8GB RAM

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd pdf_markd
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置Ollama模型
```bash
# 下载所需模型
ollama pull qwen2.5:7b
ollama pull bge-m3
```

4. 启动应用
```bash
streamlit run app.py
```

### 数据准备

1. **PDF转换** (可选)
```bash
python convert.py input.pdf -o output.md
```

2. **上传知识库**
- 通过Web界面上传Markdown格式的医学教材
- 系统会自动分块、向量化并存储

## 配置说明

主要配置在 `config.py` 中：

- **模型配置**: 嵌入模型、LLM模型、重排序模型
- **文档处理**: 分块大小(600字符)、重叠行数(3行)、批大小(20)
- **搜索配置**: 查询扩展数、召回数量、Top-K、阈值
- **Agent配置**: 最大推理步数(5步)、上下文轮数(2轮)

## 核心算法

### 1. 文档处理管道
```
PDF → Markdown → 智能分块(保留章节信息) → 向量化 → 存储
```

### 2. 搜索流程
```
用户查询 → 查询扩展(生成2个关键词) → 多路召回 → 去重 → 重排序 → 阈值过滤 → Top-3结果
```

### 3. ReAct Agent
```
用户问题 → [思考 → 检索 → 观察] × N → 最终答案
强制规则：必须检索后才能回答，禁止"裸答"
```

## 性能优化

- **缓存机制**: 嵌入向量缓存(lru_cache)
- **批处理**: 文档嵌入和数据库写入批量化
- **模型批处理**: Rerank模型支持批处理防止OOM
- **单实例**: Reranker和QueryExpander单实例化
- **流式响应**: 实时显示Agent思考过程

## 开发指南

### 添加新功能

1. **新搜索策略**: 修改 `src/rag/search.py`
2. **新Agent逻辑**: 修改 `src/agent/core.py`
3. **新文档处理器**: 修改 `src/rag/loader.py`

### 调试模式

在侧边栏开启"调试模式"，可查看：
- Rerank详细打分
- 查询扩展结果
- 召回文档详情

## 注意事项

- **医疗免责声明**: 本系统仅供学习和参考，不能替代专业医疗建议
- **数据安全**: 医疗数据请妥善保管，建议本地部署
- **模型要求**: 建议使用中文医学友好的模型（如qwen2.5系列）

## 许可证

MIT

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0 (2026-01-11)
- 初始版本发布
- RAG架构实现
- ReAct Agent
- Streamlit界面
- 中文医学优化
