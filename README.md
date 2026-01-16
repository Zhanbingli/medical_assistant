# AI Medical Knowledge Base Assistant

AI医学知识库助手 - Evidence-based Medical Q&A System with RAG Architecture

## Overview

A professional AI medical knowledge base assistant using RAG (Retrieval-Augmented Generation) architecture, with medical textbooks as the knowledge source, providing clinical consultation capabilities similar to a resident physician.

### Core Features

- **Evidence-based Medicine**: Knowledge base retrieval as the foundation for all answers
- **Chinese Optimization**: Optimized for Chinese medical terminology and textbooks
- **Parallel Retrieval Architecture**: Concurrent search across knowledge base, PubMed, and model knowledge
- **Confidence Scoring**: Each result includes confidence scores from different sources
- **Streamlit UI**: Interactive WeChat-style chat interface

## Tech Stack

- **Backend**: Python 3.x
- **Web UI**: Streamlit
- **AI Engine**:
  - Ollama (local LLM inference)
  - bge-m3 (Chinese embedding model)
  - medgemma-1.5-4b-it (medical reasoning model)
- **Vector Database**: ChromaDB
- **Document Processing**: PyMuPDF4LLM

## Project Structure

```
├── app_v3.py                  # Main application (Streamlit)
├── config.py                  # Configuration management
├── prepare_rag.py             # RAG preprocessing script
├── 诊断学_rag.md              # Cleaned medical textbook
├── medical_db/                # ChromaDB vector storage
├── src/
│   ├── agent/                 # ReAct Agent
│   │   ├── core.py           # Medical Agent
│   │   └── tools.py          # Search tools
│   ├── rag/                  # RAG Core
│   │   ├── database.py       # Database interface
│   │   ├── loader.py         # Document loader
│   │   └── search.py         # Search engine
│   ├── retrieval/            # Retrieval module (NEW)
│   │   ├── parallel_retriever.py  # Parallel retrieval
│   │   └── result_fuser.py   # Multi-source fusion
│   └── utils/
│       ├── web_search.py     # PubMed search
│       └── safety.py         # Medical safety
├── archives/                  # Archived files
└── tests/                     # Unit tests
```

## Architecture

```
User Input
    ↓
┌─────────────────────────────────────┐
│   Parallel Retrieval (simultaneous) │
│  ┌─────────┐ ┌─────────┐ ┌─────┐  │
│  │Knowledge│ │ PubMed  │ │Model│  │
│  │   Base  │ │ Search  │ │Know.│  │
│  └────┬────┘ └────┬─────┘ └──┬──┘  │
└───────┼───────────┼──────────┼─────┘
        └───────────┼──────────┘
                    ↓
         ┌──────────┴──────────┐
         │  Result Fusion      │
         │  - Deduplication    │
         │  - Priority Sort    │
         │  - Confidence Calc  │
         └──────────┬──────────┘
                    ↓
         ┌──────────┴──────────┐
         │ MedGemma Answer     │
         │ Generation          │
         └────────────────────┘
                    ↓
              Final Answer
```

## Quick Start

### Requirements

- Python 3.8+
- Ollama (with models pre-downloaded)
- At least 8GB RAM

### Installation

```bash
# Clone and enter project
cd pdf_markd

# Install dependencies
pip install -r requirements.txt

# Download required models
ollama pull bge-m3
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M

# Start application
conda activate api_env
streamlit run app_v3.py
```

### Data Preparation

1. **Prepare RAG-ready file** (optional)
```bash
python prepare_rag.py  # Cleans and optimizes markdown files
```

2. **Upload Knowledge Base**
- Upload Markdown-formatted medical textbooks via the web interface
- System automatically chunks, vectorizes, and stores

## Configuration

Main configuration in `config.py`:

- **Models**: Embedding model, LLM model
- **Document Processing**: Chunk size (600 chars), overlap (3 lines), batch size (20)
- **Search**: Query expansion count, recall count, top-k, threshold
- **RAG**: Recall results (10), rerank top-k (5)

## Recent Technical Fixes

### v1.1.0 (2026-01-16)

#### 1. Project Cleanup
- **Removed redundant files**: Archived `app.py`, `app_v2.py`, `convert.py`, `test_*.py`, `verify_*.py`
- **Cleaned up documentation**: Moved summary markdown files to `archives/`
- **Consolidated to single main app**: Now using only `app_v3.py`

#### 2. Parallel Retrieval Architecture
- **Created `parallel_retriever.py`**: Implements concurrent retrieval from 3 sources
  - Knowledge Base (ChromaDB)
  - PubMed (medical literature)
  - Model Knowledge (MedGemma's own knowledge)
- **Created `result_fuser.py`**: Merges multi-source results
  - Deduplication using similarity threshold (0.75)
  - Priority-based sorting (PubMed > Knowledge Base > Model)
  - Confidence calculation using source weights

#### 3. Bug Fix: Distance Calculation
- **Fixed `TypeError` in parallel_retriever.py**
- **Issue**: ChromaDB returned nested tuple structure causing `unsupported operand type(s) for -: 'tuple' and 'tuple'`
- **Fix**: Added explicit type conversion `float(dist) if dist else 0.5`
- **Improved distance normalization**: Now correctly calculates relevance scores from 0-1

#### 4. Query Expansion Fix
- **Problem**: MedGemma generated English query variations (e.g., "How to diagnose fever?")
- **Solution**: Updated prompt in `search.py` to require Chinese output
- **New prompt**:
```python
prompt = """针对这个医学问题，生成 {count} 个不同的检索关键词版本...
要求：1. 使用中文 2. 每个版本独占一行..."""
```
- **Result**: Now generates Chinese keywords like ['发热的诊断标准', '体温升高', '感染性疾病']

#### 5. Answer Generation Improvement
- **Simplified answer generation**: Direct MedGemma adapter call instead of Agent flow
- **Improved prompt**: Clearer instructions for Chinese medical answers
- **Enhanced cleaning**: Added regex patterns to remove MedGemma thought tags
- **Context optimization**: Simplified context to top 15 relevant lines

**Before (MedGemma output)**:
```
Here's a thinking process to arrive at the concise answer about diagnosing jaundice:
1. Understand the Goal...
```

**After (Cleaned Chinese answer)**:
```
黄疸的诊断需要结合临床表现、实验室检查和影像学检查。首先，通过测量血清总胆红素水平来初步判断是否存在黄疸。如果怀疑肝脏或胆道问题，则需要进行相应的检查。

B型超声波检查可以帮助评估肝脏大小、形态、是否有肿瘤或结石...
```

#### 6. Knowledge Base Optimization
- **Created `prepare_rag.py`**: Preprocesses medical textbooks for better RAG
- **Removed**: Table of contents, page numbers, copyright pages, digital resource markers
- **Retained**: 8 complete chapters with proper section structure
- **Result**: 1139 clean chunks from 诊断学_rag.md

#### 7. File Cleanup
- **Deleted**: `诊断学_cleaned.md` (replaced with optimized version)
- **Renamed**: Cleaned file now `诊断学_rag.md`

### Performance

| Metric | Before | After |
|--------|--------|-------|
| Knowledge Base Chunks | 1200 | 1139 |
| Query Expansion | English | Chinese |
| Answer Generation | Broken | Working |
| Parallel Retrieval | - | 3 sources |

## Core Algorithms

### 1. Document Processing Pipeline
```
PDF → Markdown → Smart Chunking (preserving section info) → Vectorization → Storage
```

### 2. Parallel Retrieval Flow
```
User Query → Query Expansion (Chinese keywords) → Parallel Search
  ├── Knowledge Base: ChromaDB with bge-m3 embeddings
  ├── PubMed: Medical literature search
  └── Model Knowledge: Direct MedGemma query
→ Result Fusion (deduplication + prioritization)
→ Context Building for Answer Generation
→ MedGemma Response
```

### 3. Confidence Scoring
```python
# Source weights
SOURCE_WEIGHTS = {
    "pubmed": 0.9,      # Highest priority - authoritative medical literature
    "knowledge_base": 0.7,  # Medium priority - local knowledge base
    "model": 0.5        # Lower priority - model's own knowledge
}

# Final confidence = source_weight * relevance_score
confidence = SOURCE_WEIGHTS[source] * relevance_score
```

## Development Guide

### Adding New Features

1. **New Search Strategy**: Modify `src/rag/search.py`
2. **New Retrieval Logic**: Modify `src/retrieval/parallel_retriever.py`
3. **New Answer Generation**: Modify `app_v3.py` `generate_answer_with_context()`
4. **New Agent Logic**: Modify `src/agent/core.py`

### Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/test_medgemma_integration.py

# Run performance benchmarks
python tests/performance/benchmark.py
```

## Important Notes

- **Medical Disclaimer**: This system is for learning and reference only, cannot replace professional medical advice
- **Data Security**: Medical data should be properly stored, local deployment recommended
- **Model Choice**: MedGemma is optimized for medical reasoning

## License

MIT

## Contributing

Issues and Pull Requests welcome!
