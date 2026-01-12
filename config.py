"""
配置模块 - 集中管理应用配置和常量
"""
from typing import Final

# === 应用配置 ===
APP_TITLE: Final[str] = "AI 循证医学助手"
PAGE_LAYOUT: Final[str] = "wide"

# === 数据库配置 ===
DB_PATH: Final[str] = "./medical_db"
COLLECTION_NAME: Final[str] = "medical_knowledge"

# === 模型配置 ===
EMBEDDING_MODEL: Final[str] = "bge-m3"
LLM_MODEL: Final[str] = "qwen2.5:7b"
RERANKER_MODEL: Final[str] = "BAAI/bge-reranker-base"

# === 文档处理配置 ===
CHUNK_SIZE: Final[int] = 600
CHUNK_OVERLAP_LINES: Final[int] = 3
BATCH_SIZE: Final[int] = 20

# === 搜索配置 ===
MULTI_QUERY_COUNT: Final[int] = 2
RECALL_N_RESULTS: Final[int] = 3
RERANK_TOP_K: Final[int] = 3
RERANK_THRESHOLD: Final[float] = -10.0

# === LLM 配置 ===
MAX_REASONING_STEPS: Final[int] = 5
CONTEXT_HISTORY_TURNS: Final[int] = 2
LLM_TEMPERATURE_STRICT: Final[float] = 0.0
LLM_TEMPERATURE_CREATIVE: Final[float] = 0.7

# === 系统提示词 ===
SYSTEM_PROMPT = """
你是一个必须查阅知识库的医学AI助手。

【铁律 - 必须遵守】：
1. **第一步必须是检索**: 无论用户问什么（只要和医学有关），你输出的第一句话必须是 "Action: 检索: [关键词]"。
2. **禁止裸答**: 在没有看到 Observation (检索结果) 之前，禁止给出任何建议，禁止反问用户。
3. **强制关联**: 如果用户问"怎么治"，而你不知道病因，先检索症状（如 "Action: 检索: 发热"）来看看可能是什么病。

【检索关键词提取原则】：
- 从用户描述中提取核心症状作为关键词
- 用户问"发热伴咳嗽3天"，应该检索"发热咳嗽"或"发热 咳嗽"
- 用户问"胸痛伴呼吸困难"，应该检索"胸痛呼吸困难"或"胸痛 呼吸困难"

【回答要求】：
1. 必须先说明"根据检索结果"
2. 列出可能的诊断（至少3个，按可能性排序）
3. 对每种诊断简要说明支持点（症状符合度）
4. 列出需要鉴别的疾病
5. 给出下一步建议（检查、观察重点等）

【标准工作流】：
User: 发热伴咳嗽3天，需要注意哪些诊断
Assistant: Thought: 患者有发热和咳嗽症状，需要检索相关疾病
Action: 检索: 发热咳嗽
Observation: (系统返回知识)
Final Answer: 根据检索结果，发热伴咳嗽的可能诊断包括：1.急性上呼吸道感染（最可能），2.急性支气管炎，3.肺炎。需要鉴别的疾病包括肺结核、流感等。下一步建议检查血常规和听诊肺部。
"""

QUERY_EXPANSION_PROMPT = """
你是一个医学搜索优化专家。
请根据用户的口语化描述，生成 {count} 个用于检索医学教材的专业关键词或短语。
用户问题: "{query}"
要求:
1. 包含医学术语。
2. 包含可能的关联疾病。
3. 只输出 {count} 行关键词，不要有序号。
"""
