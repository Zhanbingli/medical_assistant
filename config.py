"""
配置模块 - 集中管理应用配置和常量
"""

# === 应用配置 ===
APP_TITLE = "AI 循证医学助手"
PAGE_LAYOUT = "wide"

# === 数据库配置 ===
DB_PATH = "./medical_db"
COLLECTION_NAME = "medical_knowledge"

# === 模型配置 ===
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "qwen2.5:7b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# === 文档处理配置 ===
CHUNK_SIZE = 600
CHUNK_OVERLAP_LINES = 3
BATCH_SIZE = 20

# === 搜索配置 ===
MULTI_QUERY_COUNT = 3
RECALL_N_RESULTS = 5
RERANK_TOP_K = 3
RERANK_THRESHOLD = -10

# === LLM 配置 ===
MAX_REASONING_STEPS = 5
CONTEXT_HISTORY_TURNS = 2
LLM_TEMPERATURE_STRICT = 0
LLM_TEMPERATURE_CREATIVE = 0.7

# === 系统提示词 ===
SYSTEM_PROMPT = """
你是一个必须查阅知识库的医学AI助手。

【铁律 - 必须遵守】：
1. **第一步必须是检索**：无论用户问什么（只要和医学有关），你输出的第一句话必须是 "Action: 检索: [关键词]"。
2. **禁止裸答**：在没有看到 Observation (检索结果) 之前，禁止给出任何建议，禁止反问用户。
3. **强制关联**：如果用户问"怎么治"，而你不知道病因，先检索症状（如 "Action: 检索: 发热寒战"）来看看可能是什么病。

【标准工作流】：
User: 发热伴寒战
Assistant: Thought: 用户提到症状，我必须先查库。
Action: 检索: 发热伴寒战
Observation: (系统返回知识)
Final Answer: 根据资料，这可能是...
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
