# 数据库操作与Markdown处理指南

## 📚 删除功能详解

### 删除按钮会删除什么？

删除按钮会**仅删除数据库中的向量数据**，不会删除你的原始文件。

**具体删除内容**:
- ✅ ChromaDB中该文件的所有**向量嵌入**（embeddings）
- ✅ 该文件的**元数据**（文件名、章节索引、chunk长度等）
- ✅ **自动释放空间**：删除后数据库文件会变小

**不会删除**:
- ❌ 原始Markdown文件（仍在你的硬盘上）
- ❌ 数据库文件本身（`chroma.sqlite3` 不会被删除）
- ❌ 其他文件的数据

**技术实现**:
```python
# src/rag/database.py:76
self.collection.delete(where={"source": filename})
```

这执行的是**条件删除**，只删除 `source` 字段匹配该文件名的记录。

### 如何恢复已删除的数据？

因为原始文件还在，只需**重新上传**即可：

```bash
# 1. 在UI中点击"上传新书"
# 2. 选择之前删除的同名Markdown文件
# 3. 点击"开始学习"
```

系统会重新处理文件并生成向量。

---

## 📝 Markdown预处理与优化

### 为什么需要预处理？

PDF转Markdown时常带有以下问题，影响检索质量：

1. **页码干扰**：干扰向量相似度计算
2. **页眉页脚**：无关信息混入
3. **断裂段落**：影响语义完整性
4. **图片占位符**：无用信息
5. **结构混乱**：标题识别不准确

### 预处理效果示例

**优化前**:
```markdown
诊断学

第一章 绪论

1

扫描全能王

第一节 发热的定义

发热是指...

2

仅供学习交流
```

**优化后**:
```markdown
# 第一章 绪论

## 第一节 发热的定义

发热是指...
```

### 三种预处理方式

#### 方法1: 命令行工具（推荐）

```bash
# 基础用法（使用默认关键词过滤）
python scripts/clean_md.py 诊断学_cleaned.md 诊断学_优化.md --default-keywords

# 自定义关键词（去除特定页眉页脚）
python scripts/clean_md.py 输入.md 输出.md \
  --keywords "诊断学" "第.篇" "扫描全能王" "金山OFD"

# 查看帮助
python scripts/clean_md.py --help
```

**默认过滤关键词**:
- "诊断学"
- "第.篇" (匹配"第一篇"、"第二篇"等)
- "第.章" (匹配"第一章"、"第二章"等)
- "Page" (英文页码标记)
- "仅供学习交流" (常见水印)
- "扫描全能王" (扫描APP水印)

#### 方法2: Python API调用

在你的脚本中使用：

```python
from src.utils.markdown_optimizer import optimize_markdown_for_rag

# 读取原始文件
with open('诊断学_cleaned.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 优化（使用默认关键词）
optimized = optimize_markdown_for_rag(content)

# 保存优化结果
with open('诊断学_优化.md', 'w', encoding='utf-8') as f:
    f.write(optimized)

# 或者添加自定义关键词
optimized = optimize_markdown_for_rag(
    content, 
    remove_keywords=["自定义关键词1", "自定义关键词2"]
)
```

#### 方法3: UI界面（一键优化）

在 `app_v2.py` 中上传文件时，**开启"自动优化Markdown"选项**：

```python
# 侧边栏设置
optimize_md = st.toggle("🔧 自动优化Markdown", value=True, 
                       help="启用后自动清理页码、页眉页脚并优化结构")
```

启用后，上传的文件会自动经过优化再存入知识库。

---

## 🔄 完整工作流程

### 从PDF到知识库的最佳实践

```bash
# 步骤1: PDF转Markdown（如果还没有）
# 确保已安装 pymupdf4llm
pip install pymupdf4llm

# 转换为Markdown
python convert.py 诊断学.pdf -o 诊断学_raw.md

# 步骤2: 预处理Markdown
python scripts/clean_md.py 诊断学_raw.md 诊断学_优化.md --default-keywords

# 步骤3: 启动应用并上传
streamlit run app_v2.py

# 在UI中：
# 1. 点击"上传新书"
# 2. 选择"诊断学_优化.md"
# 3. 开启"自动优化Markdown"（可选，如果已预处理可关闭）
# 4. 点击"开始学习"
```

### 批量处理多本书籍

```bash
#!/bin/bash
# batch_process.sh

# 创建输出目录
mkdir -p processed

# 处理所有PDF文件
for pdf in *.pdf; do
  if [ -f "$pdf" ]; then
    echo "处理: $pdf"
    
    # 1. PDF转Markdown
    python convert.py "$pdf" -o "processed/${pdf%.pdf}_raw.md"
    
    # 2. 优化Markdown
    python scripts/clean_md.py "processed/${pdf%.pdf}_raw.md" "processed/${pdf%.pdf}_优化.md" --default-keywords
    
    echo "完成: ${pdf%.pdf}_优化.md"
  fi
done

echo "所有文件处理完成！请到processed/目录查看"
```

运行:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

---

## 🔧 高级配置

### 自定义删除关键词

编辑 `scripts/clean_md.py` 的默认关键词列表：

```python
defaults = [
    "诊断学",
    "第.篇",
    "第.章",
    "Page",
    "仅供学习交流",
    "扫描全能王",
    # 添加你的关键词
    "你的医院名称",
    "内部资料",
]
```

### 调整分块参数

如果优化后的Markdown检索效果仍不理想，可以调整分块参数：

编辑 `config.py`:

```python
CHUNK_SIZE = 800  # 增大块大小（默认600）
CHUNK_OVERLAP_LINES = 5  # 增加重叠行数（默认3）
BATCH_SIZE = 30  # 增大批处理（默认20）
```

**调整原则**:
- **内容密集**的章节 → 减小 `CHUNK_SIZE`（400-500）
- **内容稀疏**的章节 → 增大 `CHUNK_SIZE`（800-1000）
- **希望保留更多上下文** → 增大 `CHUNK_OVERLAP_LINES`（5-8）

---

## 📊 效果验证

### 如何验证优化效果？

#### 1. 文件大小对比

```bash
# 优化前
ls -lh 诊断学_cleaned.md
# -rw-r--r-- 1 user 2.1M Dec 21 11:05 诊断学_cleaned.md

# 优化后
ls -lh 诊断学_优化.md  
# -rw-r--r-- 1 user 1.8M Dec 21 11:05 诊断学_优化.md
# 减少了约 14%
```

#### 2. 检索质量测试

在UI中测试同一个问题，对比优化前后的检索结果：

**优化前可能返回**:
```
[来源: 诊断学_cleaned.md]: 第一章
12  
扫描全能王
发热定义...
```

**优化后返回**:
```
【章节：第一章 > 第一节】
## 发热的定义

发热是指体温升高...
```

优化后更少无关信息，检索更准确。

#### 3. Debug模式查看

在 `app_v2.py` 中开启Debug模式，查看Rerank打分，观察返回的文档是否干净。

---

## ❓ 常见问题

### Q1: 删除后为什么数据库文件没有变小？

**A**: ChromaDB使用SQLite，删除后空间不会立即释放。可以执行VACUUM：

```bash
# 停止应用后执行
cd medical_db
sqlite3 chroma.sqlite3 "VACUUM;"
```

或者下次大量插入数据时，SQLite会自动复用已删除的空间。

### Q2: 多次优化会损坏内容吗？

**A**: 不会丢失主要内容。但多次应用标题转换可能导致标题级别混乱（如 ## 变成 ####）。

建议：**只优化一次**，保存为新的文件。

### Q3: 为什么优化后检索效果反而变差？

**可能原因**:
1. 过度删除导致关键信息丢失
2. 标题结构被破坏
3. 段落过长或过短

**解决方案**: 
- 检查 `--keywords` 是否误删了重要内容
- 调整 `CHUNK_SIZE` 参数
- 查看 `scripts/clean_md.py` 的 `RAG优化` 部分，注释掉某些变换

### Q4: 可以同时上传多个文件吗？

**A**: 可以！在UI中支持多文件上传，会自动批量处理。

**建议**:
- 先分别优化每个文件
- 批量上传到同一个知识库
- 系统会自动去重（同名文件会跳过）

---

## 🎯 最佳实践总结

| 步骤 | 操作 | 工具 | 备注 |
|------|------|------|------|
| 1 | PDF → Markdown | `convert.py` | 保留原始文件 |
| 2 | Markdown优化 | `clean_md.py` | 使用 `--default-keywords` |
| 3 | 验证优化结果 | 文本编辑器 | 检查章节结构 |
| 4 | 上传到知识库 | `app_v2.py` | 开启"自动优化"（可选） |
| 5 | 测试检索效果 | UI界面 | 使用Debug模式查看 |
| 6 | 需要删除 | UI界面 | 点击🗑️按钮 |
| 7 | 删除后恢复 | UI界面 | 重新上传优化后的文件 |

---

## 📚 文件命名建议

```
原始: 诊断学.pdf
步骤1: 诊断学_raw.md        # PDF转换结果
步骤2: 诊断学_优化.md        # 预处理结果
步骤3: [存入知识库]         # 上传后只保留 诊断学_优化.md
```

这样即使从知识库删除，你仍有优化后的文件可以重新上传。

---

## 🎉 快速开始命令

```bash
# 1. 优化现有书籍
python scripts/clean_md.py 诊断学_cleaned.md 诊断学_优化.md --default-keywords

# 2. 启动新UI
streamlit run app_v2.py

# 3. 上传优化后的文件
# 在UI中: 📤 上传新书 → 选择 "诊断学_优化.md" → 🚀 开始学习

# 4. 测试检索
# 在聊天框输入: "发热伴咳嗽需要注意哪些诊断"
```

---

## 🔗 相关文件

- **转换工具**: `convert.py` (PDF → Markdown)
- **优化工具**: `scripts/clean_md.py` (Markdown预处理)
- **优化库**: `src/utils/markdown_optimizer.py` (核心优化逻辑)
- **数据库**: `medical_db/chroma.sqlite3` (向量存储)

---

**总结**: 删除按钮只删向量数据不删原始文件，预处理工具能显著提升检索质量，建议所有上传的文件都经过优化处理。