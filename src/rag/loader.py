"""
文档处理模块 - Markdown 处理和向量化
"""
import ollama
import uuid
from typing import List, Tuple, Optional, Callable
import logging
from functools import lru_cache

from config import CHUNK_SIZE, CHUNK_OVERLAP_LINES, BATCH_SIZE, EMBEDDING_MODEL
from .database import MedicalKnowledgeDB

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Markdown 文档智能处理器 - 优化版"""

    @staticmethod
    def split_smart(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap_lines: int = CHUNK_OVERLAP_LINES
    ) -> List[str]:
        """
        智能分块 Markdown 文档，保留章节上下文
        优化：按段落和句子切分，保持医学术语完整性

        Args:
            text: Markdown 文本
            chunk_size: 块大小（字符数）
            overlap_lines: 重叠行数

        Returns:
            文档块列表
        """
        if not text or not text.strip():
            logger.warning("输入文本为空或仅包含空白字符")
            return []
        
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_length = 0
        current_headers = []
        
        # 段落分隔符：医学教材通常在空行处分段
        paragraph_breaks = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 检测标题层级
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.strip('#').strip()
                
                # 更新标题层级结构
                if len(current_headers) >= level:
                    current_headers = current_headers[:level-1]
                current_headers.append(title)
                
                # 标题作为上下文保留
                current_chunk_lines.append(line)
                current_length += len(line)
                continue
            
            # 累加行
            current_chunk_lines.append(line)
            current_length += len(line)
            
            # 检查是否达到块大小限制
            if current_length > chunk_size:
                # 尝试在段落边界切分
                chunk_text = '\n'.join(current_chunk_lines)
                
                # 找到最后一个段落分隔点
                last_para_break = chunk_text.rfind('\n\n')
                if last_para_break > chunk_size * 0.5:  # 至少包含50%内容
                    # 在段落边界切分
                    chunk_content = chunk_text[:last_para_break].strip()
                    next_content = chunk_text[last_para_break:].strip()
                    
                    if chunk_content:
                        header_context = " > ".join(current_headers) if current_headers else "未分类"
                        full_chunk = f"【章节：{header_context}】\n\n{chunk_content}"
                        chunks.append(full_chunk)
                    
                    # 重置块，从下一段开始
                    current_chunk_lines = [line] if stripped else []
                    if next_content:
                        # 保留部分内容到下一块
                        next_lines = next_content.split('\n')
                        current_chunk_lines = next_lines[:5]  # 最多保留5行
                    current_length = sum(len(l) for l in current_chunk_lines)
                else:
                    # 硬切分，保留语义完整性
                    header_context = " > ".join(current_headers) if current_headers else "未分类"
                    full_chunk = f"【章节：{header_context}】\n\n{chunk_text}"
                    chunks.append(full_chunk)
                    
                    # 重叠策略：保留最后几行
                    current_chunk_lines = current_chunk_lines[-overlap_lines:]
                    current_length = sum(len(l) for l in current_chunk_lines)

        # 保存最后一块
        if current_chunk_lines:
            header_context = " > ".join(current_headers) if current_headers else "未分类"
            full_chunk = f"【章节：{header_context}】\n\n" + '\n'.join(current_chunk_lines)
            chunks.append(full_chunk)

        logger.info(f"文档已分块: 共 {len(chunks)} 块")
        return chunks


class DocumentEmbedder:
    """文档向量化器"""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = BATCH_SIZE
    ):
        """
        初始化嵌入器

        Args:
            model_name: 嵌入模型名称
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"文档嵌入器已初始化: 模型={model_name}, 批大小={batch_size}")

    @lru_cache(maxsize=128)
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        生成单个文本的向量（带缓存）

        Args:
            text: 文本内容

        Returns:
            向量或 None（失败时）
        """
        if not text or len(text.strip()) < 5:
            logger.warning("文本过短，跳过嵌入生成")
            return None
            
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response.get('embedding')
            if embedding is None:
                logger.error(f"Ollama 返回的 embedding 为 None")
                return None
            return embedding
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}, 文本长度: {len(text)}")
            return None

    def process_file(
        self,
        content: str,
        filename: str,
        db: MedicalKnowledgeDB,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[bool, any]:
        """
        完整的文件处理流程：分块 -> 嵌入 -> 存储

        Args:
            content: 文件内容
            filename: 文件名
            db: 数据库实例
            progress_callback: 进度回调函数 (进度比例, 状态文本)

        Returns:
            (成功标志, 结果信息/错误信息)
        """
        if not content or not content.strip():
            logger.warning(f"文件内容为空: {filename}")
            return False, "EMPTY"
            
        # 检查文件是否已存在
        existing_files = db.get_existing_files()
        if filename in existing_files:
            logger.warning(f"文件已存在: {filename}")
            return False, "EXIST"

        # 分块
        raw_chunks = MarkdownProcessor.split_smart(content, chunk_size=CHUNK_SIZE)
        total_chunks = len(raw_chunks)

        if total_chunks == 0:
            logger.warning(f"文件分块后为空: {filename}")
            return False, "EMPTY"

        logger.info(f"开始处理文件: {filename}, 共 {total_chunks} 块")

        # 批量处理
        ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
        processed_count = 0
        failed_count = 0

        for i, chunk in enumerate(raw_chunks):
            # 跳过过短的块
            if len(chunk) < 10:
                logger.debug(f"跳过过短块 (索引 {i}, 长度 {len(chunk)})")
                failed_count += 1
                continue

            # 生成嵌入
            embedding = self.embed_text(chunk)
            if embedding is None:
                logger.error(f"块 {i} 嵌入失败，跳过 (长度: {len(chunk)})")
                failed_count += 1
                continue

            # 添加到批次
            ids_batch.append(str(uuid.uuid4()))
            embeddings_batch.append(embedding)
            documents_batch.append(chunk)
            metadatas_batch.append({
                "source": filename,
                "chunk_index": i,
                "chunk_length": len(chunk)
            })

            # 批量写入数据库
            if len(ids_batch) >= self.batch_size:
                success, error = db.add_chunks(
                    ids_batch, embeddings_batch,
                    documents_batch, metadatas_batch
                )
                if not success:
                    logger.error(f"批量写入失败: {error}")
                    return False, error

                processed_count += len(ids_batch)
                ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []

            # 更新进度
            if progress_callback:
                try:
                    progress = (i + 1) / total_chunks
                    progress_callback(progress, f"正在学习新书: {filename}... ({processed_count + len(ids_batch)}/{total_chunks})")
                except Exception as e:
                    logger.warning(f"进度回调失败: {e}")

        # 写入剩余数据
        if ids_batch:
            success, error = db.add_chunks(
                ids_batch, embeddings_batch,
                documents_batch, metadatas_batch
            )
            if not success:
                logger.error(f"最后批次写入失败: {error}")
                return False, error
            processed_count += len(ids_batch)

        total_processed = processed_count + failed_count
        logger.info(f"文件处理完成: {filename}, 成功 {processed_count} 块, 失败 {failed_count} 块")
        
        if processed_count == 0:
            return False, "NO_VALID_CHUNKS"
            
        return True, processed_count
