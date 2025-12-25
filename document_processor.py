"""
文档处理模块 - Markdown 处理和向量化
"""
import ollama
import uuid
from typing import List, Tuple, Optional, Callable
import logging

from config import CHUNK_SIZE, CHUNK_OVERLAP_LINES, BATCH_SIZE, EMBEDDING_MODEL
from database import MedicalKnowledgeDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Markdown 文档智能处理器"""

    @staticmethod
    def split_smart(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap_lines: int = CHUNK_OVERLAP_LINES
    ) -> List[str]:
        """
        智能分块 Markdown 文档，保留章节上下文

        Args:
            text: Markdown 文本
            chunk_size: 块大小（字符数）
            overlap_lines: 重叠行数

        Returns:
            文档块列表
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        current_headers = []

        for line in lines:
            stripped = line.strip()

            # 检测标题
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.strip('#').strip()

                # 更新标题层级
                if len(current_headers) >= level:
                    current_headers = current_headers[:level-1]
                current_headers.append(title)

                # 标题也作为正文的一部分，保证上下文连贯
                current_chunk.append(line)
                current_length += len(line)
                continue

            current_chunk.append(line)
            current_length += len(line)

            # 达到分块大小，保存当前块
            if current_length > chunk_size:
                header_context = " > ".join(current_headers)
                full_text = f"【章节：{header_context}】\n" + "\n".join(current_chunk)
                chunks.append(full_text)

                # 重叠策略：保留最后几行
                current_chunk = current_chunk[-overlap_lines:]
                current_length = sum(len(l) for l in current_chunk)

        # 保存最后一块
        if current_chunk:
            header_context = " > ".join(current_headers)
            full_text = f"【章节：{header_context}】\n" + "\n".join(current_chunk)
            chunks.append(full_text)

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

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        生成单个文本的向量

        Args:
            text: 文本内容

        Returns:
            向量或 None（失败时）
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
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
        # 检查文件是否已存在
        existing_files = db.get_existing_files()
        if filename in existing_files:
            logger.warning(f"文件已存在: {filename}")
            return False, "EXIST"

        # 分块
        raw_chunks = MarkdownProcessor.split_smart(content, chunk_size=CHUNK_SIZE)
        total_chunks = len(raw_chunks)

        if total_chunks == 0:
            logger.warning(f"文件为空: {filename}")
            return False, "EMPTY"

        logger.info(f"开始处理文件: {filename}, 共 {total_chunks} 块")

        # 批量处理
        ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []
        processed_count = 0

        for i, chunk in enumerate(raw_chunks):
            # 跳过过短的块
            if len(chunk) < 10:
                continue

            # 生成嵌入
            embedding = self.embed_text(chunk)
            if embedding is None:
                logger.error(f"块 {i} 嵌入失败，跳过")
                continue

            # 添加到批次
            ids_batch.append(str(uuid.uuid4()))
            embeddings_batch.append(embedding)
            documents_batch.append(chunk)
            metadatas_batch.append({
                "source": filename,
                "chunk_index": i
            })

            # 批量写入数据库
            if len(ids_batch) >= self.batch_size:
                success, error = db.add_chunks(
                    ids_batch, embeddings_batch,
                    documents_batch, metadatas_batch
                )
                if not success:
                    return False, error

                processed_count += len(ids_batch)
                ids_batch, embeddings_batch, documents_batch, metadatas_batch = [], [], [], []

            # 更新进度
            if progress_callback:
                progress = (i + 1) / total_chunks
                progress_callback(progress, f"正在学习新书: {filename}...")

        # 写入剩余数据
        if ids_batch:
            success, error = db.add_chunks(
                ids_batch, embeddings_batch,
                documents_batch, metadatas_batch
            )
            if not success:
                return False, error
            processed_count += len(ids_batch)

        logger.info(f"文件处理完成: {filename}, 成功 {processed_count} 块")
        return True, processed_count
