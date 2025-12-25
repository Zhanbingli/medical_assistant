"""
数据库模块 - ChromaDB 操作封装
提供医学知识库的增删改查功能
"""
import chromadb
from typing import Set, List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalKnowledgeDB:
    """医学知识库数据库类"""

    def __init__(self, db_path: str, collection_name: str):
        """
        初始化数据库连接

        Args:
            db_path: 数据库存储路径
            collection_name: 集合名称
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"数据库已初始化: {db_path}/{collection_name}")

    def get_existing_files(self) -> Set[str]:
        """
        获取数据库中已存储的所有文件名

        Returns:
            文件名集合
        """
        try:
            data = self.collection.get(include=['metadatas'])
            if not data['metadatas']:
                return set()

            files = set([
                m.get('source')
                for m in data['metadatas']
                if m and m.get('source')
            ])
            logger.info(f"数据库中共有 {len(files)} 个文件")
            return files
        except Exception as e:
            logger.error(f"获取文件列表失败: {e}")
            return set()

    def delete_file(self, filename: str) -> tuple[bool, Optional[str]]:
        """
        从数据库中删除指定文件的所有片段

        Args:
            filename: 要删除的文件名

        Returns:
            (成功标志, 错误信息)
        """
        try:
            self.collection.delete(where={"source": filename})
            logger.info(f"已删除文件: {filename}")
            return True, None
        except Exception as e:
            error_msg = f"删除失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """
        批量添加知识片段

        Args:
            ids: 文档ID列表
            embeddings: 向量列表
            documents: 文档内容列表
            metadatas: 元数据列表

        Returns:
            (成功标志, 错误信息)
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.debug(f"成功添加 {len(ids)} 条数据")
            return True, None
        except Exception as e:
            error_msg = f"添加数据失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量

        Returns:
            检索结果字典
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            logger.debug(f"检索到 {len(results.get('documents', [[]])[0])} 条结果")
            return results
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {"documents": [[]], "metadatas": [[]]}

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            count = self.collection.count()
            files = self.get_existing_files()
            return {
                "total_chunks": count,
                "total_files": len(files),
                "files": list(files)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"total_chunks": 0, "total_files": 0, "files": []}
