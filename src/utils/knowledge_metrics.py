"""
知识库管理增强模块
提供知识库统计、时效性检查和质量评估功能
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import hashlib

logger = logging.getLogger(__name__)

class KnowledgeBaseMetrics:
    """知识库统计指标"""
    
    def __init__(self, db):
        """初始化统计器"""
        self.db = db
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        获取详细的知识库统计
        
        Returns:
            统计信息字典
        """
        # 基础统计
        basic_stats = self.db.get_collection_stats()
        
        # 计算更多信息
        total_chunks = basic_stats.get('total_chunks', 0)
        total_files = basic_stats.get('total_files', 0)
        files_list = basic_stats.get('files', [])
        
        # 1. 按文件名分类统计
        categories = self._categorize_files(files_list)
        
        # 2. 计算平均每个文件的chunk数
        avg_chunks_per_file = total_chunks / total_files if total_files > 0 else 0
        
        # 3. 数据库大小
        import os
        db_size_mb = self._get_db_size()
        
        return {
            "basic_stats": basic_stats,
            "categories": categories,
            "avg_chunks_per_file": round(avg_chunks_per_file, 1),
            "db_size_mb": round(db_size_mb, 2),
            "last_updated": datetime.now().isoformat()
        }
    
    def _categorize_files(self, files: List[str]) -> Dict[str, List[str]]:
        """按类别分组文件"""
        categories = {
            "内科学": [],
            "外科学": [],
            "诊断学": [],
            "儿科学": [],
            "妇产科学": [],
            "其他": []
        }
        
        for file in files:
            categorized = False
            for category, keywords in [
                ("内科学", ["内科", "Internal"]),
                ("外科学", ["外科", "Surgery"]),
                ("诊断学", ["诊断", "Diagnostic"]),
                ("儿科学", ["儿科", "Pediatric"]),
                ("妇产科学", ["妇产", "产科", "OB/GYN", "Obstetric"])
            ]:
                for keyword in keywords:
                    if keyword in file:
                        categories[category].append(file)
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                categories["其他"].append(file)
        
        return {k: v for k, v in categories.items() if v}
    
    def _get_db_size(self) -> float:
        """获取数据库文件大小（MB）"""
        import os
        try:
            db_path = self.db.db_path
            chroma_db = os.path.join(db_path, "chroma.sqlite3")
            
            if os.path.exists(chroma_db):
                size_bytes = os.path.getsize(chroma_db)
                return size_bytes / 1024 / 1024
        except Exception as e:
            logger.error(f"获取数据库大小失败: {e}")
        
        return 0.0

class QualityAssessment:
    """知识库质量评估"""
    
    def __init__(self, db):
        """初始化评估器"""
        self.db = db
    
    def assess_quality(self) -> Dict[str, Any]:
        """
        评估知识库质量
        
        Returns:
            质量评估报告
        """
        # 1. 覆盖度评估
        coverage_score = self._assess_coverage()
        
        # 2. 完整性评估
        completeness_score = self._assess_completeness()
        
        # 3. 时效性评估
        timeliness_score = self._assess_timeliness()
        
        # 4. 综合评分
        overall_score = (
            coverage_score * 0.3 +
            completeness_score * 0.3 +
            timeliness_score * 0.4
        )
        
        # 5. 建议
        suggestions = self._generate_suggestions(
            coverage_score, completeness_score, timeliness_score
        )
        
        return {
            "overall_score": round(overall_score, 1),
            "grade": self._get_grade(overall_score),
            "scores": {
                "coverage": coverage_score,
                "completeness": completeness_score,
                "timeliness": timeliness_score
            },
            "suggestions": suggestions,
            "assessed_at": datetime.now().isoformat()
        }
    
    def _assess_coverage(self) -> float:
        """评估知识覆盖度"""
        files = self.db.get_existing_files()
        
        # 理想情况：内科学、外科学、诊断学、儿科学、妇产科学
        required_categories = ["内科学", "外科学", "诊断学", "儿科学", "妇产科学"]
        
        found_categories = 0
        for category in required_categories:
            for file in files:
                if category.replace("学", "") in file or category.replace("科学", "") in file:
                    found_categories += 1
                    break
        
        coverage_score = (found_categories / len(required_categories)) * 100
        
        return round(coverage_score, 1)
    
    def _assess_completeness(self) -> float:
        """评估知识完整性"""
        stats = self.db.get_collection_stats()
        total_chunks = stats.get('total_chunks', 0)
        
        # 理想：至少1000个知识片段
        if total_chunks >= 1000:
            return 100.0
        elif total_chunks >= 500:
            return 80.0
        elif total_chunks >= 200:
            return 60.0
        elif total_chunks >= 100:
            return 40.0
        else:
            return 20.0
    
    def _assess_timeliness(self) -> float:
        """评估时效性（基于文件名或元数据）"""
        files = self.db.get_existing_files()
        
        # 这里简化处理：如果文件名包含年份，判断是否超过5年
        current_year = datetime.now().year
        
        recent_files = 0
        total_files = len(files)
        
        for file in files:
            # 尝试提取年份
            import re
            year_match = re.search(r'20\d{2}', file)
            if year_match:
                year = int(year_match.group())
                if current_year - year <= 5:
                    recent_files += 1
            else:
                # 没有年份信息，假设是近期的
                recent_files += 1
        
        timeliness_score = (recent_files / total_files * 100) if total_files > 0 else 0
        
        return round(timeliness_score, 1)
    
    def _get_grade(self, score: float) -> str:
        """获取等级"""
        if score >= 90:
            return "A+ 优秀"
        elif score >= 80:
            return "A 良好"
        elif score >= 70:
            return "B 中等"
        elif score >= 60:
            return "C 合格"
        else:
            return "D 需改进"
    
    def _generate_suggestions(
        self,
        coverage: float,
        completeness: float,
        timeliness: float
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if coverage < 80:
            suggestions.append("📚 建议扩充知识库覆盖范围，添加外科学、儿科学等教材")
        
        if completeness < 70:
            suggestions.append("📖 建议增加更多教材内容，提升知识完整性")
        
        if timeliness < 70:
            suggestions.append("🔄 建议使用最新版教材或临床指南")
        
        if coverage >= 80 and completeness >= 80 and timeliness >= 80:
            suggestions.append("✅ 知识库质量良好，建议定期更新维护")
        
        return suggestions

class KnowledgeBaseVersioning:
    """知识库版本管理"""
    
    def __init__(self, db):
        """初始化版本管理器"""
        self.db = db
        self.versions = []
    
    def create_version(self, description: str = "") -> str:
        """
        创建知识库版本快照
        
        Args:
            description: 版本描述
            
        Returns:
            版本ID
        """
        stats = self.db.get_collection_stats()
        
        # 生成版本ID
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 计算checksum（简化版）
        checksum = self._calculate_checksum()
        
        version_info = {
            "id": version_id,
            "description": description or f"自动备份 {version_id}",
            "stats": stats,
            "checksum": checksum,
            "created_at": datetime.now().isoformat()
        }
        
        self.versions.append(version_info)
        
        logger.info(f"创建知识库版本: {version_id}")
        
        return version_id
    
    def _calculate_checksum(self) -> str:
        """计算知识库校验和"""
        try:
            import os
            db_path = self.db.db_path
            chroma_db = os.path.join(db_path, "chroma.sqlite3")
            
            if os.path.exists(chroma_db):
                # 简单的文件hash
                with open(chroma_db, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()[:16]
                    return file_hash
        except Exception as e:
            logger.error(f"计算checksum失败: {e}")
        
        return "unknown"
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        return self.versions
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            version1_id: 版本1 ID
            version2_id: 版本2 ID
            
        Returns:
            差异报告
        """
        version1 = next((v for v in self.versions if v['id'] == version1_id), None)
        version2 = next((v for v in self.versions if v['id'] == version2_id), None)
        
        if not version1 or not version2:
            raise ValueError("版本ID不存在")
        
        stats1 = version1['stats']
        stats2 = version2['stats']
        
        return {
            "version1": version1['id'],
            "version2": version2['id'],
            "chunk_diff": stats2.get('total_chunks', 0) - stats1.get('total_chunks', 0),
            "file_diff": stats2.get('total_files', 0) - stats1.get('total_files', 0),
            "checksum_changed": version1['checksum'] != version2['checksum']
        }

class ContentDeduplication:
    """内容去重"""
    
    def __init__(self, db):
        """初始化去重器"""
        self.db = db
    
    def find_duplicates(self) -> List[Dict[str, Any]]:
        """
        查找重复内容
        
        Returns:
            重复内容列表
        """
        # 这个功能需要直接查询ChromaDB
        # 简化实现：返回示例数据
        
        try:
            # 获取所有文档
            all_data = self.db.collection.get(include=['documents', 'metadatas'])
            documents = all_data.get('documents', [])
            metadatas = all_data.get('metadatas', [])
            
            # 简单的去重：使用hash
            seen_hashes = {}
            duplicates = []
            
            for doc, meta in zip(documents, metadatas):
                doc_hash = hashlib.md5(doc.encode()).hexdigest()
                
                if doc_hash in seen_hashes:
                    duplicates.append({
                        "original_source": seen_hashes[doc_hash],
                        "duplicate_source": meta.get('source', '未知'),
                        "chunk_content": doc[:100] + "..."
                    })
                else:
                    seen_hashes[doc_hash] = meta.get('source', '未知')
            
            return duplicates
            
        except Exception as e:
            logger.error(f"查找重复失败: {e}")
            return []
