"""
Markdown 处理集成模块
整合 clean_md.py 的功能到 app_v2.py
"""
import re
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class MarkdownOptimizer:
    """Markdown 优化器 - 为RAG检索优化文档结构"""
    
    def __init__(self):
        """初始化优化器，预编译正则表达式"""
        self.patterns = {
            # 1. 页码：单独一行的数字
            'page_number': re.compile(r'^\s*\d+\s*$', re.MULTILINE),
            
            # 2. 图片占位符
            'image': re.compile(r'!\[.*?\]\(.*?\)', re.DOTALL),
            
            # 3. 断裂段落修复
            'broken_paragraph': re.compile(r'([\u4e00-\u9fa5][^。！？：；\n])\n\s*(?=[\u4e00-\u9fa5])'),
            
            # 4. 多余空行：3个及以上
            'excess_newlines': re.compile(r'\n{3,}'),
            
            # 5. 伪标题：加粗文本转标题
            'bold_header': re.compile(r'^\s*\*\*(.*?)\*\*\s*$', re.MULTILINE),
            
            # 6. 数字标题： "1.1 标题" -> "## 1.1 标题"
            'numbered_header': re.compile(r'^\s*(\d+(\.\d+)+)\s+(.{2,20})\s*$', re.MULTILINE),
            
            # 7. 错误列表格式： "1 . 内容" -> "1. 内容"
            'broken_list': re.compile(r'^\s*(\d+)\s+\.\s+', re.MULTILINE),
            
            # 8. 装饰性分隔符
            'decorative': re.compile(r'^[_\-=]{3,}$', re.MULTILINE)
        }
        
        # 默认要移除的关键词（页眉页脚）
        self.default_keywords = [
            "诊断学",
            "第.篇",
            "第.章", 
            "Page",
            "仅供学习交流",
            "扫描全能王",
            "金山OFD",
            "WPS",
            "Microsoft Word",
            "版权声明"
        ]
    
    def optimize(self, text: str, remove_keywords: Optional[List[str]] = None) -> str:
        """
        优化Markdown文档
        
        Args:
            text: 原始Markdown文本
            remove_keywords: 额外要移除的关键词列表
            
        Returns:
            优化后的Markdown文本
        """
        if not text or not text.strip():
            logger.warning("输入文本为空")
            return ""
        
        original_len = len(text)
        
        # 合并关键词列表
        keywords_to_remove = self.default_keywords.copy()
        if remove_keywords:
            keywords_to_remove.extend(remove_keywords)
        
        # 执行优化流程
        text = self._remove_page_numbers(text)
        text = self._remove_headers_footers(text, keywords_to_remove)
        text = self._remove_images(text)
        text = self._fix_broken_paragraphs(text)
        text = self._optimize_structure(text)
        text = self._normalize_whitespace(text)
        
        optimized_len = len(text)
        reduction = original_len - optimized_len
        
        logger.info(f"Markdown优化完成: {original_len} -> {optimized_len} 字符 (减少 {reduction}, {reduction/original_len*100:.1f}%)")
        
        return text
    
    def _remove_page_numbers(self, text: str) -> str:
        """移除页码"""
        return self.patterns['page_number'].sub('', text)
    
    def _remove_headers_footers(self, text: str, keywords: List[str]) -> str:
        """移除页眉页脚"""
        for keyword in keywords:
            # 转义关键词中的特殊字符
            escaped_keyword = re.escape(keyword).replace(r'\.', r'.')
            pattern = fr'^.*{escaped_keyword}.*$'
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text
    
    def _remove_images(self, text: str) -> str:
        """移除图片占位符"""
        return self.patterns['image'].sub('', text)
    
    def _fix_broken_paragraphs(self, text: str) -> str:
        """修复断裂的段落"""
        # 执行两次以处理连续的断行
        text = self.patterns['broken_paragraph'].sub(r'\1', text)
        text = self.patterns['broken_paragraph'].sub(r'\1', text)
        return text
    
    def _optimize_structure(self, text: str) -> str:
        """优化文档结构（RAG专用）"""
        # 将加粗的独立行转换为二级标题（很多PDF转Markdown会把标题识别为加粗文本）
        text = self.patterns['bold_header'].sub(r'## \1', text)
        
        # 将 "1.1 标题" 转换为二级标题
        text = self.patterns['numbered_header'].sub(r'## \1 \3', text)
        
        # 修复错误的列表格式（如："1 . 内容" -> "1. 内容"）
        text = self.patterns['broken_list'].sub(r'\1. ', text)
        
        # 移除装饰性分隔符
        text = self.patterns['decorative'].sub('', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 统一把多个空行变成两个，确保段落清晰
        return self.patterns['excess_newlines'].sub('\n\n', text).strip()

def optimize_markdown_for_rag(text: str, remove_keywords: Optional[List[str]] = None) -> str:
    """
    快捷函数：优化Markdown文档用于RAG检索
    
    Args:
        text: Markdown文本
        remove_keywords: 额外要移除的关键词
    
    Returns:
        优化后的Markdown文本
    """
    optimizer = MarkdownOptimizer()
    return optimizer.optimize(text, remove_keywords)
