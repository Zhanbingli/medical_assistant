import re
import os
import argparse
import logging
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarkdownCleaner:
    """Markdown 文档清洗与结构化工具"""

    def __init__(self, remove_keywords: Optional[List[str]] = None):
        """
        初始化清洗器

        Args:
            remove_keywords: 需要移除的关键词列表（支持正则）
        """
        self.remove_keywords = remove_keywords or []

        # 预编译正则以提升性能
        self.patterns = {
            # 1. 页码：单独一行的数字
            'page_number': re.compile(r'^\s*\d+\s*$', re.MULTILINE),

            # 2. 图片：Markdown 图片占位符
            'image': re.compile(r'!\[.*?\]\(.*?\)', re.DOTALL),

            # 3. 断裂段落：中文后接换行符再接中文
            'broken_paragraph': re.compile(r'([\u4e00-\u9fa5][^。！？：；\n])\n\s*(?=[\u4e00-\u9fa5])'),

            # 4. 多余空行：3个及以上空行
            'excess_newlines': re.compile(r'\n{3,}'),

            # 5. RAG优化 - 伪标题：单独一行的加粗文本 "**标题**" -> "## 标题"
            'bold_header': re.compile(r'^\s*\*\*(.*?)\*\*\s*$', re.MULTILINE),

            # 6. RAG优化 - 数字标题：单独一行的 "1.1 标题" -> "## 1.1 标题"
            # 假设只有少量的字算作标题，避免把列表也转成标题
            'numbered_header': re.compile(r'^\s*(\d+(\.\d+)+)\s+(.{2,20})\s*$', re.MULTILINE),

            # 7. RAG优化 - 列表修复： "1 . 内容" -> "1. 内容"
            'broken_list': re.compile(r'^\s*(\d+)\s+\.\s+', re.MULTILINE),

            # 8. 装饰性字符
            'decorative': re.compile(r'^[_\-=]{3,}$', re.MULTILINE)
        }

    def clean(self, text: str) -> str:
        """执行完整的清洗流程"""
        logger.info("开始文档清洗...")
        original_len = len(text)

        text = self._remove_page_numbers(text)
        text = self._remove_headers_footers(text)
        text = self._remove_images(text)
        text = self._fix_broken_paragraphs(text)
        text = self._optimize_structure(text)
        text = self._normalize_whitespace(text)

        cleaned_len = len(text)
        logger.info(f"清洗完成: {original_len} -> {cleaned_len} 字符 (减少 {original_len - cleaned_len})")
        return text

    def _remove_page_numbers(self, text: str) -> str:
        logger.debug("移除页码...")
        return self.patterns['page_number'].sub('', text)

    def _remove_headers_footers(self, text: str) -> str:
        logger.debug("移除页眉页脚...")
        if not self.remove_keywords:
            return text

        for keyword in self.remove_keywords:
            # (?i) 忽略大小写
            pattern = fr'^.*{keyword}.*$'
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        return text

    def _remove_images(self, text: str) -> str:
        logger.debug("移除图片...")
        return self.patterns['image'].sub('', text)

    def _fix_broken_paragraphs(self, text: str) -> str:
        logger.info("修复断裂段落...")
        # 执行两次以处理连续的断行
        text = self.patterns['broken_paragraph'].sub(r'\1', text)
        text = self.patterns['broken_paragraph'].sub(r'\1', text)
        return text

    def _optimize_structure(self, text: str) -> str:
        logger.info("优化文档结构 (RAG适配)...")

        # 将加粗的独立行转换为二级标题
        # 很多PDF转Markdown会把标题识别为加粗文本
        text = self.patterns['bold_header'].sub(r'## \1', text)

        # 将 "1.1 标题" 转换为二级标题
        text = self.patterns['numbered_header'].sub(r'## \1 \3', text)

        # 修复错误的列表格式
        text = self.patterns['broken_list'].sub(r'\1. ', text)

        # 移除装饰性分隔符
        text = self.patterns['decorative'].sub('', text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        logger.debug("标准化空白...")
        # 统一把多个空行变成两个，确保段落清晰
        return self.patterns['excess_newlines'].sub('\n\n', text).strip()

def main():
    parser = argparse.ArgumentParser(description="Markdown 文档清洗工具 (RAG 优化版)")
    parser.add_argument("input_file", help="输入 Markdown 文件路径")
    parser.add_argument("output_file", help="输出文件路径")
    parser.add_argument("--keywords", nargs="+", help="如果不希望保留某些关键词所在的行（如页眉），请在此列出")
    parser.add_argument("--default-keywords", action="store_true", help="使用预设的常见无关关键词 (推荐)")

    args = parser.parse_args()

    # 准备关键词
    keywords = args.keywords or []
    if args.default_keywords:
        defaults = [
            "诊断学",
            "第.篇",  # 匹配如 "第一篇"
            "第.章",  # 匹配如 "第一章"
            "Page",   # 匹配 "Page 12" 这种
            "仅供学习交流",
            "扫描全能王",
        ]
        keywords.extend(defaults)

    if not os.path.exists(args.input_file):
        logger.error(f"找不到输入文件: {args.input_file}")
        return

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        cleaner = MarkdownCleaner(remove_keywords=keywords)
        cleaned_content = cleaner.clean(content)

        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        logger.info(f"✅ 处理成功！已保存至: {os.path.abspath(args.output_file)}")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main()
