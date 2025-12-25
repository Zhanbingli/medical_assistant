import pymupdf4llm
import time
import os
import argparse
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def convert_pdf_to_md(input_path: str, output_path: str = None, write_images: bool = False):
    """
    将 PDF 文件转换为 Markdown 文件。

    Args:
        input_path (str): 输入 PDF 文件的路径。
        output_path (str, optional): 输出 Markdown 文件的路径。如果未提供，将使用输入文件名。
        write_images (bool, optional): 是否提取图片。默认为 False。
    """
    input_file = Path(input_path)

    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return

    if output_path:
        output_file = Path(output_path)
    else:
        output_file = input_file.with_suffix(".md")
        # 如果是同名文件（例如本来就是markdown），避免覆盖源文件，加个后缀
        if output_file == input_file:
             output_file = input_file.with_stem(input_file.stem + "_converted").with_suffix(".md")

    logger.info(f"🚀 开始转换: {input_path} -> {output_file}")
    start_time = time.time()

    try:
        # 核心转换代码
        md_text = pymupdf4llm.to_markdown(input_path, write_images=write_images)

        # 保存文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_text)

        end_time = time.time()
        logger.info(f"✅ 转换成功！")
        logger.info(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
        logger.info(f"📂 输出文件: {output_file.absolute()}")

    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        # 在某些严重错误下可能需要抛出或者退出，这里仅记录日志

def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF 转 Markdown 工具")
    parser.add_argument("input", help="输入的 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出的 Markdown 文件路径 (可选)")
    parser.add_argument("--images", action="store_true", help="是否提取图片 (默认不提取)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    convert_pdf_to_md(args.input, args.output, args.images)
