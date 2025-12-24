import pymupdf4llm
import time
import os

# ================= 配置区域 =================
# 把这里的名字改成你真实的 PDF 文件名
pdf_filename = "诊断学.pdf"
# 输出的文件名
md_filename = "诊断学_cleaned.md"
# ===========================================

def convert_pdf_to_md(input_path, output_path):
    print(f"🚀 开始转换: {input_path} ...")
    start_time = time.time()

    try:
        # 核心转换代码：to_markdown 会自动处理表格和文字
        # write_images=False 表示暂时不提取图片，专注文字，保持纯净
        md_text = pymupdf4llm.to_markdown(input_path, write_images=False)

        # 保存文件
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        end_time = time.time()
        print(f"✅ 转换成功！")
        print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
        print(f"📂 输出文件: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    convert_pdf_to_md(pdf_filename, md_filename)
