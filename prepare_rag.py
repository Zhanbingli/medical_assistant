#!/usr/bin/env python
"""
诊断学_cleaned.md 预处理脚本 - 简单有效版
直接提取干净的医学内容
"""

import re

INPUT_FILE = './诊断学_cleaned.md'
OUTPUT_FILE = './诊断学_rag.md'


def main():
    print("="*60)
    print("清理诊断学文件 - RAG优化版")
    print("="*60)
    
    # 读取文件
    print(f"\n读取: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 方法：直接处理整个内容
    lines = content.split('\n')
    
    # 找到第一篇内容的实际开始位置
    start_idx = None
    for i, line in enumerate(lines):
        if '症状（symptom）是指病人主观' in line:
            start_idx = i
            break
    
    if start_idx is None:
        start_idx = 1650
    
    print(f"开始行: {start_idx + 1}")
    
    # 处理每一行
    output_lines = []
    skip_block = False
    last_part = ""
    
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        
        # 跳过数字资源/目标测试块
        if any(x in line for x in ['本章数字资源', '本节数字资源', '本章目标测试', '本节目标测试', '本章配套数字资源']):
            skip_block = True
            continue
        
        # 检测新篇章开始
        part_match = re.match(r'^(第一篇|第二篇|第三篇|第四篇|第五篇|第六篇|第七篇|第八篇)\s', line)
        if part_match:
            current_part = part_match.group(1)
            if current_part != last_part:
                output_lines.append('\n' + '='*60)
                output_lines.append(current_part)
                output_lines.append('='*60 + '\n')
                last_part = current_part
            skip_block = False
            continue
        
        # 退出跳过模式
        if skip_block and (line.startswith('第') or len(line) > 100):
            skip_block = False
        
        if skip_block:
            continue
        
        # 清理行
        # 移除TOC标记
        line = re.sub(r'\[\s*[.\s]*\d+\s*\]', '', line)
        line = re.sub(r'\[\s*[.\s]+\s*\]', '', line)
        line = re.sub(r'\*\*\d+\*\*', '', line)
        line = re.sub(r'\s+\d+\s*$', '', line)
        line = re.sub(r'\s+\d+\s*$', '', line)  # again
        
        # 跳过空行和纯数字
        if not line:
            if output_lines and output_lines[-1].strip():
                output_lines.append('')
        elif len(line) < 5 or line.isdigit():
            continue
        else:
            output_lines.append(line)
    
    # 合并
    result = '\n'.join(output_lines)
    result = re.sub(r'\n{5,}', '\n\n\n', result)
    
    # 写入
    print(f"\n写入: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"大小: {len(result)} 字符")
    
    # 统计
    print("\n" + "-"*60)
    parts = re.findall(r'(第一篇|第二篇|第三篇|第四篇|第五篇|第六篇|第七篇|第八篇)[^\n]*', result)
    print(f"篇章: {list(dict.fromkeys(parts))}")
    
    print("\n关键词:")
    for kw in ['发热', '问诊', '体格检查', '诊断']:
        print(f"  {kw}: {result.count(kw)}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    main()
