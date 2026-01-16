"""
医疗AI Agent - 核心推理模块
实现严格的循证医学推理和检索
"""
from typing import List, Dict, Generator, Tuple, Any
import ollama
import logging
import re
from config import (
    LLM_MODEL, SYSTEM_PROMPT, MAX_REASONING_STEPS,
    LLM_TEMPERATURE_STRICT, CONTEXT_HISTORY_TURNS,
    MEDGEMMA_SYSTEM_PROMPT, MedGemmaConfig
)
from src.llm import get_adapter
from .tools import SearchTool

logger = logging.getLogger(__name__)


class MedicalAgent:
    """Medical AI Agent - 严格的循证医学推理器"""

    def __init__(self, search_tool: SearchTool):
        """初始化医学Agent"""
        self.search_tool = search_tool

    def _parse_action(self, ai_content: str) -> Tuple[str, str]:
        """
        解析AI响应以提取搜索动作
        支持中英文关键词，适应 MedGemma 输出格式

        Args:
            ai_content: 原始AI响应内容
            
        Returns:
            Tuple of (action_type, keyword) 或 ("", "") 如果没有找到action
        """
        # 定义所有可能的搜索关键词 (中英文)
        search_keywords = ["检索", "search", "Search", "SEARCH", "查找", "查询"]
        
        found_keyword = None
        for kw in search_keywords:
            if kw in ai_content:
                found_keyword = kw
                break
        
        if not found_keyword:
            return "", ""
        
        # 支持的分隔符模式
        splitter_patterns = [
            rf"{re.escape(found_keyword)}[:：]\s*(.+)",  # "检索: 关键词"
            rf"{re.escape(found_keyword)}\s+(.+)",        # "search 关键词"
            rf"{re.escape(found_keyword)}\n(.+)",          # "search\n关键词"
        ]
        
        for pattern in splitter_patterns:
            match = re.search(pattern, ai_content, re.IGNORECASE)
            if match:
                keyword = match.group(1).strip()
                # 清理关键词，移除可能的引用或格式
                keyword = re.sub(r'^["\'\[\]]+|["\'\[\]]+$', '', keyword).strip()
                if keyword and len(keyword) > 1:
                    return "search", keyword
        
        return "", ""

    def _check_final_answer(self, ai_content: str) -> bool:
        """检查是否包含最终答案标记 (中英文)"""
        final_markers = [
            "Final Answer", "最终答案", "final answer", 
            "Final answer", "答案", "Answer", "answer"
        ]
        return any(marker in ai_content for marker in final_markers)

    def _extract_final_answer(self, ai_content: str) -> str:
        """
        从AI响应中提取最终答案
        支持中英文格式

        Args:
            ai_content: AI响应内容

        Returns:
            提取的最终答案文本
        """
        # 尝试多种模式 (中英文)
        patterns = [
            r"Final Answer[:：]?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"最终答案[:：]?\s*(.+?)(?=\n\n|\n|$)",
            r"Answer[:：]?\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            r"答案[:：]?\s*(.+?)(?=\n\n|\n|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ai_content, re.IGNORECASE | re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer and len(answer.strip()) > 10:
                    # 清理答案
                    answer = re.sub(r'^["\'\[\]]+|["\'\[\]]+$', '', answer).strip()
                    return answer
        
        # 回退：使用最终答案标记后的内容
        fallback_markers = ["Final Answer", "最终答案", "Answer", "答案"]
        for marker in fallback_markers:
            if marker in ai_content:
                parts = ai_content.split(marker, 1)
                if len(parts) > 1:
                    answer = parts[1].lstrip(":：: ").strip()
                    if answer:
                        return answer

        return ai_content.strip()

    def run(self, user_input: str, history: List[Dict]) -> Generator[Tuple[str, Any], None, None]:
        """
        运行ReAct推理循环

        Args:
            user_input: 用户医学查询
            history: 对话历史

        Yields:
            事件类型和数据的元组
            ("THOUGHT", str): Agent的推理过程
            ("ACTION_START", str): 正在执行的动作（搜索关键词）
            ("OBSERVATION", str): 检索到的知识
            ("FINAL_ANSWER", str): 最终医学回答
        """
        if not user_input or not user_input.strip():
            yield ("FINAL_ANSWER", "请输入有效的医学问题。")
            return

        # 构建对话上下文
        messages = [{"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT}]

        # 添加最近N轮对话历史
        history_start = max(0, len(history) - CONTEXT_HISTORY_TURNS * 2)
        for msg in history[history_start:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})

        last_action_keyword = ""
        reasoning_count = 0
        final_answer_yielded = False
        last_observation = ""

        # ReAct推理循环
        while reasoning_count < MAX_REASONING_STEPS:
            reasoning_count += 1

            try:
                # 调用AI模型（严格模式）
                adapter = get_adapter()
                response = adapter.chat(
                    messages=messages,
                    temperature=MedGemmaConfig.TEMPERATURE_STRICT
                )

                ai_content = response.get('message', {}).get('content', "")
                if not ai_content:
                    logger.warning("收到空AI响应")
                    if not final_answer_yielded:
                        yield ("FINAL_ANSWER", "抱歉，我未能生成有效的回答。请重试或调整问题。")
                        final_answer_yielded = True
                        return

                # 1. 发送思维过程
                yield ("THOUGHT", ai_content)
                messages.append({"role": "assistant", "content": ai_content})

                # 2. 解析动作
                action_type, keyword = self._parse_action(ai_content)

                # 3. 执行搜索动作
                if action_type == "search" and keyword:
                    # 防止重复搜索
                    if keyword == last_action_keyword:
                        obs = f"已搜索过关键词【{keyword}】，没有新信息。请尝试基于已有知识回答或更换关键词搜索。上一次检索结果：{last_observation}"
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": f"Observation: {obs}"})
                    else:
                        yield ("ACTION_START", keyword)

                        # 执行知识检索
                        search_result = self.search_tool.execute(keyword)
                        last_observation = search_result
                        last_action_keyword = keyword

                        obs = f"Observation: {search_result}"
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": f"Observation: {obs}"})

                    # 继续下一步推理（不返回最终答案）
                    continue

                # 4. 检查是否有最终答案
                if self._check_final_answer(ai_content):
                    final_answer = self._extract_final_answer(ai_content)
                    if final_answer and len(final_answer.strip()) > 20:
                        yield ("FINAL_ANSWER", final_answer)
                        final_answer_yielded = True
                        return

                # 5. 回退：如果响应足够长但没有明确标记
                if len(ai_content.strip()) > 100:
                    # 清理掉 Thought 和 Action 前缀 (中英文)
                    skip_prefixes = [
                        "Thought:", "思考:", "Action:", "行动:",
                        "检索:", "search:", "Search:",
                        "Observation:", "观察:", "obs:",
                        "<thought>", "<action>", "<search>"
                    ]
                    
                    cleaned_lines = []
                    for line in ai_content.split('\n'):
                        skip = False
                        line_stripped = line.strip()
                        for prefix in skip_prefixes:
                            if line_stripped.startswith(prefix):
                                skip = True
                                break
                        if not skip and line_stripped:
                            cleaned_lines.append(line)
                    
                    cleaned_answer = '\n'.join(cleaned_lines).strip()
                    if cleaned_answer:
                        yield ("FINAL_ANSWER", cleaned_answer)
                        final_answer_yielded = True
                        return

            except Exception as e:
                logger.error(f"ReAct第{reasoning_count}步出错: {e}")
                if not final_answer_yielded:
                    yield ("FINAL_ANSWER", f"抱歉，分析过程中出现错误：{str(e)}")
                    final_answer_yielded = True

        # 达到最大推理步数
        logger.warning(f"ReAct循环已达到最大步数（{MAX_REASONING_STEPS}步）")
        yield ("FINAL_ANSWER", "抱歉，我未能在知识库中找到相关资料，无法得出明确结论。建议：\n1. 提供更具体的症状描述\n2. 使用专业医学术语\n3. 检查相关章节是否已录入知识库")
        final_answer_yielded = True
