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
    LLM_TEMPERATURE_STRICT, CONTEXT_HISTORY_TURNS
)
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

        Args:
            ai_content: 原始AI响应内容
            
        Returns:
            Tuple of (action_type, keyword) 或 ("", "") 如果没有找到action
        """
        if "检索" not in ai_content:
            return "", ""
            
        # 支持多种格式
        lines = ai_content.split('\n')
        for line in lines:
            if "检索" in line:
                # 尝试不同的分隔符
                for splitter in ["检索:", "检索：", "检索 "]:
                    if splitter in line:
                        parts = line.split(splitter, 1)
                        if len(parts) > 1:
                            keyword = parts[1].strip()
                            if keyword:
                                return "search", keyword
        return "", ""

    def _check_final_answer(self, ai_content: str) -> bool:
        """检查是否包含最终答案标记"""
        return "Final Answer" in ai_content or "最终答案" in ai_content

    def _extract_final_answer(self, ai_content: str) -> str:
        """
        从AI响应中提取最终答案

        Args:
            ai_content: AI响应内容

        Returns:
            提取的最终答案文本
        """
        # 尝试多种模式
        patterns = [
            r"Final Answer[:：]\s*(.*)",
            r"最终答案[:：]\s*(.*)",
            r"Answer[:：]\s*(.*"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ai_content, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        # 回退：使用"Final Answer"或"最终答案"后的内容
        for marker in ["Final Answer", "最终答案", "Answer"]:
            if marker in ai_content:
                parts = ai_content.split(marker, 1)
                if len(parts) > 1:
                    # 移除开头的标点
                    answer = parts[1].lstrip(":：:").strip()
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
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages,
                    options={'temperature': LLM_TEMPERATURE_STRICT}
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
                    # 清理掉Thought和Action前缀
                    cleaned_lines = []
                    skip_prefixes = ["Thought:", "思考:", "Action:", "检索:", "检索：", "检索", "Observation:", "Observation:"]
                    
                    for line in ai_content.split('\n'):
                        skip = False
                        for prefix in skip_prefixes:
                            if line.strip().startswith(prefix):
                                skip = True
                                break
                        if not skip and line.strip():
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
