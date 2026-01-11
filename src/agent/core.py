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
    """Medical AI Agent implementing ReAct logic with strict evidence-based protocols"""

    def __init__(self, search_tool: SearchTool):
        """
        Initialize medical agent with search capabilities
        
        Args:
            search_tool: Tool for retrieving medical knowledge
        """
        self.search_tool = search_tool

    def _parse_action(self, ai_content: str) -> Tuple[str, str]:
        """
        Parse AI response to extract search action
        
        Args:
            ai_content: Raw AI response content
            
        Returns:
            Tuple of (action_type, keyword) or ("", "") if no action found
        """
        if "检索" not in ai_content:
            return "", ""
            
        # Support multiple formats: "Action: 检索: keyword" or "检索: keyword" or "检索：keyword"
        lines = ai_content.split('\n')
        for line in lines:
            if "检索" in line:
                # Try different splitters
                for splitter in ["检索:", "检索：", "检索 "]:
                    if splitter in line:
                        parts = line.split(splitter, 1)
                        if len(parts) > 1:
                            keyword = parts[1].strip()
                            if keyword:
                                return "search", keyword
                                
        return "", ""

    def _check_final_answer(self, ai_content: str) -> bool:
        """Check if AI response contains final answer marker"""
        return "Final Answer" in ai_content or "最终答案" in ai_content

    def _extract_final_answer(self, ai_content: str) -> str:
        """Extract final answer from AI response"""
        # Try multiple patterns
        patterns = [
            r"Final Answer[:：]\s*(.*)",
            r"最终答案[:：]\s*(.*)",
            r"Answer[:：]\s*(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, ai_content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
                
        # Fallback: return content after marker
        for marker in ["Final Answer", "最终答案", "Answer"]:
            if marker in ai_content:
                parts = ai_content.split(marker, 1)
                if len(parts) > 1:
                    # Remove leading punctuation
                    answer = parts[1].lstrip(":：").strip()
                    if answer:
                        return answer
                        
        return ai_content.strip()

    def run(self, user_input: str, history: List[Dict]) -> Generator[Tuple[str, Any], None, None]:
        """
        Run the ReAct agent loop with medical knowledge retrieval
        
        Args:
            user_input: User's medical query
            history: Conversation history for context
            
        Yields:
            Tuples of (event_type, event_data):
            - ("THOUGHT", str): Agent's reasoning process
            - ("ACTION_START", str): Action being executed (search keyword)
            - ("OBSERVATION", str): Retrieved knowledge
            - ("FINAL_ANSWER", str): Final medical response
        """
        if not user_input or not user_input.strip():
            yield ("FINAL_ANSWER", "请输入有效的医学问题。")
            return

        # Build conversation context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add recent conversation history (last N turns)
        history_start = max(0, len(history) - CONTEXT_HISTORY_TURNS * 2)
        for msg in history[history_start:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_input})

        last_action_keyword = ""
        reasoning_count = 0

        # ReAct reasoning loop
        while reasoning_count < MAX_REASONING_STEPS:
            reasoning_count += 1
            
            try:
                # Get AI response with strict temperature for consistent reasoning
                response = ollama.chat(
                    model=LLM_MODEL,
                    messages=messages,
                    options={'temperature': LLM_TEMPERATURE_STRICT}
                )
                
                ai_content = response.get('message', {}).get('content', "")
                if not ai_content:
                    logger.warning("Empty AI response received")
                    yield ("FINAL_ANSWER", "抱歉，我未能生成有效的回答。请重试或调整问题。")
                    return

                # Emit reasoning thought process
                yield ("THOUGHT", ai_content)
                messages.append({"role": "assistant", "content": ai_content})

                # Parse action from response
                action_type, keyword = self._parse_action(ai_content)
                
                # Execute search action if found
                if action_type == "search" and keyword:
                    # Prevent duplicate searches
                    if keyword == last_action_keyword:
                        obs = "已搜索过该词，无新信息。请基于已有知识尝试总结或换用关键词搜索。"
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": f"Observation: {obs}"})
                    else:
                        yield ("ACTION_START", keyword)
                        
                        # Execute knowledge retrieval
                        search_result = self.search_tool.execute(keyword)
                        
                        obs = f"{search_result}"
                        last_action_keyword = keyword
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": f"Observation: {obs}"})

                    # Continue to next reasoning step
                    continue

                # Check for final answer
                if self._check_final_answer(ai_content):
                    final_answer = self._extract_final_answer(ai_content)
                    yield ("FINAL_ANSWER", final_answer)
                    return

                # Fallback: direct answer if response is detailed enough
                # Increased threshold to prevent premature answers
                if len(ai_content.strip()) > 80:
                    # Ensure it's not just a thought or action declaration
                    if not (ai_content.strip().startswith(("Thought", "Action", "检索", "思考"))):
                        yield ("FINAL_ANSWER", ai_content.strip())
                        return

            except Exception as e:
                logger.error(f"ReAct step {reasoning_count} failed: {e}")
                yield ("FINAL_ANSWER", f"抱歉，分析过程中出现错误: {str(e)}")
                return

        # Maximum reasoning steps reached
        logger.warning(f"ReAct loop exhausted after {MAX_REASONING_STEPS} steps")
        yield ("FINAL_ANSWER", "抱歉，我未能在知识库中找到相关资料，无法得出明确结论。请尝试：\n1. 提供更具体的症状描述\n2. 使用不同的医学术语\n3. 检查相关章节是否已录入知识库")
