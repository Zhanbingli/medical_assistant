from typing import List, Dict, Generator, Tuple, Any
import ollama
import logging
from config import (
    LLM_MODEL, SYSTEM_PROMPT, MAX_REASONING_STEPS,
    LLM_TEMPERATURE_STRICT, CONTEXT_HISTORY_TURNS
)
from .tools import SearchTool

logger = logging.getLogger(__name__)

class MedicalAgent:
    """Medical AI Agent implementing ReAct logic"""

    def __init__(self, search_tool: SearchTool):
        self.search_tool = search_tool

    def run(self, user_input: str, history: List[Dict]) -> Generator[Tuple[str, Any], None, None]:
        """
        Run the agent loop.
        Yields: (event_type, event_data)
        event_type: 'THOUGHT', 'ACTION_START', 'OBSERVATION', 'FINAL_ANSWER'
        """
        # 1. Build Context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add history (last N turns)
        # Simplified: history should only contain user/assistant messages
        history_start = max(0, len(history) - CONTEXT_HISTORY_TURNS * 2)
        for msg in history[history_start:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        last_action = ""

        # 2. ReAct Loop
        for step in range(MAX_REASONING_STEPS):
            response = ollama.chat(
                model=LLM_MODEL,
                messages=messages,
                options={'temperature': LLM_TEMPERATURE_STRICT}
            )
            ai_content = response['message']['content']

            # Emit thought (Cleaned up for UI)
            yield ("THOUGHT", ai_content)

            messages.append(response['message'])

            # 1. First Priority: Detect Retrieval Action
            # Support multiple formats: "Action: 检索: keyword" or just "检索: keyword"
            if "检索" in ai_content:
                # Extract keyword logic
                keyword = ""
                lines = ai_content.split('\n')
                for line in lines:
                    if "检索" in line and (":" in line or "：" in line):
                        # Extract part after the colon
                        splitter = "检索:" if "检索:" in line else "检索："
                        parts = line.split(splitter)
                        if len(parts) > 1:
                            keyword = parts[-1].strip()
                            break

                # If a valid keyword is found, execute search immediately
                if keyword:
                    if keyword == last_action:
                        obs = "Observation: 已搜索过该词，无新信息。请尝试总结。"
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": obs})
                    else:
                        yield ("ACTION_START", keyword)

                        # Execute Search
                        result = self.search_tool.execute(keyword)

                        obs = f"Observation: {result}"
                        last_action = keyword
                        yield ("OBSERVATION", obs)
                        messages.append({"role": "user", "content": obs})

                    # Continue to next reasoning step, do NOT return Final Answer yet
                    continue

            # 2. Second Priority: Check for Final Answer (Only if no search is triggered)
            if "Final Answer" in ai_content:
                final_answer = ai_content.split("Final Answer")[-1].lstrip(":").lstrip("：").strip()
                yield ("FINAL_ANSWER", final_answer)
                return

            # 3. Fallback: Direct answer if long enough and no retrieval
            if len(ai_content) > 50: # Increased threshold for safety
                yield ("FINAL_ANSWER", ai_content)
                return

        # Loop exhausted
        yield ("FINAL_ANSWER", "抱歉，我未查到相关资料，未能得出明确结论。")
