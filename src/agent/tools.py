from typing import Protocol, List, Tuple
from abc import abstractmethod

class Tool(Protocol):
    """Abstract base class for agent tools"""
    name: str
    description: str

    @abstractmethod
    def execute(self, query: str) -> str:
        pass

class SearchTool:
    """Wrapper for MedicalSearchEngine"""
    name = "检索"
    description = "用于查阅医学知识库，获取专业资料。"

    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.last_logs = []

    def execute(self, query: str) -> str:
        """Execute search and return result string"""
        result, logs = self.search_engine.search(query, debug=True)
        self.last_logs = logs  # Store logs for UI if needed
        return result
