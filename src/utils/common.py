"""通用工具模块"""
from typing import Any, Callable, TypeVar, ParamSpec
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')
P = ParamSpec('P')

def singleton(cls: T) -> T:
    """单例装饰器"""
    instances = {}
    
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    """失败重试装饰器"""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator
