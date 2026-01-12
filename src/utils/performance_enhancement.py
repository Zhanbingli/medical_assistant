"""
性能优化和可扩展性增强
"""
from typing import Dict, Any, Optional
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """缓存管理器（支持Redis/Memory）"""
    
    def __init__(self, backend: str = "memory"):
        """
        初始化缓存管理器
        
        Args:
            backend: 后端类型 ("memory" 或 "redis")
        """
        self.backend = backend
        self.cache = {}
        
        if backend == "redis":
            try:
                import redis
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                logger.info("Redis缓存初始化成功")
            except Exception as e:
                logger.warning(f"Redis不可用，降级为内存缓存: {e}")
                self.backend = "memory"
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if self.backend == "redis":
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis获取失败: {e}")
        else:
            return self.cache.get(key)
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        try:
            if self.backend == "redis":
                self.redis_client.setex(
                    key, ttl, json.dumps(value)
                )
            else:
                self.cache[key] = value
            return True
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            if self.backend == "redis":
                self.redis_client.delete(key)
            else:
                if key in self.cache:
                    del self.cache[key]
            return True
        except Exception as e:
            logger.error(f"缓存删除失败: {e}")
            return False
    
    def clear(self) -> bool:
        """清空所有缓存"""
        try:
            if self.backend == "redis":
                self.redis_client.flushdb()
            else:
                self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"缓存清空失败: {e}")
            return False

class AsyncQueryProcessor:
    """异步查询处理器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化异步处理器
        
        Args:
            max_workers: 最大并发数
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"异步处理器初始化，并发数: {max_workers}")
    
    async def process_batch(
        self,
        queries: list,
        process_func: callable
    ) -> list:
        """
        批量异步处理查询
        
        Args:
            queries: 查询列表
            process_func: 处理函数
            
        Returns:
            处理结果列表
        """
        loop = asyncio.get_event_loop()
        
        # 使用线程池执行同步函数
        tasks = [
            loop.run_in_executor(
                self.executor, process_func, query
            ) for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"查询{i}失败: {result}")
                final_results.append(None)
            else:
                final_results.append(result)
        
        return final_results
    
    def shutdown(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)
        logger.info("异步处理器已关闭")

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, endpoints: list):
        """
        初始化负载均衡器
        
        Args:
            endpoints: 服务端点列表
        """
        self.endpoints = endpoints
        self.current_index = 0
        self.response_times = {endpoint: [] for endpoint in endpoints}
    
    def get_next_endpoint(self) -> str:
        """
        获取下一个端点（轮询）
        
        Returns:
            端点URL
        """
        endpoint = self.endpoints[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.endpoints)
        return endpoint
    
    def update_response_time(self, endpoint: str, time_ms: float):
        """
        更新端点响应时间
        
        Args:
            endpoint: 端点URL
            time_ms: 响应时间（毫秒）
        """
        self.response_times[endpoint].append(time_ms)
        # 只保留最近100次
        if len(self.response_times[endpoint]) > 100:
            self.response_times[endpoint] = self.response_times[endpoint][-100:]
    
    def get_best_endpoint(self) -> str:
        """
        获取最佳端点（响应时间最短）
        
        Returns:
            端点URL
        """
        best_endpoint = None
        best_avg_time = float('inf')
        
        for endpoint, times in self.response_times.items():
            if len(times) > 0:
                avg_time = sum(times) / len(times)
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_endpoint = endpoint
        
        return best_endpoint or self.endpoints[0]

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        """初始化监控器"""
        self.metrics = {
            "query_count": 0,
            "query_times": [],
            "embedding_count": 0,
            "embedding_times": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def record_query(self, time_ms: float):
        """记录查询"""
        self.metrics["query_count"] += 1
        self.metrics["query_times"].append(time_ms)
        # 只保留最近1000次
        if len(self.metrics["query_times"]) > 1000:
            self.metrics["query_times"] = self.metrics["query_times"][-1000:]
    
    def record_embedding(self, time_ms: float):
        """记录嵌入生成"""
        self.metrics["embedding_count"] += 1
        self.metrics["embedding_times"].append(time_ms)
        if len(self.metrics["embedding_times"]) > 1000:
            self.metrics["embedding_times"] = self.metrics["embedding_times"][-1000:]
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.metrics["cache_misses"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            统计信息
        """
        query_times = self.metrics["query_times"]
        embedding_times = self.metrics["embedding_times"]
        
        stats = {
            "total_queries": self.metrics["query_count"],
            "total_embeddings": self.metrics["embedding_count"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": 0.0
        }
        
        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            stats["cache_hit_rate"] = (
                self.metrics["cache_hits"] / 
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            ) * 100
        
        if query_times:
            stats["avg_query_time"] = sum(query_times) / len(query_times)
            stats["p95_query_time"] = sorted(query_times)[int(len(query_times) * 0.95)]
            stats["p99_query_time"] = sorted(query_times)[int(len(query_times) * 0.99)]
        
        if embedding_times:
            stats["avg_embedding_time"] = sum(embedding_times) / len(embedding_times)
        
        return stats

class RequestThrottler:
    """请求限流器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        初始化限流器
        
        Args:
            max_requests: 最大请求数
            time_window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def is_allowed(self) -> bool:
        """
        检查是否允许请求
        
        Returns:
            是否允许
        """
        import time
        current_time = time.time()
        
        # 清理过期请求
        self.requests = [
            req_time for req_time in self.requests 
            if current_time - req_time < self.time_window
        ]
        
        # 检查是否超限
        if len(self.requests) >= self.max_requests:
            return False
        
        # 记录当前请求
        self.requests.append(current_time)
        return True
    
    def get_wait_time(self) -> float:
        """
        获取需要等待的时间
        
        Returns:
            等待时间（秒）
        """
        import time
        current_time = time.time()
        
        if len(self.requests) < self.max_requests:
            return 0.0
        
        # 找到最早的那个请求
        oldest_request = min(self.requests)
        wait_time = oldest_request + self.time_window - current_time
        
        return max(0, wait_time)

# 使用示例
if __name__ == "__main__":
    # 1. 缓存管理器
    cache = CacheManager(backend="memory")
    cache.set("test", {"value": 123}, ttl=60)
    result = cache.get("test")
    print(f"缓存结果: {result}")
    
    # 2. 性能监控
    monitor = PerformanceMonitor()
    monitor.record_query(150)
    monitor.record_query(200)
    monitor.record_cache_hit()
    monitor.record_cache_miss()
    
    stats = monitor.get_stats()
    print(f"性能统计: {stats}")
    
    # 3. 请求限流
    throttler = RequestThrottler(max_requests=5, time_window=60)
    for i in range(7):
        if throttler.is_allowed():
            print(f"请求{i}: 允许")
        else:
            wait = throttler.get_wait_time()
            print(f"请求{i}: 超限，等待{wait:.1f}秒")
