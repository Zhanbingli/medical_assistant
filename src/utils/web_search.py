"""
联网搜索模块
当知识库为空或知识过时时，使用网络搜索补充信息
"""
import requests
from typing import List, Dict, Any, Optional
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float = 0.0

class WebSearchEngine:
    """网络搜索引擎基类"""
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        raise NotImplementedError

class DuckDuckGoSearch(WebSearchEngine):
    """DuckDuckGo搜索引擎（免费，无需API Key）"""
    
    def __init__(self):
        """初始化DuckDuckGo搜索"""
        self.base_url = "https://duckduckgo.com/html/"
        self.api_url = "https://api.duckduckgo.com/"
    
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """使用DuckDuckGo搜索"""
        try:
            # 使用Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 0
            }
            
            response = requests.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"DuckDuckGo搜索失败: {response.status_code}")
                return []
            
            data = response.json()
            results = []
            
            # 提取相关主题
            related_topics = data.get('RelatedTopics', [])
            for topic in related_topics[:num_results]:
                if isinstance(topic, dict) and 'Text' in topic and 'FirstURL' in topic:
                    results.append(SearchResult(
                        title=topic.get('Text', ''),
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        source="DuckDuckGo",
                        relevance_score=1.0 - results.__len__() * 0.1
                    ))
            
            logger.info(f"DuckDuckGo搜索完成: {len(results)}条结果")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo搜索异常: {e}")
            return []

class WikipediaSearch(WebSearchEngine):
    """维基百科搜索（适合医学知识）"""
    
    def __init__(self, language: str = "zh"):
        """
        初始化维基百科搜索
        
        Args:
            language: 语言代码 (zh=中文, en=英文）
        """
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
    
    def search(self, query: str, num_results: int = 3) -> List[SearchResult]:
        """使用维基百科搜索"""
        try:
            # 搜索API
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'utf8': '',
                'format': 'json',
                'srlimit': num_results
            }
            
            response = requests.get(self.api_url, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"维基百科搜索失败: {response.status_code}")
                return []
            
            data = response.json()
            results = []
            
            search_results = data.get('query', {}).get('search', [])
            for item in search_results:
                # 获取页面摘要
                page_params = {
                    'action': 'query',
                    'prop': 'extracts',
                    'exintro': True,
                    'explaintext': True,
                    'format': 'json',
                    'titles': item.get('title'),
                    'redirects': True
                }
                
                page_response = requests.get(self.api_url, params=page_params, timeout=5)
                page_data = page_response.json()
                
                page_info = page_data.get('query', {}).get('pages', {})
                if page_info:
                    page_id = list(page_info.keys())[0]
                    extract = page_info[page_id].get('extract', '')
                    
                    results.append(SearchResult(
                        title=item.get('title', ''),
                        url=f"https://{self.language}.wikipedia.org/wiki/{item['title']}",
                        snippet=extract[:200] + "..." if len(extract) > 200 else extract,
                        source="Wikipedia",
                        relevance_score=1.0 - results.__len__() * 0.1
                    ))
            
            logger.info(f"维基百科搜索完成: {len(results)}条结果")
            return results
            
        except Exception as e:
            logger.error(f"维基百科搜索异常: {e}")
            return []

class PubMedSearch(WebSearchEngine):
    """PubMed搜索（医学专业文献）"""
    
    def __init__(self):
        """初始化PubMed搜索"""
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def search(self, query: str, num_results: int = 3) -> List[SearchResult]:
        """使用PubMed搜索"""
        try:
            # 搜索文献
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': num_results
            }
            
            search_response = requests.get(
                self.base_url + "esearch.fcgi",
                params=search_params,
                timeout=10
            )
            
            if search_response.status_code != 200:
                logger.error(f"PubMed搜索失败: {search_response.status_code}")
                return []
            
            search_data = search_response.json()
            id_list = search_data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                return []
            
            # 获取摘要
            summary_params = {
                'db': 'pubmed',
                'id': ','.join(id_list),
                'retmode': 'json',
                'rettype': 'abstract'
            }
            
            summary_response = requests.get(
                self.base_url + "efetch.fcgi",
                params=summary_params,
                timeout=10
            )
            
            summary_data = summary_response.json()
            results = []
            
            articles = summary_data.get('result', {})
            if isinstance(articles, dict) and 'uids' in articles:
                for uid in articles['uids']:
                    article = articles[uid]
                    title = article.get('title', '')
                    abstract = article.get('abstract', '')
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=abstract[:300] + "..." if len(abstract) > 300 else abstract,
                        source="PubMed",
                        relevance_score=1.0 - results.__len__() * 0.1
                    ))
            
            logger.info(f"PubMed搜索完成: {len(results)}条结果")
            return results
            
        except Exception as e:
            logger.error(f"PubMed搜索异常: {e}")
            return []

class HybridWebSearch:
    """混合网络搜索引擎（多个来源）"""
    
    def __init__(self):
        """初始化混合搜索引擎"""
        self.engines = [
            WikipediaSearch(language="zh"),
            DuckDuckGoSearch(),
            # PubMedSearch()  # 可选：需要PubMed API Key
        ]
    
    def search(
        self,
        query: str,
        num_results: int = 5,
        engines: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        执行混合搜索
        
        Args:
            query: 搜索查询
            num_results: 每个引擎返回的结果数
            engines: 使用的引擎列表（None表示全部）
            
        Returns:
            合并的搜索结果列表
        """
        all_results = []
        
        for engine in self.engines:
            if engines and engine.__class__.__name__ not in engines:
                continue
            
            try:
                results = engine.search(query, num_results)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"{engine.__class__.__name__}搜索失败: {e}")
        
        # 去重
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # 按相关性排序
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return unique_results[:num_results * 2]

class SearchStrategy:
    """搜索策略决策器"""
    
    def __init__(self, db):
        """
        初始化搜索策略器
        
        Args:
            db: 数据库实例
        """
        self.db = db
        self.web_search = HybridWebSearch()
    
    def decide_search_strategy(self, query: str) -> Dict[str, Any]:
        """
        决定搜索策略
        
        Args:
            query: 用户查询
            
        Returns:
            策略信息字典
        """
        stats = self.db.get_collection_stats()
        total_chunks = stats.get('total_chunks', 0)
        total_files = stats.get('total_files', 0)
        
        strategy = {
            "use_local": True,
            "use_web": False,
            "reason": "",
            "web_sources": []
        }
        
        # 策略1：知识库为空 → 仅网络搜索
        if total_chunks == 0 or total_files == 0:
            strategy["use_local"] = False
            strategy["use_web"] = True
            strategy["reason"] = "知识库为空，使用网络搜索"
            strategy["web_sources"] = ["Wikipedia", "DuckDuckGo"]
            return strategy
        
        # 策略2：知识库内容过少 → 本地优先，网络补充
        if total_chunks < 100:
            strategy["use_local"] = True
            strategy["use_web"] = True
            strategy["reason"] = f"知识库内容较少（{total_chunks}条），使用本地+网络混合搜索"
            strategy["web_sources"] = ["Wikipedia", "DuckDuckGo"]
            return strategy
        
        # 策略3：知识过时检查（基于文件名中的年份）
        files = stats.get('files', [])
        current_year = time.localtime().tm_year
        
        old_files = 0
        for file in files:
            # 查找文件名中的年份（如：2023年诊断学）
            import re
            year_match = re.search(r'20\d{2}', file)
            if year_match:
                year = int(year_match.group())
                if current_year - year > 5:
                    old_files += 1
        
        if old_files > 0 and old_files / len(files) > 0.3:
            # 超过30%的文件超过5年
            strategy["use_local"] = True
            strategy["use_web"] = True
            strategy["reason"] = f"部分知识可能过时（{old_files}/{len(files)}个文件>5年），建议结合网络搜索"
            strategy["web_sources"] = ["Wikipedia", "PubMed", "DuckDuckGo"]
            return strategy
        
        # 默认：仅本地搜索
        strategy["reason"] = "知识库充足，使用本地搜索"
        return strategy

# 使用示例
if __name__ == "__main__":
    # 1. DuckDuckGo搜索
    ddg = DuckDuckGoSearch()
    results = ddg.search("发热伴咳嗽", num_results=3)
    print("DuckDuckGo结果:")
    for r in results:
        print(f"- {r.title}: {r.snippet}")
    
    # 2. 维基百科搜索
    wiki = WikipediaSearch(language="zh")
    results = wiki.search("肺炎", num_results=2)
    print("\n维基百科结果:")
    for r in results:
        print(f"- {r.title}: {r.snippet}")
    
    # 3. 混合搜索
    hybrid = HybridWebSearch()
    results = hybrid.search("糖尿病", num_results=3)
    print("\n混合搜索结果:")
    for r in results:
        print(f"[{r.source}] {r.title}: {r.snippet[:100]}")
