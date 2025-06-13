import os
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class NewsCollector:
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize NewsCollector with Google Custom Search JSON API.
        
        Args:
            api_key: Google Custom Search JSON API key
            search_engine_id: Custom Search Engine ID (cx parameter)
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    def search_news(self, company_name: str, num_results: int = 20) -> List[Dict]:
        """Search for news articles about the company using Google Custom Search."""
        search_queries = [
            f"{company_name} financial news",
            f"{company_name} stock prediction news"  # Reduced to 2 queries as in training notebook
        ]
        
        all_results = []
        results_per_query = min(10, num_results // len(search_queries))  # Max 10 per query (API limit)
        
        for query in search_queries:
            # Add news site restrictions to get better news results
            enhanced_query = f"{query} site:reuters.com OR site:bloomberg.com OR site:cnbc.com OR site:marketwatch.com OR site:yahoo.com OR site:finance.yahoo.com"
            
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": enhanced_query,
                "num": results_per_query,
                "dateRestrict": "m1",  # Results from last month
                "safe": "active",
                "fields": "items(title,snippet,link,displayLink)"  # Only get fields we need
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()
                results = response.json()
                
                if "items" in results:
                    for item in results["items"]:
                        # Convert to format similar to SerpAPI for compatibility
                        article = {
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "link": item.get("link", ""),
                            "source": item.get("displayLink", "")
                        }
                        all_results.append(article)
                        
            except requests.exceptions.RequestException as e:
                print(f"Error searching for '{query}': {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error searching for '{query}': {str(e)}")
                continue
        
        print(f"Collected {len(all_results)} articles for {company_name}")
        return all_results[:num_results]
    
    def extract_article_content(self, url: str) -> str:
        """Extract the main content from a news article URL."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            return content[:2000]  # Limit content length
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return ""
    
    def cluster_and_summarize(self, articles: List[Dict], num_clusters: int = 3) -> List[str]:
        """Cluster articles and select representative ones from each cluster."""
        if not articles:
            return []
            
        # Extract titles and snippets
        texts = []
        for article in articles:
            title = article.get('title', '')
            snippet = article.get('snippet', '')
            if title and snippet:
                texts.append(f"{title}. {snippet}")
            elif title:
                texts.append(title)
        
        if len(texts) < num_clusters:
            return texts
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts)
            
            # Perform clustering
            actual_clusters = min(num_clusters, len(texts))
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Select representative articles from each cluster
            summaries = []
            for cluster_id in range(actual_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # Select the article closest to cluster center
                    cluster_center = kmeans.cluster_centers_[cluster_id]
                    distances = np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1)
                    best_idx = cluster_indices[np.argmin(distances)]
                    summaries.append(texts[best_idx])
            
            return summaries
            
        except Exception as e:
            print(f"Error clustering articles: {str(e)}. Returning first {num_clusters} articles.")
            return texts[:num_clusters]

    def collect_and_process_news(self, company_name: str, num_articles: int = 9) -> Dict:
        """Main method to collect and process news for a company."""
        # Search for news
        news_results = self.search_news(company_name, num_results=60)  # Get more to cluster better
        
        # Extract content for each article (optional - can be CPU intensive)
        # for article in news_results:
        #     if 'link' in article:
        #         article['content'] = self.extract_article_content(article['link'])
        
        # Cluster and summarize to get diverse articles
        summaries = self.cluster_and_summarize(news_results, num_clusters=3)
        
        return {
            'raw_articles': news_results,
            'summaries': summaries,
            'selected_articles': summaries[:num_articles]  # Return top articles for training
        } 