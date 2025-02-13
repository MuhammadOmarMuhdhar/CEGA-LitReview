import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Implements token bucket algorithm for rate limiting."""
    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = datetime.now()
        self.window = deque(maxlen=requests_per_second)
    
    def acquire(self):
        """Wait if necessary and acquire a token."""
        now = datetime.now()
        
        # Remove timestamps older than 1 second
        while self.window and (now - self.window[0]) > timedelta(seconds=1):
            self.window.popleft()
        
        # If we've made too many requests in the last second, wait
        if len(self.window) >= self.requests_per_second:
            sleep_time = 1 - (now - self.window[0]).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = datetime.now()
        
        self.window.append(now)

def create_session() -> requests.Session:
    """Creates a requests session with retry logic and timeouts."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,  # Increased backoff factor
        status_forcelist=[429, 500, 502, 503, 504],  # Added 429 for rate limits
        respect_retry_after_header=True  # Honor server's retry-after header
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    
    # Add default headers
    session.headers.update({
        'User-Agent': 'YourApp/1.0 (mailto:pepdataviz@gmail.com)'
    })
    return session

def make_request(session: requests.Session, url: str, params: dict, rate_limiter: RateLimiter) -> dict:
    """Makes a rate-limited request to OpenAlex API."""
    rate_limiter.acquire()
    
    try:
        response = session.get(url, params=params, timeout=30)
        
        # Handle rate limiting explicitly
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            return make_request(session, url, params, rate_limiter)
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise

def get_dois(works: List[str], session: requests.Session, rate_limiter: RateLimiter) -> List[str]:
    """Extracts DOIs from referenced works using batch requests."""
    if not works:
        return []

    dois = []
    base_url = "https://api.openalex.org/works"

    # Process works in batches of 50 (reduced from 200 for better rate limiting)
    for i in range(0, len(works), 50):
        batch = works[i:i+50]
        work_ids = [w.split('/')[-1] for w in batch]
        filter_str = f"openalex:{('|').join(work_ids)}"
        
        try:
            url = f"{base_url}?filter={filter_str}"
            data = make_request(session, url, {}, rate_limiter)
            
            batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
            clean_dois = [doi.strip('https://doi.org/') for doi in batch_dois if doi]
            dois.extend(clean_dois)
                
        except requests.RequestException as e:
            logger.error(f"Error fetching batch: {e}")
            
    return dois

def clean_abstract(abstract_index: Union[Dict, str]) -> str:
    """
    Reconstructs an abstract from OpenAlex's inverted index format into readable text.
    Args:
        abstract_index (dict or str): Either a dictionary containing the inverted index
            format from OpenAlex (word -> positions) or a plain text string.
    Returns:
        str: The reconstructed abstract text with words in correct order.
    """
    if not isinstance(abstract_index, dict):
        return abstract_index
    # Use generator expression for memory efficiency
    word_positions = ((pos, word) 
                     for word, positions in abstract_index.items() 
                     if isinstance(positions, list)
                     for pos in positions)
    return ' '.join(word for _, word in sorted(word_positions, key=lambda x: x[0]))

def extract_papers(broad_field: str, keyword: str = "Psychology", limit: Optional[int] = None) -> List[Dict]:
    """Extracts paper information from OpenAlex API with improved rate limiting."""
    if keyword is not None and not isinstance(keyword, str):
        raise ValueError("Keyword must be a string")
    
    if limit is not None and not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    
    base_url = "https://api.openalex.org/works"
    params = {
        "mailto": "pepdataviz@gmail.com", 
        "search": keyword,
        "cursor": "*",
        "per_page": min(50, limit) if limit else 50, 
        "filter": "type:article"
    }

    papers = []
    total_fetched = 0
    session = create_session()
    rate_limiter = RateLimiter(requests_per_second=10)  # OpenAlex allows ~10 req/second

    try:
        with ThreadPoolExecutor(max_workers=3) as executor: 
            while True:
                try:
                    data = make_request(session, base_url, params, rate_limiter)
                    
                    futures = []
                    current_papers = []
                    
                    for work in data.get('results', []):
                        if limit and total_fetched >= limit:
                            return papers
                        
                        citing_works = work.get('cited_by_api_url', '')
                        citing_works_stripped = [x.strip() for x in 
                            citing_works.replace('works?filter=cites:', '').split(',')
                            if x.strip()]
                        referenced_works = work.get('referenced_works', [])
                        
                        future_citing = executor.submit(get_dois, citing_works_stripped, session, rate_limiter)
                        future_referenced = executor.submit(get_dois, referenced_works, session, rate_limiter)
                        futures.extend([future_citing, future_referenced])
                        
                        paper = {
                            'doi': work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None,
                            'title': work.get('display_name', None),
                            'link': work.get('link', None),
                            'authors': [auth['author']['display_name'] for auth in work.get('authorships', [])],
                            "keyword": keyword,
                            'publication': 'Openalex',
                            'country': [auth['institutions'][0]['country_code'] if auth.get('institutions') else None 
                                      for auth in work.get('authorships', [])],
                            'date': work.get('publication_year', None),
                            'field': broad_field,
                            'institution': [auth['institutions'][0]['display_name'] if auth.get('institutions') else None 
                                          for auth in work.get('authorships', [])],
                            'abstract': clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),
                            'cited_by_count': work.get('cited_by_count', 0),
                            'citing_works': None,
                            'referenced_works': None
                        }
                        current_papers.append(paper)
                        total_fetched += 1
                    
                    for i, paper in enumerate(current_papers):
                        paper['citing_works'] = futures[i*2].result()
                        paper['referenced_works'] = futures[i*2+1].result()
                        papers.append(paper)
                    
                    next_cursor = data.get('meta', {}).get('next_cursor')
                    if not next_cursor:
                        break
                    
                    params['cursor'] = next_cursor
                    
                except requests.RequestException as e:
                    logger.error(f"Error fetching data: {e}")
                    continue
                
    finally:
        session.close()
    
    return papers