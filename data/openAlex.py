import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
from collections import deque

class Scraper:
    """
    A class for extracting paper information from the OpenAlex API with rate limiting
    and retry logic.
    """
    
    def __init__(self, api_key = None, requests_per_second: int = 10, email: str = "pepdataviz@gmail.com"):
        """
        Initialize the OpenAlexExtractor.
        
        Args:
            requests_per_second: Maximum number of requests per second (default: 10)
            email: Email for the 'mailto' parameter (default: "pepdataviz@gmail.com")
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        
        # Create session with retry logic
        self.session = self._create_session(email, api_key)
        
        # Base URL for OpenAlex API
        self.base_url = "https://api.openalex.org/works"
        
        # Store email for API requests
        self.email = email

        # Store API key for authentication
        self.api_key = api_key

        # Initialize rate limiter
        if api_key is None:
            self.rate_limiter = self._create_rate_limiter(requests_per_second)
        else:
            self.rate_limiter = self._create_rate_limiter(100)  # Create proper rate limiter structure

    
    def _create_rate_limiter(self, requests_per_second: int):
        """
        Creates a rate limiter using token bucket algorithm.
        
        Args:
            requests_per_second: Maximum number of requests per second
            
        Returns:
            A dictionary containing rate limiter state
        """
        return {
            'requests_per_second': requests_per_second,
            'tokens': requests_per_second,
            'last_update': datetime.now(),
            'window': deque(maxlen=requests_per_second)
        }
    
    def _acquire_token(self):
        """Wait if necessary and acquire a token."""
        now = datetime.now()
        
        # Remove timestamps older than 1 second
        while self.rate_limiter['window'] and (now - self.rate_limiter['window'][0]) > timedelta(seconds=1):
            self.rate_limiter['window'].popleft()
        
        # If we've made too many requests in the last second, wait
        if len(self.rate_limiter['window']) >= self.rate_limiter['requests_per_second']:
            sleep_time = 2 - (now - self.rate_limiter['window'][0]).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = datetime.now()
        
        self.rate_limiter['window'].append(now)
    
    def _create_session(self, email: str, api_key: str ) -> requests.Session:
        """
        Creates a requests session with retry logic and timeouts.
        
        Args:
            email: Email to use in the User-Agent header
            
        Returns:
            Configured requests.Session object
        """
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
            'User-Agent': f'YourApp/1.0 (mailto:{email})',
            'api_key': api_key
        })
        return session
    
    def _make_request(self, url: str, params: dict) -> dict:
        """
        Makes a rate-limited request to OpenAlex API.
        
        Args:
            url: API endpoint URL
            params: Query parameters for the request
            
        Returns:
            JSON response as a dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        self._acquire_token()
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            # Handle rate limiting explicitly
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                self.logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self._make_request(url, params)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    def _get_dois(self, works: List[str]) -> List[str]:
        """
        Extracts DOIs from referenced works using batch requests.
        
        Args:
            works: List of work IDs or URL with filter query
            
        Returns:
            List of DOIs
        """
        if not works:
            return []

        dois = []

        # If works is a URL, extract the filter query
        if isinstance(works, str) and works.startswith('https://api.openalex.org/works?filter=cites:'):
            filter_str = works.split('filter=')[-1]
            try:
                data = self._make_request(self.base_url, {'filter': filter_str})
                batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
                clean_dois = [doi.strip('https://doi.org/') for doi in batch_dois if doi]
                dois.extend(clean_dois)
            except requests.RequestException as e:
                self.logger.error(f"Error fetching citing works: {e}")
        else:
            for i in range(0, len(works), 100):
                batch = works[i:i+100]
                work_ids = [w.split('/')[-1] for w in batch]
                filter_str = f"openalex:{('|').join(work_ids)}"
                
                try:
                    url = f"{self.base_url}?filter={filter_str}"
                    data = self._make_request(url, {})
                    
                    batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
                    clean_dois = [doi.strip('https://doi.org/') for doi in batch_dois if doi]
                    dois.extend(clean_dois)
                        
                except requests.RequestException as e:
                    self.logger.error(f"Error fetching batch: {e}")
                    
        return dois
    
    def _clean_abstract(self, abstract_index: Union[Dict, str]) -> str:
        """
        Reconstructs an abstract from OpenAlex's inverted index format into readable text.
        
        Args:
            abstract_index: OpenAlex's inverted index format or a string
            
        Returns:
            Readable abstract text
        """
        if not isinstance(abstract_index, dict):
            return abstract_index
        word_positions = ((pos, word) 
                         for word, positions in abstract_index.items() 
                         if isinstance(positions, list)
                         for pos in positions)
        return ' '.join(word for _, word in sorted(word_positions, key=lambda x: x[0]))
    
    def _validate_date_format(self, date_str: str) -> None:
        """Validate that the date string is in YYYY-MM-DD format."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format. Got: {date_str}")
    
    def run(self, broad_field: str, keyword: str = "Psychology", limit: Optional[int] = None,
            start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """
        Extracts paper information from OpenAlex API with improved rate limiting and date filtering.
        """
        if keyword is not None and not isinstance(keyword, str):
            raise ValueError("Keyword must be a string")
        
        if limit is not None and not isinstance(limit, int):
            raise ValueError("Limit must be an integer")
            
        # Validate date formats
        if start_date is not None:
            self._validate_date_format(start_date)
            
        if end_date is not None:
            self._validate_date_format(end_date)
            
        # Validate date range
        if start_date and end_date:
            if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
                raise ValueError("Start date must be before end date")
        
        # Build filter string
        filters = ["type:article"]
        
        if start_date:
            filters.append(f"from_publication_date:{start_date}")
            
        if end_date:
            filters.append(f"to_publication_date:{end_date}")
        
        filter_str = ",".join(filters)
        
        params = {
            "mailto": self.email, 
            "search": keyword,
            "cursor": "*",
            "per_page": min(50, limit) if limit else 100, 
            "filter": filter_str
        }

        papers = []
        total_fetched = 0
        
        # Add logging setup
        start_time = time.time()
        request_count = 0
        
        # Log extraction start
        self.logger.info(f"Starting extraction for keyword: '{keyword}' in field: '{broad_field}'")
        self.logger.info(f"Date range: {start_date or 'No start'} to {end_date or 'No end'}")
        self.logger.info(f"Target limit: {limit or 'No limit'}")

        try:
            with ThreadPoolExecutor(max_workers=4) as executor: 
                while True:
                    try:
                        data = self._make_request(self.base_url, params)
                        request_count += 1
                        
                        # Log API response info
                        total_available = data.get('meta', {}).get('count', 0)
                        current_batch_size = len(data.get('results', []))
                        
                        self.logger.info(f"Total papers available in OpenAlex: {total_available:,}")
                        
                        futures = []
                        current_papers = []
                        
                        for work in data.get('results', []):
                            if limit and total_fetched >= limit:
                                self.logger.info(f"Reached extraction limit of {limit} papers")
                                return papers
                            
                            citing_works = work.get('cited_by_api_url', '')
                            referenced_works = work.get('referenced_works', [])
                            
                            future_citing = executor.submit(self._get_dois, citing_works)
                            future_referenced = executor.submit(self._get_dois, referenced_works)
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
                                'abstract': self._clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),
                                'cited_by_count': work.get('cited_by_count', 0),
                                'citing_works': None,
                                'referenced_works': None
                            }
                            current_papers.append(paper)
                            total_fetched += 1
                        
                        # Process citations for current batch
                        for i, paper in enumerate(current_papers):
                            paper['citing_works'] = futures[i*2].result()
                            paper['referenced_works'] = futures[i*2+1].result()
                            papers.append(paper)
                        
                        # Log progress every batch
                        elapsed_time = time.time() - start_time
                        papers_per_second = total_fetched / elapsed_time if elapsed_time > 0 else 0
                        
                        self.logger.info(f"Progress: {total_fetched:,} papers extracted | "
                                    f"Rate: {papers_per_second:.1f} papers/sec | "
                                    f"Elapsed: {elapsed_time:.1f}s")
                        
                        # Log progress percentage if we know the total
                        if limit:
                            progress_pct = (total_fetched / limit) * 100
                            self.logger.info(f"Progress: {progress_pct:.1f}% complete")
                        
                        next_cursor = data.get('meta', {}).get('next_cursor')
                        if not next_cursor:
                            self.logger.info("No more pages available - extraction complete")
                            break
                        
                        params['cursor'] = next_cursor
                        
                    except requests.RequestException as e:
                        self.logger.error(f"Request failed on attempt #{request_count}: {e}")
                        continue
                    
        finally:
            self.session.close()
            
            # Final summary log
            total_time = time.time() - start_time
            avg_rate = total_fetched / total_time if total_time > 0 else 0
            
            self.logger.info("="*60)
            self.logger.info("EXTRACTION SUMMARY")
            self.logger.info("="*60)
            self.logger.info(f"Keyword: {keyword}")
            self.logger.info(f"Field: {broad_field}")
            self.logger.info(f"Total papers extracted: {total_fetched:,}")
            self.logger.info(f"Total API requests made: {request_count}")
            self.logger.info(f"Total time: {total_time:.1f} seconds")
            self.logger.info(f"Average extraction rate: {avg_rate:.1f} papers/second")
            self.logger.info(f"Average papers per request: {total_fetched/request_count:.1f}" if request_count > 0 else "N/A")
            self.logger.info("="*60)
        
        return papers