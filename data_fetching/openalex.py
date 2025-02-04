# import requests 

# def clean_abstract(abstract_index):
#     """
#     Reconstructs an abstract from OpenAlex's inverted index format into readable text.
    
#     Args:
#         abstract_index (dict or str): Either a dictionary containing the inverted index
#             format from OpenAlex (word -> positions) or a plain text string.
    
#     Returns:
#         str: The reconstructed abstract text with words in correct order, or the original
#             input if it wasn't in inverted index format.
#     """
#     # Return as-is if not in inverted index format
#     if not isinstance(abstract_index, dict):
#         return abstract_index
    
#     # Store word-position pairs for sorting
#     word_positions = []
#     for word, positions in abstract_index.items():
#         # Each word can appear in multiple positions
#         if isinstance(positions, list):
#             for pos in positions:
#                 word_positions.append((pos, word))
    
#     # Sort words based on their position in the original text
#     word_positions.sort(key=lambda x: x[0])
#     # Reconstruct the text by joining words in order
#     return ' '.join(word for _, word in word_positions)

# def get_dois(works):
#     """
#     Extracts DOIs (Digital Object Identifiers) from referenced works using batch requests.
#     Handles large lists of works by processing them in batches to respect API limitations.
    
#     Args:
#         works (list): List of OpenAlex work URLs, each containing a work identifier.
    
#     Returns:
#         list: A list of cleaned DOIs (without 'https://doi.org/' prefix) extracted 
#             from the referenced works.
#     """
#     dois = []  
#     base_url = "https://api.openalex.org/works"

#     # Process works in batches of 200 to stay within API limits
#     for i in range(0, len(works), 200):
#         batch = works[i:i+200]  
        
#         # Extract the work ID from the end of each URL
#         work_ids = [w.split('/')[-1] for w in batch]
        
#         # Create an OR filter to fetch multiple works in one request
#         # Format: openalex:ID1|ID2|ID3...
#         filter_str = f"openalex:{('|').join(work_ids)}"
        
#         try:
#             # Construct and execute the batch request
#             url = f"{base_url}?filter={filter_str}"
#             response = requests.get(url)
            
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # Extract and clean DOIs:
#                 # 1. Get DOIs from works that have them
#                 # 2. Remove the 'https://doi.org/' prefix
#                 # 3. Filter out any empty DOIs
#                 batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
#                 clean_dois = [doi.strip('https://doi.org/') for doi in batch_dois if doi]
#                 dois.extend(clean_dois)
        
#         except Exception as e:
#             print(f"Error fetching batch: {e}")
            
#     return dois

# def extract_papers(broad_field, keyword="Psychology", limit=None):
#     """
#     Extracts paper information from OpenAlex API based on specified criteria.
    
#     Args:
#         broad_field (str): The broad academic field to categorize the papers under.
#         keyword (str, optional): Search term to filter papers. Defaults to "Psychology".
#         limit (int, optional): Maximum number of papers to retrieve. Defaults to None.
    
#     Returns:
#         list: List of dictionaries containing detailed information about each paper:
#             - doi: Digital Object Identifier
#             - title: Paper title
#             - link: URL to the paper
#             - authors: List of author names
#             - publication: Always 'Openalex'
#             - country: List of author institution country codes
#             - date: Publication year
#             - field: Broad academic field
#             - institution: List of author institution names
#             - abstract: Paper abstract
#             - cited_by_count: Number of citations
#             - citing_works: List of DOIs that cite this paper
#             - referenced_works: List of DOIs this paper cites
#     """
#     # Validate input parameters
#     if keyword is not None and not isinstance(keyword, str):
#         raise ValueError("Keyword must be a string") 
    
#     if limit is not None and not isinstance(limit, int):
#         raise ValueError("Limit must be an integer")
    
#     base_url = "https://api.openalex.org/works"
    
#     # Set up initial API request parameters
#     params = {
#         "mailto": "muhammad_muhdhar@berkeley.edu",  # Required for higher rate limits
#         "search": keyword,
#         "cursor": "*",  # Start with first page
#         "per_page": min(200, limit) if limit else 200,  # Respect API pagination limits
#         "filter": f"type:article",
#     }

#     papers = []
#     total_fetched = 0

#     try:
#         while True:
#             # Fetch current page of results
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()  # Raise exception for bad status codes
#             data = response.json()
            
#             # Process each paper on the current page
#             for work in data.get('results', []):
#                 if limit and total_fetched >= limit:
#                     return papers
                
#                 # Prepare citing works URL for reference
#                 citing_works_url = f"https://api.openalex.org/works?filter=cites:{work['id']}"
                
#                 # Extract and clean citing works URLs
#                 citing_works = work.get('cited_by_api_url', [])
#                 citing_works_stripped = [x.strip() for x in citing_works.replace('works?filter=cites:', '').split(',')]
#                 referenced_works = work.get('referenced_works', [])
                
#                 # Construct paper dictionary with all available metadata
#                 paper = {
#                     'doi': work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None,
#                     'title': work.get('display_name', None),
#                     'link': work.get('link', None),
#                     'authors': [auth['author']['display_name'] for auth in work.get('authorships', [])],
#                     'publication': 'Openalex',
#                     'country': [auth['institutions'][0]['country_code'] if auth.get('institutions') else None 
#                               for auth in work.get('authorships', [])],
#                     'date': work.get('publication_year', None),
#                     'field': broad_field,
#                     'institution': [auth['institutions'][0]['display_name'] if auth.get('institutions') else None 
#                                   for auth in work.get('authorships', [])],
#                     'abstract': clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),
#                     'cited_by_count': work.get('cited_by_count', 0),
#                     'citing_works': get_dois(citing_works_stripped),
#                     'referenced_works': get_dois(referenced_works)
#                 }
#                 papers.append(paper)
#                 total_fetched += 1
            
#             # Check for more pages
#             next_cursor = data.get('meta', {}).get('next_cursor')
#             if not next_cursor:  # No more results available
#                 break
                
#             # Update cursor for next page
#             params['cursor'] = next_cursor
            
#         return data
            
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching data: {e}")
#         return papers

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_session() -> requests.Session:
    """Creates a requests session with retry logic and timeouts."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('https://', adapter)
    return session

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

def get_dois(works: List[str], session: requests.Session) -> List[str]:
    """
    Extracts DOIs from referenced works using batch requests.
    
    Args:
        works (list): List of OpenAlex work URLs.
        session (requests.Session): Requests session for connection reuse.
    
    Returns:
        list: A list of cleaned DOIs.
    """
    if not works:
        return []

    dois = []
    base_url = "https://api.openalex.org/works"

    # Process works in batches of 200
    for i in range(0, len(works), 200):
        batch = works[i:i+200]
        work_ids = [w.split('/')[-1] for w in batch]
        filter_str = f"openalex:{('|').join(work_ids)}"
        
        try:
            url = f"{base_url}?filter={filter_str}"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # Basic rate limiting
            time.sleep(0.1)
            
            if response.status_code == 200:
                data = response.json()
                batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
                clean_dois = [doi.strip('https://doi.org/') for doi in batch_dois if doi]
                dois.extend(clean_dois)
                
        except requests.Timeout:
            logger.error("Request timed out while fetching DOIs")
        except requests.RequestException as e:
            logger.error(f"Error fetching batch: {e}")
            
    return dois

def extract_papers(broad_field: str, keyword: str = "Psychology", limit: Optional[int] = None) -> List[Dict]:
    """
    Extracts paper information from OpenAlex API with improved efficiency.
    
    Args:
        broad_field (str): The broad academic field to categorize the papers under.
        keyword (str, optional): Search term to filter papers. Defaults to "Psychology".
        limit (int, optional): Maximum number of papers to retrieve. Defaults to None.
    
    Returns:
        list: List of dictionaries containing paper information.
    """
    if keyword is not None and not isinstance(keyword, str):
        raise ValueError("Keyword must be a string")
    
    if limit is not None and not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    
    base_url = "https://api.openalex.org/works"
    params = {
        "mailto": "muhammad_muhdhar@berkeley.edu",
        "search": keyword,
        "cursor": "*",
        "per_page": min(200, limit) if limit else 200,
        "filter": "type:article"
    }

    papers = []
    total_fetched = 0
    session = create_session()

    try:
        with ThreadPoolExecutor(max_workers=5) as executor:
            while True:
                try:
                    response = session.get(base_url, params=params, timeout=30)
                    response.raise_for_status()
                    time.sleep(0.1)  # Rate limiting
                    data = response.json()
                    
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
                        
                        # Submit tasks to thread pool
                        future_citing = executor.submit(get_dois, citing_works_stripped, session)
                        future_referenced = executor.submit(get_dois, referenced_works, session)
                        futures.extend([future_citing, future_referenced])
                        
                        paper = {
                            'doi': work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None,
                            'title': work.get('display_name', None),
                            'link': work.get('link', None),
                            'authors': [auth['author']['display_name'] for auth in work.get('authorships', [])],
                            'publication': 'Openalex',
                            'country': [auth['institutions'][0]['country_code'] if auth.get('institutions') else None 
                                      for auth in work.get('authorships', [])],
                            'date': work.get('publication_year', None),
                            'field': broad_field,
                            'institution': [auth['institutions'][0]['display_name'] if auth.get('institutions') else None 
                                          for auth in work.get('authorships', [])],
                            'abstract': clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),
                            'cited_by_count': work.get('cited_by_count', 0),
                            'citing_works': None,  # Will be filled with future result
                            'referenced_works': None  # Will be filled with future result
                        }
                        current_papers.append(paper)
                        total_fetched += 1
                    
                    # Get results from futures and update papers
                    for i, paper in enumerate(current_papers):
                        paper['citing_works'] = futures[i*2].result()
                        paper['referenced_works'] = futures[i*2+1].result()
                        papers.append(paper)
                    
                    next_cursor = data.get('meta', {}).get('next_cursor')
                    if not next_cursor:
                        break
                    
                    params['cursor'] = next_cursor
                    
                except requests.Timeout:
                    logger.error("Request timed out")
                    continue
                except requests.RequestException as e:
                    logger.error(f"Error fetching data: {e}")
                    continue
                
    finally:
        session.close()
    
    return papers