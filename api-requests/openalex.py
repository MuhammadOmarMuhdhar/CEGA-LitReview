import requests

def clean_abstract(abstract_index):
    if not isinstance(abstract_index, dict):
        return abstract_index
        
    word_positions = []
    for word, positions in abstract_index.items():
        if isinstance(positions, list):
            for pos in positions:
                word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return ' '.join(word for _, word in word_positions)

def get_dois(works):
    """
    Extract DOIs from referenced works using batch requests
    Args:
        works (list): List of OpenAlex work URLs
        email (str): Email for polite pool access
    Returns:
        list: List of DOIs
    """
    dois = []
    base_url = "https://api.openalex.org/works"
    
    # Process works in batches of 50
    for i in range(0, len(works), 50):
        batch = works[i:i+50]
        # Extract work IDs and join with OR operator
        work_ids = [w.split('/')[-1] for w in batch]
        filter_str = f"id:{('|').join(work_ids)}"
        
        try:
            response = requests.get(
                base_url,
                params={
                    'filter': filter_str
                }
            )
            if response.status_code == 200:
                data = response.json()
                batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
                dois.extend(batch_dois)
        except Exception as e:
            print(f"Error fetching batch: {e}")
            
    return dois



def extract_papers_with_requests(broad_field, keyword="Psychology", limit=None):
    # Input validation
    if keyword is not None and not isinstance(keyword, str):
        raise ValueError("Keyword must be a string") 
    
    if limit is not None and not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    
    base_url = "https://api.openalex.org/works"
    
    params = {
        "mailto": "muhammad_muhdhar@berkeley.edu",
        "search": keyword,
        "cursor": "*",  
        "per_page": min(200, limit) if limit else 200 ,
        "filter": "type:article"
    }

    papers = []
    total_fetched = 0

    try:
        while True:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process current page of results
            for work in data.get('results', []):
                if limit and total_fetched >= limit:
                    return papers
                
                citing_works = work.get('citing_works', [])
                referenced_works =  work.get('referenced_works', [])
                
                    
                paper = {
                    'doi': work.get('doi', '').replace('https://doi.org/', ''),
                    'title': work.get('display_name', None),
                    'link': work.get('link', None),
                    'authors': [auth['author']['display_name'] for auth in work.get('authorships', [])],
                    'publication': 'Openalex',
                    'country': [auth['institutions'][0]['country_code'] if auth.get('institutions') else None for auth in work.get('authorships', [])],
                    'date': work.get('publication_year', None),
                    'field': broad_field ,
                    'institution': [auth['institutions'][0]['display_name'] if auth.get('institutions') else None for auth in work.get('authorships', [])],
                    'abstract': clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),
                    'cited_by_count': work.get('cited_by_count', 0),
                    'referenced_works': get_dois(referenced_works),
                    'citing_works': get_dois(citing_works)
                }
                papers.append(paper)
                total_fetched += 1
            
            # Get next cursor from meta
            next_cursor = data.get('meta', {}).get('next_cursor')
            if not next_cursor:  # Exit if no more results
                break
                
            # Update cursor for next iteration
            params['cursor'] = next_cursor
            
        return data
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return papers