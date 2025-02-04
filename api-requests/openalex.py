import requests

def clean_abstract(abstract_index):
    """
    Converts an inverted index of abstract terms into a readable abstract string.
    
    Args:
        abstract_index (dict): Inverted index where keys are words and values 
                               are lists of positions in the abstract.
    
    Returns:
        str: A reconstructed abstract as a single string, or the input if it's not a dictionary.
    """
    # Check if the input is a dictionary; if not, return it as-is
    if not isinstance(abstract_index, dict):
        return abstract_index
        
    word_positions = []  # List to store (position, word) tuples
    
    # Iterate through the dictionary to collect positions and corresponding words
    for word, positions in abstract_index.items():
        if isinstance(positions, list):  # Ensure positions are in list format
            for pos in positions:
                word_positions.append((pos, word))
    
    # Sort the words by their positions
    word_positions.sort(key=lambda x: x[0])
    
    # Reconstruct the abstract as a space-separated string of words
    return ' '.join(word for _, word in word_positions)


def get_dois(works):
    """
    Extracts DOIs (Digital Object Identifiers) from referenced works using batch requests.
    
    Args:
        works (list): List of OpenAlex work URLs.
    
    Returns:
        list: A list of DOIs extracted from the referenced works.
    """
    dois = []  # List to store DOIs
    base_url = "https://api.openalex.org/works"
    
    # Process works in batches of 50 (API limitation)
    for i in range(0, len(works), 50):
        batch = works[i:i+50]  # Create a batch of up to 50 works
        
        # Extract work IDs from URLs (last part of the URL)
        work_ids = [w.split('/')[-1] for w in batch]
        
        # Construct filter string using OR operator between IDs
        filter_str = f"id:{('|').join(work_ids)}"
        
        try:
            # Make a GET request to fetch data for the batch
            response = requests.get(
                base_url,
                params={
                    'filter': filter_str
                }
            )
            
            # Check if the response is successful
            if response.status_code == 200:
                data = response.json()
                
                # Extract DOIs from the results and add them to the list
                batch_dois = [work.get('doi') for work in data.get('results', []) if work.get('doi')]
                dois.extend(batch_dois)
        
        except Exception as e:
            print(f"Error fetching batch: {e}")  # Log any errors during the request
            
    return dois


def extract_papers_with_requests(broad_field, keyword="Psychology", limit=None):
    """
    Fetches academic papers from OpenAlex based on a keyword and other parameters.
    
    Args:
        broad_field (str): The overall field or category of research.
        keyword (str): The keyword to search for papers. Defaults to "Psychology".
        limit (int, optional): Maximum number of papers to fetch. Defaults to None (no limit).
    
    Returns:
        list: A list of dictionaries containing paper details.
    """
    
    # Validate input arguments
    if keyword is not None and not isinstance(keyword, str):
        raise ValueError("Keyword must be a string") 
    
    if limit is not None and not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    
    base_url = "https://api.openalex.org/works"  # Base URL for OpenAlex API
    
    params = {
        "mailto": "muhammad_muhdhar@berkeley.edu",  # Email for polite pool access
        "search": keyword,                          # Search query
        "cursor": "*",                              # Cursor for pagination
        "per_page": min(200, limit) if limit else 200,  # Max results per page (200 or limit)
        "filter": "type:article"                   # Filter to include only articles
    }

    papers = []  # List to store fetched papers
    total_fetched = 0  # Counter for total fetched papers

    try:
        while True:
            # Make a GET request to fetch data based on current parameters
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()  # Parse JSON response
            
            # Process each paper in the current page of results
            for work in data.get('results', []):
                if limit and total_fetched >= limit:  # Stop fetching if limit is reached
                    return papers
                
                citing_works = work.get('citing_works', [])  # Works citing this paper
                referenced_works = work.get('referenced_works', [])  # Works referenced by this paper
                
                paper = {
                    'doi': work.get('doi', '').replace('https://doi.org/', ''),  # DOI without URL prefix
                    'title': work.get('display_name', None),                     # Paper title
                    'link': work.get('link', None),                             # Link to the paper
                    'authors': [auth['author']['display_name'] 
                                for auth in work.get('authorships', [])],       # List of authors' names
                    'publication': 'Openalex',                                  # Publication source
                    'country': [auth['institutions'][0]['country_code'] 
                                if auth.get('institutions') else None 
                                for auth in work.get('authorships', [])],       # Authors' countries
                    'date': work.get('publication_year', None),                 # Publication year
                    'field': broad_field,                                       # Broad research field
                    'institution': [auth['institutions'][0]['display_name'] 
                                    if auth.get('institutions') else None 
                                    for auth in work.get('authorships', [])],   # Authors' institutions
                    'abstract': clean_abstract(work.get('abstract_inverted_index', 'No Abstract')),  # Abstract text
                    'cited_by_count': work.get('cited_by_count', 0),            # Number of times cited
                    'referenced_works': get_dois(referenced_works),             # DOIs of referenced works
                    'citing_works': get_dois(citing_works)                      # DOIs of citing works
                }
                
                papers.append(paper)  # Add paper details to the list
                total_fetched += 1
            
            next_cursor = data.get('meta', {}).get('next_cursor')  # Get cursor for next page
            
            if not next_cursor:  # Exit loop if no more pages are available
                break
            
            params['cursor'] = next_cursor  # Update cursor for next iteration
            
        return papers
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")  # Log any request-related errors
        
        return papers

