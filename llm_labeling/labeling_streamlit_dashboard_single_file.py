import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import math
import json
import random
import tiktoken

# Check for pycountry and install if missing
try:
    import pycountry
except ImportError:
    st.error("""
    The required package 'pycountry' is not installed. Please install it using:
    ```
    pip install pycountry
    ```
    Or install all requirements using:
    ```
    pip install -r requirements.txt
    ```
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Research Paper Analysis Dashboard",
    page_icon="📚",
    layout="wide"
)

# Constants for rate limiting and batch processing
REQUESTS_PER_MINUTE = 15  # Corrected rate limit
TOKENS_PER_MINUTE = 1000000  # 1M tokens per minute for Gemini 2.0
REQUESTS_PER_DAY = 1500  # Increased daily limit
MAX_INPUT_TOKENS = 32768  # Maximum tokens per request for Gemini 2.0 Flash
MAX_OUTPUT_TOKENS = 2048  # Maximum expected tokens for analysis response
INITIAL_BATCH_SIZE = 100  # Start with larger batches
FALLBACK_BATCH_SIZE = 50  # Fallback size if rate limit is hit
WAIT_TIME = 60 / REQUESTS_PER_MINUTE  # Minimum wait time between requests (4 seconds)
MAX_PAPERS = REQUESTS_PER_DAY * INITIAL_BATCH_SIZE  # Theoretical maximum papers per day
MAX_RETRIES = 3  # Maximum retry attempts for rate limit errors
BASE_DELAY = 4  # Base delay for exponential backoff

# Create a set of valid country names and special options
VALID_COUNTRIES = {country.name for country in pycountry.countries}
SPECIAL_OPTIONS = {"NA", "10+ countries"}
COUNTRY_ALIASES = {
    "United States": "United States of America",
    "USA": "United States of America",
    "US": "United States of America",
    "UK": "United Kingdom",
    "UAE": "United Arab Emirates",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Taiwan": "Taiwan, Province of China",
    "Vietnam": "Viet Nam",
    "Syria": "Syrian Arab Republic",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Iran": "Iran, Islamic Republic of",
    "Bolivia": "Bolivia, Plurinational State of",
    "Tanzania": "Tanzania, United Republic of"
}

# Add request tracking
class RequestTracker:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0
        self.last_minute_requests = []
    
    def add_request(self, success=True, rate_limited=False):
        current_time = time.time()
        self.last_minute_requests = [t for t in self.last_minute_requests if current_time - t < 60]
        self.last_minute_requests.append(current_time)
        self.total_requests += 1
        if rate_limited:
            self.rate_limited_requests += 1
        elif success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def get_current_rpm(self):
        current_time = time.time()
        self.last_minute_requests = [t for t in self.last_minute_requests if current_time - t < 60]
        return len(self.last_minute_requests)
    
    def get_stats(self):
        return {
            "total": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "rate_limited": self.rate_limited_requests,
            "current_rpm": self.get_current_rpm()
        }

# Initialize request tracker in session state
if 'request_tracker' not in st.session_state:
    st.session_state.request_tracker = RequestTracker()

def count_tokens(text):
    """
    Estimate token count for input text.
    Uses OpenAI's tiktoken for accurate estimation, falls back to word-based approximation.
    
    Args:
        text (str): Input text to count tokens for
        
    Returns:
        int: Estimated token count
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # Fallback to simple approximation if tiktoken not available
        return len(text.split()) * 1.3  # Rough approximation: average tokens per word

def estimate_batch_tokens(batch_df, prompt_template):
    """
    Estimate total tokens for a batch of abstracts including prompt and overhead.
    
    Args:
        batch_df (pd.DataFrame): DataFrame containing abstracts to process
        prompt_template (str): Template text for analysis prompt
        
    Returns:
        int: Total estimated tokens for the batch
    """
    template_tokens = count_tokens(prompt_template)
    abstract_tokens = sum(count_tokens(str(abstract)) for abstract in batch_df['abstract'])
    overhead_per_paper = 50  # Tokens for JSON structure, DOI, and formatting per paper
    overhead_tokens = len(batch_df) * overhead_per_paper
    
    return template_tokens + abstract_tokens + overhead_tokens

def get_optimal_batch_size(df, prompt_template, target_tokens=MAX_INPUT_TOKENS * 0.9):
    """
    Calculate optimal batch size based on token limits and average abstract length.
    Uses 90% of max tokens by default to leave buffer for variations.
    
    Args:
        df (pd.DataFrame): DataFrame containing all abstracts
        prompt_template (str): Template text for analysis prompt
        target_tokens (int): Target token limit (default: 90% of MAX_INPUT_TOKENS)
        
    Returns:
        int: Optimal number of abstracts per batch
    """
    sample_size = min(10, len(df))
    sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
    avg_tokens_per_abstract = estimate_batch_tokens(sample_df, "") / len(sample_df)
    
    template_tokens = count_tokens(prompt_template)
    overhead_per_paper = 50
    
    available_tokens = target_tokens - template_tokens
    tokens_per_paper = avg_tokens_per_abstract + overhead_per_paper
    max_papers = math.floor(available_tokens / tokens_per_paper)
    
    return max(1, min(max_papers, INITIAL_BATCH_SIZE))

def validate_country(country_str):
    """
    Validate and normalize country names using pycountry.
    Handles special cases and common aliases.
    
    Args:
        country_str (str): Country name to validate
        
    Returns:
        str: Normalized country name or original string if special case
    """
    if not country_str or pd.isna(country_str):
        return "NA"
    
    # Clean the input string
    country_str = country_str.strip()
    
    # Check for special options
    if country_str in SPECIAL_OPTIONS:
        return country_str
    
    # Check aliases first
    if country_str in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[country_str]
    
    # Check if it's already a valid country name
    if country_str in VALID_COUNTRIES:
        return country_str
    
    # Try to search for the country
    try:
        # Try exact search first
        country = pycountry.countries.search_fuzzy(country_str)[0]
        return country.name
    except LookupError:
        return f"Invalid country: {country_str}"

def create_batch_prompt(abstracts_df, prompt_template):
    """
    Create a prompt for batch processing with token limit management.
    Dynamically adjusts batch size if token limit would be exceeded.
    
    Args:
        abstracts_df (pd.DataFrame): DataFrame containing abstracts to process
        prompt_template (str): Template text for analysis prompt
        
    Returns:
        tuple: (formatted prompt string, number of abstracts included)
    """
    abstracts_list = []
    current_tokens = count_tokens(prompt_template)
    
    for idx, row in abstracts_df.iterrows():
        abstract_text = f"Abstract {idx + 1} (DOI: {row['doi']}):\n{row['abstract']}"
        abstract_tokens = count_tokens(abstract_text)
        
        if current_tokens + abstract_tokens + 100 > MAX_INPUT_TOKENS:
            st.warning(f"Batch size adjusted due to token limit. Processing {len(abstracts_list)} abstracts instead of {len(abstracts_df)}")
            break
        
        abstracts_list.append(abstract_text)
        current_tokens += abstract_tokens
    
    # Get list of valid country names for the prompt
    valid_country_examples = random.sample(list(VALID_COUNTRIES), 10)  # Sample 10 random countries
    
    batch_prompt = f"""You are a research paper analyzer. Your task is to analyze multiple research paper abstracts and provide structured results.

RESPONSE FORMAT:
You must respond with a valid JSON object in exactly this format:
{{
    "results": [
        {{
            "abstract_number": 1,
            "doi": "paper_doi",
            "analysis": "1,2,4;3,7,12;1,2,6;1-1,3-2;USA;3"
        }}
    ]
}}

STRICT RULES:
1. Your response must be ONLY the JSON object, nothing else
2. Each abstract's analysis must follow the exact format: numbers for tasks 1-4,6 and country names for task 5
3. Use semicolons (;) to separate tasks and commas (,) to separate items within tasks
4. Provide exactly 6 answers per abstract, separated by 5 semicolons
5. Do not include any explanations or additional text outside the JSON structure
6. Ensure all DOIs match exactly as provided
7. For Task 5 (countries), use ONLY:
   - Official country names from the standard list (examples: {', '.join(valid_country_examples)})
   - "NA" if no sample was collected
   - "10+ countries" if more than 10 countries
   - Use commas to separate multiple countries (max 10)

Here are the abstracts to analyze:

{chr(10).join(abstracts_list)}

Original Analysis Instructions:
{prompt_template}

Remember: Respond ONLY with the JSON object, no other text."""
    return batch_prompt, len(abstracts_list)

def process_with_retries(model, prompt, retry_count=0):
    """
    Process a request with exponential backoff retry logic for rate limits.
    Added safety parameters for generation and request tracking.
    """
    try:
        # Check current request rate
        current_rpm = st.session_state.request_tracker.get_current_rpm()
        if current_rpm >= REQUESTS_PER_MINUTE:
            wait_time = max(WAIT_TIME, 60 - (time.time() - st.session_state.request_tracker.last_minute_requests[0]))
            st.warning(f"Rate limit approaching ({current_rpm}/{REQUESTS_PER_MINUTE} rpm). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        generation_config = {
            'temperature': 0.1,
            'top_p': 0.95,
            'top_k': 40,
            'candidate_count': 1
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Track successful request
        st.session_state.request_tracker.add_request(success=True)
        
        # Check if response is blocked
        if hasattr(response, 'prompt_feedback'):
            if response.prompt_feedback.block_reason:
                st.warning(f"Response was blocked: {response.prompt_feedback.block_reason}")
                st.session_state.request_tracker.add_request(success=False)
                return None
        
        return response
    except Exception as e:
        error_str = str(e)
        if "429" in error_str and retry_count < MAX_RETRIES:
            # Track rate limited request
            st.session_state.request_tracker.add_request(rate_limited=True)
            
            retry_delay = BASE_DELAY
            if "retry_delay" in error_str:
                try:
                    retry_delay = int(error_str.split("retry_delay")[1].split("seconds:")[1].split("}")[0].strip())
                except:
                    retry_delay = exponential_backoff(retry_count)
            else:
                retry_delay = exponential_backoff(retry_count)
            
            st.warning(f"Rate limit hit. Waiting {retry_delay:.1f} seconds before retry {retry_count + 1}/{MAX_RETRIES}...")
            time.sleep(retry_delay)
            return process_with_retries(model, prompt, retry_count + 1)
        else:
            # Track failed request
            st.session_state.request_tracker.add_request(success=False)
            raise e

def process_batch(batch_df, model, prompt_template, status_container):
    """
    Process a batch of abstracts with dynamic batch size adjustment.
    Starts with larger batches and reduces size if rate limits are hit.
    
    Args:
        batch_df (pd.DataFrame): DataFrame containing abstracts to process
        model: Gemini model instance
        prompt_template (str): Template text for analysis prompt
        status_container: Streamlit container for status updates
        
    Returns:
        list: Processing results for all abstracts
    """
    try:
        # Reset index to ensure proper alignment
        batch_df = batch_df.reset_index(drop=True)
        
        # Start with optimal size up to INITIAL_BATCH_SIZE
        optimal_size = min(get_optimal_batch_size(batch_df, prompt_template), INITIAL_BATCH_SIZE)
        current_batch_size = optimal_size
        all_results = []
        processed_indices = set()  # Track which papers have been processed
        rate_limit_hits = 0  # Track rate limit occurrences
        
        while len(processed_indices) < len(batch_df):
            # Get next batch of unprocessed papers
            remaining_df = batch_df[~batch_df.index.isin(processed_indices)]
            if len(remaining_df) == 0:
                break
                
            start_idx = remaining_df.index[0]
            end_idx = min(start_idx + current_batch_size, len(batch_df))
            sub_batch_df = batch_df.iloc[start_idx:end_idx].copy()
            
            batch_prompt, actual_size = create_batch_prompt(sub_batch_df, prompt_template)
            token_count = count_tokens(batch_prompt)
            
            with st.expander(f"Batch Details (Size: {current_batch_size})", expanded=False):
                st.info(f"""
                Token Usage:
                - Input tokens: {token_count:,}
                - Max allowed: {MAX_INPUT_TOKENS:,}
                - Papers in batch: {actual_size}
                - Current batch size: {current_batch_size}
                - Rate limit hits: {rate_limit_hits}
                """)
            
            if token_count > MAX_INPUT_TOKENS:
                st.warning(f"Token limit exceeded ({token_count:,} > {MAX_INPUT_TOKENS:,}). Reducing batch size.")
                current_batch_size = max(1, current_batch_size // 2)
                continue
            
            try:
                response = process_with_retries(model, batch_prompt)
                if response is None:
                    st.error("Response was blocked by safety settings. Skipping batch.")
                    # Mark these as processed but failed
                    for idx in sub_batch_df.index:
                        processed_indices.add(idx)
                        all_results.append({
                            'doi': sub_batch_df.loc[idx, 'doi'],
                            'gemini_response': "ERROR: Response blocked by safety settings"
                        })
                    continue
                
                results = parse_batch_response(response.text)
                
                # Process results and update tracking
                for idx, (_, row) in enumerate(sub_batch_df.iterrows()):
                    processed_indices.add(row.name)
                    if idx < len(results):
                        result = results[idx]
                        if result['doi'] != row['doi']:
                            st.warning(f"DOI mismatch detected. Expected {row['doi']}, got {result['doi']}")
                        all_results.append({
                            'doi': row['doi'],
                            'gemini_response': result['analysis']
                        })
                    else:
                        all_results.append({
                            'doi': row['doi'],
                            'gemini_response': "ERROR: Missing from response"
                        })
                
                # Reset rate limit counter on successful processing
                rate_limit_hits = 0
                
                # If we've processed several batches successfully, try increasing the batch size
                if len(processed_indices) % (current_batch_size * 3) == 0 and current_batch_size < INITIAL_BATCH_SIZE:
                    current_batch_size = min(INITIAL_BATCH_SIZE, current_batch_size * 2)
                    st.success(f"Processing stable. Increasing batch size to {current_batch_size}")
                
                if len(processed_indices) < len(batch_df):
                    time.sleep(WAIT_TIME)
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:  # Rate limit hit
                    rate_limit_hits += 1
                    st.warning(f"Rate limit hit ({rate_limit_hits} times). Reducing batch size from {current_batch_size} to {max(FALLBACK_BATCH_SIZE, current_batch_size // 2)}")
                    current_batch_size = max(FALLBACK_BATCH_SIZE, current_batch_size // 2)
                    # Don't mark as processed, will retry with smaller batch
                    continue
                else:
                    st.error(f"Error processing batch: {str(e)}")
                    # Mark as processed but failed
                    for idx in sub_batch_df.index:
                        processed_indices.add(idx)
                        all_results.append({
                            'doi': sub_batch_df.loc[idx, 'doi'],
                            'gemini_response': f"ERROR: {str(e)}"
                        })
        
        return all_results
    except Exception as e:
        st.error(f"Error in batch processing: {str(e)}")
        return []

# Title and description
st.title("Research Paper Analysis Dashboard")

# Create two columns for the description
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    This dashboard helps analyze research papers using Gemini AI. Upload a CSV file containing DOIs and abstracts,
    configure your API key, and customize the analysis prompt.
    """)

with col2:
    st.info("""
    **Gemini 2.0 Flash Limits**:
    - 15 requests per minute
    - 1M tokens per minute
    - 1,500 requests per day
    - Up to 32K input tokens per request
    """)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input with better explanation
    st.subheader("1. API Key")
    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        help="Required for accessing Gemini AI API"
    )
    if api_key:
        genai.configure(api_key=api_key)
    
    # File uploader with better explanation
    st.subheader("2. Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV with DOI and abstract columns",
        type="csv",
        help="CSV file must contain 'doi' and 'abstract' columns"
    )
    
    # Processing options
    if uploaded_file is not None:
        try:
            # Try to read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if the file is empty
            if df.empty:
                st.error("The uploaded CSV file is empty. Please upload a file with data.")
            # Check if required columns exist
            elif 'doi' not in df.columns or 'abstract' not in df.columns:
                st.error("Required columns missing. Available columns: " + ", ".join(df.columns))
                st.info("Please ensure your CSV has 'doi' and 'abstract' columns.")
            else:
                # Count valid entries (non-null DOIs and abstracts)
                valid_entries = df[df['doi'].notna() & df['abstract'].notna()].shape[0]
                if valid_entries == 0:
                    st.error("No valid entries found with both DOI and abstract.")
                else:
                    st.success(f"✅ Found {valid_entries:,} valid papers")
                    
                    st.subheader("3. Processing Options")
                    
                    # Calculate maximum allowed papers
                    max_allowed = min(valid_entries, MAX_PAPERS)
                    
                    # Paper selection method
                    selection_method = st.radio(
                        "Paper Selection Method",
                        ["Process All", "Random Sample", "First N Papers"],
                        help="Choose how to select papers for analysis"
                    )
                    
                    if selection_method == "Process All":
                        sample_size = min(valid_entries, MAX_PAPERS)
                        st.info(f"Will process all {sample_size:,} papers")
                    else:
                        # Sample size slider with dynamic max value
                        sample_size = st.slider(
                            "Number of papers to analyze",
                            min_value=100,
                            max_value=max_allowed,
                            value=min(1000, max_allowed),
                            step=100,
                            help=f"Maximum {MAX_PAPERS:,} papers per day"
                        )
                    
                    # Display batch information
                    num_batches = math.ceil(sample_size / INITIAL_BATCH_SIZE)
                    estimated_time = num_batches * (WAIT_TIME + 10)  # 10 seconds processing time per batch
                    
                    st.info(f"""
                    **Processing Details**:
                    - Papers to analyze: {sample_size:,}
                    - Number of batches: {num_batches:,}
                    - Papers per batch: {INITIAL_BATCH_SIZE:,}
                    - Estimated time: {estimated_time/60:.1f} minutes
                    """)
                    
        except pd.errors.EmptyDataError:
            st.error("The uploaded file appears to be empty.")
        except Exception as e:
            st.error(f"Error reading the CSV file: {str(e)}")

# Sci-Hub URLs and headers (kept from original script)
SCI_HUB_URLS = [
    "https://sci-hub.ru/",
    "https://sci-hub.se/",
    "https://sci-hub.st/",
    "https://sci-hub.ee/",
    "https://sci-hub.shop/"
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Default prompt
DEFAULT_PROMPT = '''You are a helpful literature review assistant. Focus on the paper itself (rather than others' works discussed in this paper) and analyze the file.

IMPORTANT FORMAT INSTRUCTIONS:
1. Output ONLY numbers for each task (except Task 5 which requires country names)
2. Separate numbers within each task using commas (,)
3. Separate different tasks using semicolons (;)
4. Do not include any explanations or additional text
5. Provide exactly 6 answers (one for each task) separated by 5 semicolons

Example correct formats:
- "1,2,4; 3,7,12; 1,2,6; 1-1,3-2; USA; 3"
- "1; 6,7,8; 15; 3-1; NA; 9"
- "2,3,5; 1,4,5,13; 2,8,10; 2-2,5,8; India,China; 4"

Task 1: Determine which of the following kind of "poverty" is being focused in this paper. Select all that apply:

1. Low resource level: Situations where individuals or families experience difficulty meeting basic needs such as food, shelter, or clothing
2. Resource volatility: Conditions characterized by unstable or unpredictable access to financial resources
3. Physical environment: The quality and safety of the physical surroundings
4. Human capital inputs: Investments in education, health, and social services
5. Social environment: The presence of social stigma, discrimination, or stereotypes

Task 2: Determine which of the following psychological mechanisms are being studied in this paper. Select all that apply:

Cognitive function:
1. Attention
2. Cognitive flexibility
3. Executive control
4. Fluid intelligence
5. Memory

Mental health:
6. Anxiety
7. Depression
8. Happiness
9. Stress

Beliefs:
10. Aspirations
11. Internalized stigma
12. Locus of control
13. Self-efficacy
14. Self-esteem

Preferences:
15. Time preferences
16. Risk preferences

Task 3: Determine which of the following behaviors are discussed in this paper. Select all that apply:

Economic:
1. Borrowing
2. Consumption
3. Investing
4. Labor supply
5. Occupational choice
6. Saving

Health:
7. Exercise
8. Food diet
9. Medication adherence

Intergenerational:
10. Educational achievement
11. Family interaction
12. Parenting

Social:
13. Crime
14. Violence

15. NA — if no behaviors were discussed

Task 4: Determine which of the following study types are mentioned or used for this paper's methodology:

Quantitative:
1. Experimental
    1-1. Randomized (controlled) trial
    1-2. Natural experiments
2. Quasi-experimental
    2-1. Regression discontinuity
    2-2. Difference-in-difference
    2-3. Instrumental variable design
    2-4. Natural experiments
3. Non-experimental
    3-1. Observational
    3-2. Cross-sectional
    3-3. Longitudinal / Panel
4. Meta-analysis

Qualitative:
5. Case-study
6. Ethnography
7. First-hand observations
8. Interviews
9. Focus groups
10. Recordings

Mixed Method:
11. Systematic review

Task 5: List the countries in which the sample was collected.
- If more than 10 countries: write "10+ countries"
- If no sample was collected: write "NA"
- Otherwise: list all countries separated by commas

Task 6: Select the sample size from the largest study conducted in this publication:
1. Less than 100
2. 100–499
3. 500–999
4. 1,000–4,999
5. 5,000–9,999
6. 10,000–49,999
7. 50,000–99,999
8. 100,000 or more
9. NA'''

# Add mapping dictionaries after the DEFAULT_PROMPT
task1_map = {
    "1": "Low resource level", "2": "Resource volatility", "3": "Physical environment",
    "4": "Human capital inputs", "5": "Social environment"
}
task2_map = {
    "1": "Attention", "2": "Cognitive flexibility", "3": "Executive control", "4": "Fluid intelligence", "5": "Memory",
    "6": "Anxiety", "7": "Depression", "8": "Happiness", "9": "Stress",
    "10": "Aspirations", "11": "Internalized stigma", "12": "Locus of control", "13": "Self-efficacy", "14": "Self-esteem",
    "15": "Time preferences", "16": "Risk preferences"
}
task3_map = {
    "1": "Borrowing", "2": "Consumption", "3": "Investing", "4": "Labor supply", "5": "Occupational choice", "6": "Saving",
    "7": "Exercise", "8": "Food diet", "9": "Medication adherence",
    "10": "Educational achievement", "11": "Family interaction", "12": "Parenting",
    "13": "Crime", "14": "Violence", "15": "NA"
}
task4_map = {
    "1-1": "Randomized (controlled) trial", "1-2": "Natural experiments",
    "2-1": "Regression discontinuity", "2-2": "Difference-in-difference", "2-3": "Instrumental variable design", "2-4": "Natural experiments",
    "3-1": "Observational", "3-2": "Cross-sectional", "3-3": "Longitudinal / Panel",
    "4": "Meta-analysis", "5": "Case-study", "6": "Ethnography", "7": "First-hand observations",
    "8": "Interviews", "9": "Focus groups", "10": "Recordings", "11": "Systematic review"
}
task6_map = {
    "1": "<100", "2": "100–499", "3": "500–999", "4": "1,000–4,999",
    "5": "5,000–9,999", "6": "10,000–49,999", "7": "50,000–99,999",
    "8": "100,000+", "9": "NA"
}

def map_labels(code_str, mapping):
    if pd.isna(code_str):
        return ""
    return ", ".join([mapping.get(code.strip(), code.strip()) for code in code_str.split(",")])

def get_broad_study_type(study_str):
    if pd.isna(study_str):
        return ""
    methods = [s.strip() for s in study_str.split(",")]
    broad_types = set()
    for m in methods:
        if m in [
            "Randomized (controlled) trial", "Natural experiments",
            "Regression discontinuity", "Difference-in-difference", "Instrumental variable design",
            "Observational", "Cross-sectional", "Longitudinal / Panel",
            "Meta-analysis"
        ]:
            broad_types.add("Quantitative")
        elif m in [
            "Case-study", "Ethnography", "First-hand observations",
            "Interviews", "Focus groups", "Recordings"
        ]:
            broad_types.add("Qualitative")
        elif m == "Systematic review":
            broad_types.add("Mixed Method")
    return ", ".join(sorted(broad_types))

def get_psych_mechanism_broad(mech_str):
    if pd.isna(mech_str):
        return ""
    items = [s.strip() for s in mech_str.split(",")]
    broad_cats = set()
    for m in items:
        if m in ["Attention", "Cognitive flexibility", "Executive control", "Fluid intelligence", "Memory"]:
            broad_cats.add("Cognitive function")
        elif m in ["Anxiety", "Depression", "Happiness", "Stress"]:
            broad_cats.add("Mental health")
        elif m in ["Aspirations", "Internalized stigma", "Locus of control", "Self-efficacy", "Self-esteem"]:
            broad_cats.add("Beliefs")
        elif m in ["Time preferences", "Risk preferences"]:
            broad_cats.add("Preferences")
    return ", ".join(sorted(broad_cats))

# Prompt editor
with st.expander("Edit Analysis Prompt", expanded=False):
    prompt = st.text_area("Customize the analysis prompt", DEFAULT_PROMPT, height=400)

def download_pdf_from_scihub(doi_or_url, out_filename):
    for base_url in SCI_HUB_URLS:
        try:
            st.write(f"Trying {base_url} ...")
            resp = requests.get(base_url + doi_or_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.content, "html.parser")
            iframe = soup.find("iframe")
            if not iframe or not iframe.get("src"):
                continue
            pdf_url = iframe["src"]
            if pdf_url.startswith("//"):
                pdf_url = "https:" + pdf_url
            elif pdf_url.startswith("/"):
                pdf_url = base_url.rstrip("/") + pdf_url
            st.write(f"PDF URL found: {pdf_url}")
            pdf_resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
            if pdf_resp.status_code == 200 and pdf_resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
                with open(out_filename, "wb") as f:
                    f.write(pdf_resp.content)
                st.write(f"PDF downloaded successfully: {out_filename}")
                return out_filename
        except Exception as e:
            st.write(f"Error with {base_url}: {e}")
    st.write(f"Failed to download PDF for DOI: {doi_or_url}")
    return None

def parse_batch_response(response_text):
    """
    Parse the JSON response from Gemini with enhanced error handling and country validation.
    
    Args:
        response_text (str): Raw response text from Gemini
        
    Returns:
        list: Parsed results or empty list on error
    """
    try:
        # Try to parse the response as JSON directly
        response_data = json.loads(response_text)
        results = response_data.get('results', [])
        
        # Validate countries in the results
        for result in results:
            if 'analysis' in result:
                parts = result['analysis'].split(';')
                if len(parts) >= 5:  # Make sure we have enough parts
                    # Extract and validate countries
                    countries = parts[4].strip()
                    if ',' in countries:
                        # Multiple countries
                        country_list = [c.strip() for c in countries.split(',')]
                        if len(country_list) > 10:
                            validated_countries = "10+ countries"
                        else:
                            validated_countries = ','.join(validate_country(c) for c in country_list)
                    else:
                        # Single country
                        validated_countries = validate_country(countries)
                    
                    # Reconstruct the analysis string with validated countries
                    parts[4] = validated_countries
                    result['analysis'] = ';'.join(parts)
        
        return results
    except json.JSONDecodeError as e:
        try:
            # Try to extract JSON from the response if there's additional text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                response_data = json.loads(json_str)
                return parse_batch_response(json_str)  # Recursively parse with the extracted JSON
            else:
                # If no JSON found, try to parse the semicolon-separated format directly
                results = []
                lines = response_text.strip().split('\n')
                current_doi = None
                current_analysis = None
                
                for line in lines:
                    if 'DOI:' in line:
                        current_doi = line.split('DOI:')[1].strip()
                    elif ';' in line and current_doi:
                        # Parse and validate countries in the direct format
                        parts = line.strip().split(';')
                        if len(parts) >= 5:
                            countries = parts[4].strip()
                            if ',' in countries:
                                country_list = [c.strip() for c in countries.split(',')]
                                if len(country_list) > 10:
                                    validated_countries = "10+ countries"
                                else:
                                    validated_countries = ','.join(validate_country(c) for c in country_list)
                            else:
                                validated_countries = validate_country(countries)
                            parts[4] = validated_countries
                            current_analysis = ';'.join(parts)
                            
                            results.append({
                                'doi': current_doi,
                                'analysis': current_analysis
                            })
                        current_doi = None
                
                if results:
                    return results
                
                st.error(f"""
                Failed to parse response. Response received:
                ```
                {response_text[:500]}...
                ```
                """)
                return []
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            return []

def exponential_backoff(retry_count):
    """Calculate delay with exponential backoff and jitter"""
    delay = min(300, BASE_DELAY * (2 ** retry_count))  # Cap at 5 minutes
    jitter = random.uniform(0, 0.1 * delay)  # Add 0-10% jitter
    return delay + jitter

def display_request_stats():
    """Display current request statistics in the sidebar"""
    stats = st.session_state.request_tracker.get_stats()
    with st.sidebar:
        st.subheader("API Usage Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Requests", stats["total"])
            st.metric("Failed Requests", stats["failed"])
        with col2:
            st.metric("Successful Requests", stats["successful"])
            st.metric("Rate Limited", stats["rate_limited"])
        
        # Add progress bar for rate limits
        current_rpm = stats["current_rpm"]
        st.progress(current_rpm / REQUESTS_PER_MINUTE, f"Current Rate: {current_rpm}/{REQUESTS_PER_MINUTE} rpm")
        
        # Daily limit progress
        daily_progress = stats["total"] / REQUESTS_PER_DAY
        st.progress(daily_progress, f"Daily Usage: {stats['total']}/{REQUESTS_PER_DAY} requests")

# Main analysis section
if st.button("Start Analysis", type="primary") and uploaded_file is not None and api_key:
    # Create progress containers
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("Analysis Progress")
        progress_cols = st.columns([2, 1])
        with progress_cols[0]:
            progress_bar = st.progress(0)
        with progress_cols[1]:
            status_text = st.empty()
        
        details_expander = st.expander("Processing Details", expanded=True)
    
    try:
        # Display initial request stats
        display_request_stats()
        
        # Reset the file pointer to the beginning
        uploaded_file.seek(0)
        
        # Sample the dataframe
        try:
            df = pd.read_csv(uploaded_file)
            valid_df = df[df['doi'].notna() & df['abstract'].notna()]
            
            if selection_method == "Random Sample":
                sampled_df = valid_df.sample(n=sample_size, random_state=42)
            elif selection_method == "First N Papers":
                sampled_df = valid_df.head(sample_size)
            else:  # Process All
                sampled_df = valid_df.head(MAX_PAPERS)  # Limit to maximum allowed
            
        except Exception as e:
            st.error(f"Error processing the CSV file: {str(e)}")
            st.stop()

        # Initialize Gemini 2.0 Flash model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Calculate number of batches with new limits
        num_batches = math.ceil(len(sampled_df) / INITIAL_BATCH_SIZE)
        all_results = []
        start_time = time.time()
        
        # Add session state for tracking progress
        if 'processed_count' not in st.session_state:
            st.session_state.processed_count = 0
        
        # Process in batches with new rate limits
        for batch_idx in range(num_batches):
            start_idx = batch_idx * INITIAL_BATCH_SIZE
            end_idx = min((batch_idx + 1) * INITIAL_BATCH_SIZE, len(sampled_df))
            batch_df = sampled_df.iloc[start_idx:end_idx]
            
            # Update progress
            progress = (batch_idx + 1) / num_batches
            progress_bar.progress(progress)
            status_text.write(f"Batch {batch_idx + 1} of {num_batches}")
            
            with details_expander:
                st.write(f"Processing papers {start_idx + 1:,} to {end_idx:,}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Papers Processed", f"{end_idx:,}")
                with col2:
                    elapsed_time = time.time() - start_time
                    st.metric("Time Elapsed", f"{elapsed_time/60:.1f} min")
                with col3:
                    papers_per_minute = end_idx / (elapsed_time/60) if elapsed_time > 0 else 0
                    st.metric("Papers/Minute", f"{papers_per_minute:.1f}")
            
            # Process batch with new token limits
            batch_results = process_batch(batch_df, model, prompt, status_text)
            all_results.extend(batch_results)
            
            # Update processed count
            st.session_state.processed_count += sum(1 for r in batch_results if "ERROR" not in r['gemini_response'])
            
            # Add a shorter delay between batches due to higher rate limits
            if batch_idx < num_batches - 1:
                with details_expander:
                    st.write("Short pause between batches...")
                time.sleep(WAIT_TIME)  # 1 second between requests
            
            # Update request stats periodically during processing
            if batch_idx % 5 == 0:  # Update every 5 batches
                display_request_stats()
        
        # Create results DataFrame
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Process the results with detailed labels
            task_cols = ['task1', 'task2', 'task3', 'task4', 'task5', 'task6']
            results_df[task_cols] = results_df['gemini_response'].str.split(';', expand=True)
            
            # Clean whitespace
            for col in task_cols:
                results_df[col] = results_df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Map detailed labels
            results_df['Poverty_Context'] = results_df['task1'].apply(lambda x: map_labels(x, task1_map))
            results_df['Psychological_Mechanisms'] = results_df['task2'].apply(lambda x: map_labels(x, task2_map))
            results_df['Behaviors'] = results_df['task3'].apply(lambda x: map_labels(x, task3_map))
            results_df['Study_Type'] = results_df['task4'].apply(lambda x: map_labels(x, task4_map))
            results_df['Sample_Country'] = results_df['task5']
            results_df['Sample_Size'] = results_df['task6'].apply(lambda x: map_labels(x, task6_map))
            
            # Add broad categories
            results_df["Study_Type_Broad"] = results_df["Study_Type"].apply(get_broad_study_type)
            results_df["Psychological_Mechanisms_Broad"] = results_df["Psychological_Mechanisms"].apply(get_psych_mechanism_broad)
            
            # Drop intermediate task columns
            results_df.drop(columns=task_cols, inplace=True)
            
            # Merge with original CSV data
            uploaded_file.seek(0)  # Reset file pointer
            original_df = pd.read_csv(uploaded_file)
            
            # Merge the results with the original dataframe, keeping only analyzed papers
            final_df = original_df.merge(results_df, on='doi', how='inner')
            
            # Reorder columns to put analysis results after original columns
            analysis_cols = [
                "gemini_response",
                "Poverty_Context", "Psychological_Mechanisms_Broad", "Psychological_Mechanisms",
                "Behaviors", "Study_Type_Broad", "Study_Type",
                "Sample_Country", "Sample_Size"
            ]
            
            # Get original columns (excluding 'doi' since it's already first)
            original_cols = [col for col in original_df.columns if col != 'doi']
            
            # Create final column order
            final_cols = ['doi'] + original_cols + analysis_cols
            
            # Reorder columns
            final_df = final_df[final_cols]
            
            # Display results
            with results_container:
                st.header("Analysis Results")
                st.write(f"Successfully analyzed {len(final_df)} papers")
                st.dataframe(final_df)
                
                # Download buttons for results
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"gemini_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Update the rate limit usage display
                st.info(f"""
                Processing Statistics:
                - Total papers processed: {len(final_df):,}
                - API calls used: {num_batches:,} out of {REQUESTS_PER_DAY:,} daily limit
                - Average papers per request: {len(final_df) / num_batches:.1f}
                - Processing speed: {papers_per_minute:.1f} papers/minute
                - Total processing time: {(time.time() - start_time)/60:.1f} minutes
                """)
                
                # Summary metrics
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Papers Analyzed", f"{len(final_df):,}")
                with metric_cols[1]:
                    st.metric("Processing Time", f"{(time.time() - start_time)/60:.1f} min")
                with metric_cols[2]:
                    st.metric("API Calls Used", f"{num_batches:,}")
                with metric_cols[3]:
                    success_rate = len(final_df) / len(sampled_df) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.warning("No results were generated. Please check the entries and try again.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Display final request stats
        display_request_stats()
    
    finally:
        # Display final request stats
        display_request_stats()
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        # Reset session state
        if 'processed_count' in st.session_state:
            del st.session_state.processed_count

else:
    # Initial instructions
    if not uploaded_file:
        st.info("👆 Please upload a CSV file with 'doi' and 'abstract' columns to begin.")
    if not api_key:
        st.warning("🔑 Please enter your Gemini API key in the sidebar to begin.")

# Update the sidebar to show token estimation
if uploaded_file is not None:
    try:
        # First check if file is empty
        uploaded_file.seek(0)
        first_char = uploaded_file.read(1)
        if not first_char:
            st.sidebar.error("The uploaded file is empty. Please upload a valid CSV file.")
        else:
            # Reset file pointer and try to read
            uploaded_file.seek(0)
            try:
                # Try reading with different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        continue
                    except pd.errors.EmptyDataError:
                        st.sidebar.error("The CSV file appears to be empty or corrupted.")
                        break
                    except Exception as e:
                        continue
                
                # Check if DataFrame was successfully created
                if 'df' not in locals():
                    st.sidebar.error("Could not read the CSV file with any supported encoding.")
                    st.sidebar.info("Please ensure your file is a properly formatted CSV file.")
                elif df.empty:
                    st.sidebar.error("The CSV file contains no data.")
                elif 'abstract' not in df.columns:
                    st.sidebar.error("Required 'abstract' column not found in CSV.")
                    st.sidebar.info(f"Available columns: {', '.join(df.columns)}")
                else:
                    # Remove any completely empty rows
                    df = df.dropna(how='all')
                    
                    # Check for valid abstracts
                    valid_abstracts = df['abstract'].notna()
                    if not valid_abstracts.any():
                        st.sidebar.error("No valid abstracts found in the file.")
                    else:
                        # Calculate token estimates only for valid abstracts
                        valid_df = df[valid_abstracts].copy()
                        sample_size = min(5, len(valid_df))
                        if sample_size > 0:
                            sample_df = valid_df.sample(n=sample_size)
                            total_tokens = 0
                            token_counts = []
                            
                            for abstract in sample_df['abstract']:
                                try:
                                    tokens = count_tokens(str(abstract))
                                    token_counts.append(tokens)
                                    total_tokens += tokens
                                except Exception as e:
                                    st.sidebar.warning(f"Error counting tokens for an abstract: {str(e)}")
                            
                            if token_counts:
                                avg_tokens = total_tokens / len(token_counts)
                                max_tokens = max(token_counts)
                                min_tokens = min(token_counts)
                                
                                st.sidebar.info(f"""
                                Token Estimates (based on {sample_size} sample abstracts):
                                - Average tokens per abstract: {avg_tokens:.0f}
                                - Min tokens: {min_tokens}
                                - Max tokens: {max_tokens}
                                - Estimated max batch size: {math.floor(MAX_INPUT_TOKENS / (avg_tokens + 100)):.0f}
                                """)
                        else:
                            st.sidebar.warning("Not enough valid abstracts for token estimation.")
            
            except pd.errors.ParserError as e:
                st.sidebar.error(f"Error parsing CSV file: {str(e)}")
                st.sidebar.info("Please ensure your file is a properly formatted CSV file.")
            except Exception as e:
                st.sidebar.error(f"Unexpected error reading file: {str(e)}")
                st.sidebar.info("Please check the file format and try again.")
    except Exception as e:
        st.sidebar.error(f"Error processing file: {str(e)}")
        st.sidebar.info("Please ensure you've uploaded a valid CSV file.") 