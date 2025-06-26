import streamlit as st
import pandas as pd
import os 
import sys
import plotly.graph_objs as go
from streamlit_tree_select import tree_select
from dotenv import load_dotenv
import ast
import json
import psutil 
import time
import re
import gc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from visuals import bar, sankey, heatMap
from data.bigQuery import Client

st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

def log_memory(step_name: str) -> float:
    """
    Recommended replacement for the current log_memory function.
    Uses USS (Unique Set Size) when available for more accurate measurement.
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Use USS if available (Linux/macOS), fallback to RSS
        if hasattr(memory_info, 'uss'):
            memory_mb = memory_info.uss / 1024 / 1024
            metric = "USS"
        else:
            memory_mb = memory_info.rss / 1024 / 1024
            metric = "RSS"
        
        logger.info(f"[MEMORY] {step_name}: {memory_mb:.1f} MB ({metric})")
        
        # Optional: Add warning for high memory usage
        if memory_mb > 2000:  # 2GB threshold
            logger.warning(f"[MEMORY_HIGH] High memory usage detected: {memory_mb:.1f} MB")
        
        return memory_mb
        
    except Exception as e:
        logger.error(f"[MEMORY_ERROR] Failed to measure memory: {e}")
        return 0.0

# Log major filter changes
def log_filter_change(filter_type, count):
    logger.info(f"[FILTER] {filter_type}: {count} selected")
    log_memory(f"after_{filter_type}_change")

# Log major data operations
def log_data_op(operation, rows=None):
    if rows:
        logger.info(f"[DATA] {operation}: {rows} rows")
    else:
        logger.info(f"[DATA] {operation}")
    log_memory(f"after_{operation}")

def monitor_and_clear_cache():
    """Monitor memory usage and clear cache if needed"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Clear cache if memory usage exceeds 1.5GB
        if memory_mb > 900:
            logger.warning(f"[CACHE] Clearing cache - memory at {memory_mb:.1f}MB")
            st.cache_data.clear()
            st.cache_resource.clear()
            gc.collect()
            log_memory("after_cache_clear")
            return True
        return False
    except Exception:
        return False

def load_filters_json():
    """Load filters JSON once and cache it"""
    with open('data/trainingData/labels.json', 'r') as f:
        return json.load(f)

@st.cache_data(ttl=300, max_entries=1, show_spinner="Processing data...")  # Reduced from 300 to 300 seconds (kept same)
def preprocess_papers_data(papers_df):
    """One-time expensive preprocessing with categorical optimization"""
    log_data_op("preprocess_start", len(papers_df))
    
    df = papers_df.copy()
    
    # Pre-parse institutions (convert lists to strings for caching)
    df['institutions_list'] = df['institution'].apply(
        lambda x: str(list(dict.fromkeys([i for i in ast.literal_eval(str(x)) if i is not None])))
    )
    log_memory("after_institutions_parsing")
    
    # Pre-parse countries (convert lists to strings for caching)
    df['countries_list'] = df['country_of_study'].apply(
        lambda x: str([i.strip() for i in str(x).split(',') if i.strip() and i.strip().lower() != 'nan'])
    )
    log_memory("after_countries_parsing")
    
    df['institutions_list'] = df['institutions_list'].astype('category')
    df['countries_list'] = df['countries_list'].astype('category')
    log_memory("after_categorical_conversion")
    
    return df

# Cache expensive data processing operations 
@st.cache_data(ttl=300, max_entries=1, show_spinner="Loading Filters...")
def process_countries_and_institutions(preprocessed_df):
    """Process and extract unique countries and institutions from preprocessed dataset"""
    log_data_op("processing_locations", len(preprocessed_df))
    
    # Extract unique countries from string representations
    countries = []
    for country_str in preprocessed_df['countries_list']:
        country_list = ast.literal_eval(country_str)  # Convert string back to list
        countries.extend(country_list)
    countries = list(set(countries))
    countries = [str(country) for country in countries if country and str(country).lower() != 'nan']
    
    # Extract unique institutions from string representations
    institutions = []
    for inst_str in preprocessed_df['institutions_list']:
        inst_list = ast.literal_eval(inst_str)  # Convert string back to list
        institutions.extend(inst_list)
    institutions = list(set(institutions))
    institutions = [str(inst) for inst in institutions if inst]
    
    log_data_op("locations_processed", f"{len(countries)} countries, {len(institutions)} institutions")
    return sorted(countries), sorted(institutions)

def filtering(preprocessed_df, selected_country, selected_institution):
    """Super fast filtering using preprocessed data"""
    # Use view instead of copy to save memory
    result = preprocessed_df  # Start with original reference
    initial_rows = len(result)
    
    if selected_country != 'All':
        mask = result['countries_list'].apply(lambda x: selected_country in ast.literal_eval(x))
        result = result[mask].copy()  # Only copy when actually filtering
        logger.info(f"[FILTER] Country '{selected_country}': {initial_rows} -> {len(result)} rows")
        # Force cleanup of mask
        del mask
        gc.collect()
        log_memory("after_country_filter_with_cleanup")
    
    if selected_institution != 'All':
        pre_inst_rows = len(result)
        mask = result['institutions_list'].apply(lambda x: selected_institution in ast.literal_eval(x))
        result = result[mask].copy()  # Only copy when actually filtering
        logger.info(f"[FILTER] Institution '{selected_institution}': {pre_inst_rows} -> {len(result)} rows")
        # Force cleanup of mask
        del mask
        gc.collect()
        log_memory("after_institution_filter_with_cleanup")
    
    log_memory("after_geographic_filter")
    return result

@st.cache_data(ttl=300, max_entries=1, show_spinner="Loading Metadata...")
def calculate_statistics(filtered_df, papers_df, selected_country, selected_institution):
    """Calculate statistics for the filtered dataset"""
    # Filter publications based on the DOIs in the working dataframe
    filtered_publications = papers_df[papers_df['doi'].isin(filtered_df['doi'].tolist())]
    total_papers = len(filtered_publications)
    
    # Date range
    if not filtered_publications.empty:
        min_date = filtered_publications['date'].min()
        max_date = filtered_publications['date'].max()
        date_range = f"{min_date} – {max_date}"
    else:
        date_range = "No data available"
    
    # Countries count - use preprocessed data
    if selected_country != 'All':
        countries_count = 1
    else:
        all_countries = []
        for country_str in filtered_df['countries_list']:
            country_list = ast.literal_eval(country_str)  # Convert string back to list
            all_countries.extend(country_list)
        countries_count = len(set(all_countries))
    
    # Institutions count - use preprocessed data
    if selected_institution != 'All':
        institutions_count = 1
    else:
        all_institutions = []
        for inst_str in filtered_df['institutions_list']:
            inst_list = ast.literal_eval(inst_str)  # Convert string back to list
            all_institutions.extend(inst_list)
        institutions_count = len(set(all_institutions))
    
    return {
        'total_papers': total_papers,
        'date_range': date_range,
        'countries_count': countries_count,
        'institutions_count': institutions_count
    }

def get_filtered_data(selected_country, selected_institution):
    """Cache filtered results to avoid recomputation"""
    preprocessed_papers = get_preprocessed_papers()
    return filtering(preprocessed_papers, selected_country, selected_institution)

def get_exploded_sankey_data(
    working_df_dois,
    all_selected_context,
    all_selected_study_types,
    all_selected_mechanisms,
    all_selected_behaviors
):
    log_data_op("sankey_filter_start", len(working_df_dois))
    
    labels_data = get_labels_data()

    # Step 1: Initial DOI filter - add light cleanup
    sankey_df = labels_data[labels_data['doi'].isin(working_df_dois)]
    del labels_data  # Just this one cleanup
    log_memory("after_doi_filter")

    # Step 2: Pre-explosion filters
    def apply_pre_filter(df, col, selections):
        if selections:
            pattern = '|'.join([re.escape(val) for val in selections])
            return df[df[col].astype(str).str.contains(pattern, na=False, case=False)]
        return df
            
    sankey_df = apply_pre_filter(sankey_df, 'poverty_context', all_selected_context)
    sankey_df = apply_pre_filter(sankey_df, 'study_type', all_selected_study_types)
    sankey_df = apply_pre_filter(sankey_df, 'mechanism', all_selected_mechanisms)
    sankey_df = apply_pre_filter(sankey_df, 'behavior', all_selected_behaviors)
    log_memory("after_pre_filters")

    # Step 3: Convert relevant columns to string and explode
    def safe_explode(df, col):
        df[col] = df[col].astype(str).str.split(',').apply(lambda x: [v.strip() for v in x])
        return df.explode(col)

    for col, selections in [
        ('poverty_context', all_selected_context),
        ('mechanism', all_selected_mechanisms),
        ('study_type', all_selected_study_types),
        ('behavior', all_selected_behaviors)
    ]:
        sankey_df = safe_explode(sankey_df, col)
    log_memory("after_explosions")

    # Step 4: Final exact matching
    def exact_filter(df, col, selections):
        if selections:
            return df[df[col].isin(selections)]
        return df

    sankey_df = exact_filter(sankey_df, 'poverty_context', all_selected_context)
    sankey_df = exact_filter(sankey_df, 'study_type', all_selected_study_types)
    sankey_df = exact_filter(sankey_df, 'mechanism', all_selected_mechanisms)
    sankey_df = exact_filter(sankey_df, 'behavior', all_selected_behaviors)
    log_memory("after_exact_filters")
    
    # Step 5: Convert back to categorical or Arrow string
    for col in ['poverty_context', 'mechanism', 'study_type', 'behavior']:
        if col in sankey_df.columns:
            sankey_df[col] = pd.Series(pd.array(sankey_df[col], dtype="string[pyarrow]"))

    # Light cleanup at the end
    result = sankey_df.copy()
    del sankey_df
    gc.collect()
    
    log_data_op("sankey_filter_complete", len(result))
    log_memory("after_sankey_light_cleanup")
    return result

# Function to load environment variables with Streamlit compatibility
def load_environment_variables():
    """
    Load environment variables with fallback to Streamlit secrets
    """
    # Try to load from .env file first (for local development)
    load_dotenv()
    
    def get_env_var(key, default=None):
        """Get environment variable with fallback to Streamlit secrets"""
        # First try regular environment variables
        value = os.getenv(key)
        if value is not None:
            return value
        
        # Then try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
        # Return default or raise error
        if default is not None:
            return default
        else:
            st.error(f"Environment variable '{key}' not found. Please set it in your .env file or Streamlit secrets.")
            st.stop()
    
    return get_env_var

@st.cache_data(ttl=7200, max_entries=1, show_spinner="Loading Configuration...")
def get_configuration():
    """Load and cache configuration from environment variables"""
    get_env_var = load_environment_variables()
    
    # Fetch API key from environment variables
    api_key = get_env_var("GEMINI_API_KEY")
    
    # Load Google Sheets credentials from environment variables
    credentials = {
        "type": get_env_var("type"),
        "project_id": get_env_var("project_id"),
        "private_key_id": get_env_var("private_key_id"),
        "private_key": get_env_var("private_key").replace("\\n", "\n"),
        "client_email": get_env_var("client_email"),
        "client_id": get_env_var("client_id"),
        "auth_uri": get_env_var("auth_uri"),
        "token_uri": get_env_var("token_uri"),
        "auth_provider_x509_cert_url": get_env_var("auth_provider_x509_cert_url"),
        "client_x509_cert_url": get_env_var("client_x509_cert_url"),
        "universe_domain": get_env_var("universe_domain")
    }
    
    email = get_env_var("USERNAME")
    password = get_env_var("PASSWORD")
    
    return api_key, credentials, email, password

# Assign variables from cached configuration
api_key, credentials, email, password = get_configuration()

st.sidebar.markdown("""
         <style>
            .sidebar-icons {
                    position: absolute;
                    bottom: -40px;
                    right: 80;
                    display: flex;
                    flex-direction: column;
                    align-items: flex-left;
                            }
            .sidebar-icons img {
                    width: 30px; /* Adjust size as needed */
                    margin-bottom: 10px;
                            }
        </style>
                        """,
                        unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Connecting to Database...")
def get_bigquery_client():
    try:
        client = Client(credentials, 'literature-452020')
        # Test the connection immediately
        if not client._is_client_healthy():
            logger.error("Failed to establish healthy database connection")
            st.error("Failed to establish healthy database connection")
            st.stop()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

def get_healthy_bigquery_client():
    """Get a healthy BigQuery client, handling refresh automatically"""
    client = get_bigquery_client()
    
    # This will automatically refresh the connection if unhealthy
    client.get_healthy_client()
    return client

# ============================================================================
# UNIFIED DATA LOADER - REPLACES ALL SEPARATE LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300, max_entries=1, show_spinner="Connection to Database...")  # Reduced from 3600 to 600 seconds
def load_unified_papers_data():
    """
    Single function to load ALL paper data once (excluding abstracts for memory efficiency).
    Replaces: load_country_institution_data, load_label_data, load_umap
    """
    log_data_op("unified_data_load_start")
    
    try:
        client = get_healthy_bigquery_client()
        
        # ONE query to get everything EXCEPT abstracts (abstracts loaded on-demand)
        df = client.execute_query("""
            SELECT 
                doi, title, authors, date,
                country, institution, country_of_study,
                study_type, poverty_context, mechanism, behavior,
                UMAP1, UMAP2
            FROM `literature-452020.psychology_of_poverty_literature.papers`
        """)
        
        log_data_op("query_complete", len(df))
        
        # Convert to categorical ONCE for all relevant columns
        categorical_columns = [
            'country', 'institution', 'country_of_study', 
            'study_type', 'poverty_context', 'mechanism', 
            'behavior', 'authors'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        log_data_op("unified_data_load_complete", len(df))
        
        # Check memory after caching this large dataset
        log_memory("after_large_data_cached")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load unified data: {str(e)}")
        st.error(f"Failed to load unified data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# LIGHTWEIGHT DATA ACCESSORS - NO SEPARATE DATABASE CALLS
# ============================================================================

def get_papers_data():
    """Get geography/institution subset - NO separate database call"""
    df = load_unified_papers_data()
    return df[['doi', 'country', 'date', 'institution', 'country_of_study']]

def get_labels_data():
    """Get labels subset - NO separate database call"""
    df = load_unified_papers_data()
    return df[['doi', 'authors', 'study_type', 'poverty_context', 'mechanism', 'behavior']]

def get_umap_data():
    """Get UMAP subset - NO separate database call"""
    df = load_unified_papers_data()
    result = df[['title', 'doi', 'UMAP1', 'UMAP2', 'date']].copy()  # Explicit copy
    
    # Convert numeric columns once
    result.loc[:, 'UMAP1'] = pd.to_numeric(result['UMAP1'], errors='coerce')
    result.loc[:, 'UMAP2'] = pd.to_numeric(result['UMAP2'], errors='coerce')
    
    return result

def get_full_paper_data():
    """Get complete dataset when needed"""
    return load_unified_papers_data()

# ============================================================================
# KEEP TOPICS SEPARATE (DIFFERENT TABLE)
# ============================================================================

@st.cache_data(ttl=300, max_entries=1, show_spinner="Loading Heatmap Data...")
def load_topics():
    try:
        client = get_healthy_bigquery_client()
        result = client.execute_query(
            "SELECT * "
            "FROM `literature-452020.psychology_of_poverty_literature.topics`"
        )
        
        # Convert numeric columns once
        result.loc[:, 'umap_1_mean'] = pd.to_numeric(result['umap_1_mean'], errors='coerce')
        result.loc[:, 'umap_2_mean'] = pd.to_numeric(result['umap_2_mean'], errors='coerce')
        
        log_data_op("topics_loaded", len(result))
        return result
    except Exception as e:
        logger.error(f"Failed to load topics: {str(e)}")
        st.error(f"Failed to load topics: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# SEPARATE ABSTRACT LOADING - KEEP DATABASE QUERIES FOR MEMORY EFFICIENCY
# ============================================================================

@st.cache_data(ttl=300, max_entries=10, show_spinner="Loading Paper Details...")
def load_abstract_data(title):
    """Load abstract with separate database query - abstracts are too large for unified cache"""
    try:
        client = get_healthy_bigquery_client()
        # Handle all problematic characters
        safe_title = title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        
        result = client.execute_query(
            f"""
            SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE title = '{safe_title}'
            LIMIT 1
            """
        )
        log_data_op("abstract_loaded")
        return result
    except Exception as e:
        logger.error(f"Failed to load abstract data: {str(e)}")
        st.error(f"Failed to load abstract data: {str(e)}")
        return pd.DataFrame()

def get_working_df_exploded_cached(country, institution, contexts, study_types, mechanisms, behaviors):
    """Process on demand - now much faster due to pre-filtering"""
    working_df = get_filtered_data(country, institution)
    
    return get_exploded_sankey_data(
        working_df['doi'].tolist(),
        list(contexts),  # Convert tuples to lists
        list(study_types), 
        list(mechanisms),
        list(behaviors)
    )

def get_preprocessed_papers():
    papers_df = get_papers_data()
    return preprocess_papers_data(papers_df)

@st.fragment
def paper_details_fragment():
    """
    Isolated fragment for paper selection and details display.
    This will only rerun when interactions happen within this fragment,
    preventing the entire app from rerunning when selecting papers.
    """
    
    # Paper count display
    with st.expander("Number of Papers Visualized", expanded=True):
        paper_count = len(st.session_state.ui_state.get('current_papers_list', []))
        st.markdown(f"**{paper_count:,}**")

    # Paper selection and details
    with st.expander("Select Paper", expanded=True):
        available_papers = st.session_state.ui_state.get('current_papers_list', [])
        
        if available_papers:
            # Use a completely independent selectbox
            selected_paper = st.selectbox(
                "Select Paper", 
                available_papers,
                key="isolated_paper_selector",  # Unique key to avoid conflicts
                help="Select a paper to view detailed information"
            )

            # Only load abstract if paper changed (completely independent operation)
            if selected_paper != st.session_state.ui_state.get('current_selected_paper'):
                logger.info(f"[PAPER] Selected: {selected_paper[:50]}...")
                st.session_state.ui_state['current_selected_paper'] = selected_paper
                with st.spinner("Loading paper details..."):
                    st.session_state.ui_state['current_paper_details'] = load_abstract_data(selected_paper)
                log_memory("after_paper_selection")
            
            # Display cached paper details
            filtered_data = st.session_state.ui_state.get('current_paper_details', pd.DataFrame())

            if not filtered_data.empty:
                st.markdown("###### Paper Details")
                # Display the selected paper's details

                authors = filtered_data['authors'].to_list()
                authors = ast.literal_eval(authors[0])
                authors = [author for author in authors if author != 'Insufficient info']
                authors = ', '.join(authors)

                context = filtered_data['poverty_context'].to_list()
                context = list(set(context))
                context = [c for c in context if c != 'Insufficient info']
                if len(context) > 1:
                    context = ', '.join(context)
                elif len(context) == 1:
                    context = context[0]
                else:
                    context = "None"
                
                study_types = filtered_data['study_type'].to_list()
                study_types = [study for study in study_types if study != 'Insufficient info']
                study_types = list(set(study_types))
                if len(study_types) > 1:
                    study_types = ', '.join(study_types)
                elif len(study_types) == 1:
                    study_types = study_types[0]
                else:
                    study_types = "None"

                mechanisms =  filtered_data['mechanism'].to_list()
                mechanisms = [m for m in mechanisms if m != 'Insufficient info']
                mechanisms = list(set(mechanisms))

                behavior = filtered_data['behavior'].to_list()
                behavior = [b for b in behavior if b != 'Insufficient info']
                behavior = list(set(behavior))
                if len(behavior) > 1:
                    behavior = ', '.join(behavior)
                elif len(behavior) == 1:
                    behavior = behavior[0]
                else:
                    behavior = "None"

                if len(mechanisms) > 1:
                    mechanisms = ', '.join(mechanisms)
                elif len(mechanisms) == 1:
                    mechanisms = mechanisms[0]
                else:
                    mechanisms = "None"

                st.markdown(f"**Title:** {filtered_data['title'].values[0]}")
                st.markdown(f"**Authors:** {authors}")
                st.markdown(f"**Context:** {context}")
                st.markdown(f"**Study Type:** {study_types}")
                st.markdown(f"**Mechanism:** {mechanisms}")
                st.markdown(f"**Behavior:** {behavior}")
                st.markdown(f"**Abstract:** {filtered_data['abstract'].values[0]}")
            else:
                st.write("No details available for selected paper.")
        else:
            st.write("No papers available for visualization with current filters.")

from functools import lru_cache

@st.cache_data
def load_and_process_filters():
    """Cache the filter loading and tree building to avoid repeated computation"""
    filters = load_filters_json()
    
    # Pre-build trees once and cache them
    study_types_tree = build_tree_optimized(filters['study_types'])
    mechanisms_tree = build_tree_optimized(filters['mechanisms']) 
    behaviors_tree = build_tree_optimized(filters['Behaviors'])
    
    return filters, study_types_tree, mechanisms_tree, behaviors_tree

def build_tree_optimized(data, path=""):
    """Optimized tree builder - reduces string operations and memory allocation"""
    if not data:
        return []
    
    tree = []
    for key, value in data.items():
        node_path = f"{path} > {key}" if path else key
        
        if isinstance(value, dict):
            children = build_tree_optimized(value, node_path)
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
        elif isinstance(value, list):
            # Pre-allocate list and use list comprehension
            children = [
                {"label": item, "value": f"{node_path} > {item}"} 
                for item in value
            ]
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
    return tree

def process_tree_selections(selections, min_depth=2):
    """Optimized selection processing - reduces string splitting operations"""
    if not selections or not selections.get('checked'):
        return []
    
    result = []
    for value in selections['checked']:
        # Split once and reuse
        parts = value.split(' > ')
        if len(parts) >= min_depth:
            result.append(parts[min_depth - 1])
    return result

@st.fragment
def sankey_heatmap_fragment():
    """
    Fragment that includes filters, Sankey diagram, heatmap, and data processing
    """

    filters, study_types_tree, mechanisms_tree, behaviors_tree = load_and_process_filters()
        
    # Get current filter selections (lightweight check)
    selected_contexts = st.session_state.get("sankey_contexts", [])
    selected_study_types = st.session_state.get("sankey_study_types", {})
    selected_mechanisms = st.session_state.get("sankey_mechanisms", {})
    selected_behaviors = st.session_state.get("sankey_behaviors", {})
   
    # Process selections BEFORE signature calculation
    all_selected_context = []
    if selected_contexts:
        for key in selected_contexts:
            all_selected_context.extend(filters['poverty_contexts'][key])
    
    all_selected_study_types = process_tree_selections(selected_study_types, min_depth=3)
    all_selected_mechanisms = process_tree_selections(selected_mechanisms, min_depth=2)
    all_selected_behaviors = process_tree_selections(selected_behaviors, min_depth=2)

    previous_quick_signature = st.session_state.get('quick_filter_signature', 0)
   
    # Create signature using processed results only
    quick_signature = hash((
        tuple(sorted(all_selected_context)) if all_selected_context else (),
        tuple(sorted(all_selected_study_types)) if all_selected_study_types else (),
        tuple(sorted(all_selected_mechanisms)) if all_selected_mechanisms else (),
        tuple(sorted(all_selected_behaviors)) if all_selected_behaviors else (),
        st.session_state.ui_state.get('selected_country', 'All'),
        st.session_state.ui_state.get('selected_institution', 'All')
    ))
   
    if quick_signature == previous_quick_signature and previous_quick_signature != 0:
        # Skip expensive computation but continue with UI rendering
        pass
    
    st.session_state['quick_filter_signature'] = quick_signature

    with st.spinner("Applying Filters..."):
        # Load cached data       
        col1, col2 = st.columns([1, 6])
        
        with col1:
            # All your filter UI code
            st.markdown("###### Poverty Contexts")
            selected_contexts = st.multiselect(
                "Select", 
                list(filters['poverty_contexts'].keys()), 
                key="sankey_contexts"
            )
            
            st.markdown("###### Study Types") 
            selected_study_types = tree_select(study_types_tree, key="sankey_study_types")
            
            st.markdown("###### Psychological Mechanisms")
            selected_mechanisms = tree_select(mechanisms_tree, key="sankey_mechanisms")
            
            st.markdown("###### Behavioral Outcomes")
            selected_behaviors = tree_select(behaviors_tree, key="sankey_behaviors")
            
            # Process selections efficiently
            all_selected_context = []
            if selected_contexts:
                for key in selected_contexts:
                    all_selected_context.extend(filters['poverty_contexts'][key])
            
            all_selected_study_types = process_tree_selections(selected_study_types, min_depth=3)
            all_selected_mechanisms = process_tree_selections(selected_mechanisms, min_depth=2)
            all_selected_behaviors = process_tree_selections(selected_behaviors, min_depth=2)
            
            # Get exploded data (cached based on sankey filters)
            working_df_exploded = get_working_df_exploded_cached(
                st.session_state.ui_state['selected_country'],
                st.session_state.ui_state['selected_institution'],
                tuple(all_selected_context),
                tuple(all_selected_study_types),
                tuple(all_selected_mechanisms),
                tuple(all_selected_behaviors)
            )

            # Post-Sankey cleanup
            variables_to_cleanup = ['working_df', 'umap_data', 'labels_data']
            cleaned_count = 0
            for var_name in variables_to_cleanup:
                if var_name in locals():
                    del locals()[var_name]
                    cleaned_count += 1

            logger.info(f"[POST_SANKEY_CLEANUP] Deleted {cleaned_count} variables")

            # Multiple garbage collection passes to clean up Series objects
            for i in range(5):
                collected = gc.collect()
                if collected == 0:
                    break
                logger.debug(f"[POST_SANKEY_GC] Pass {i+1}: collected {collected} objects")

            log_memory("after_post_sankey_aggressive_cleanup")
            check_memory()

        with col2:
            with st.spinner("Generating Sankey Diagram..."):   
                # Create columns for the toggle chips
                col1_inner, col2_inner, col3_inner, col4_inner = st.columns(4)

                with col1_inner:
                    show_context = st.checkbox("Poverty Context", value=True, key="node_context")

                with col2_inner:
                    show_study = st.checkbox("Study Type", value=True, key="node_study")

                with col3_inner:
                    show_mechanism = st.checkbox("Psychological Mechanism", value=True, key="node_mechanism")

                with col4_inner:
                    show_behavior = st.checkbox("Behavioral Outcomes", value=True, key="node_behavior")

                # Build selected nodes list
                selected_nodes = []
                if show_context:
                    selected_nodes.append('poverty_context')
                if show_study:
                    selected_nodes.append('study_type')
                if show_mechanism:
                    selected_nodes.append('mechanism')
                if show_behavior:
                    selected_nodes.append('behavior')
                else: 
                    selected_nodes = ['poverty_context', 'study_type', 'mechanism', 'behavior']

                # Prepare active_filters dictionary for the sankey function
                active_filters = {
                    'contexts': selected_contexts if selected_contexts else [],
                    'study_types': all_selected_study_types if all_selected_study_types else [],
                    'mechanisms': all_selected_mechanisms if all_selected_mechanisms else [],
                    'behaviors': all_selected_behaviors if all_selected_behaviors else []
                }
                
                # Create and display the Sankey diagram with adaptive detail
                sankey_diagram = sankey.Sankey(filters_json = filters)

                if not working_df_exploded.empty:
                    sankey_fig = sankey_diagram.draw(working_df_exploded, 
                                                    active_filters=active_filters,
                                                    columns_to_show = selected_nodes)
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.write("No data available for selected filters.")

    # Get data for heatmap directly (no session state caching)
    umap_data = get_umap_data()
    plot_df = umap_data[umap_data['doi'].isin(working_df_exploded['doi'].tolist())].copy()
    topics_df = load_topics()
    
    # Update session state for paper details fragment only
    st.session_state.ui_state['current_papers_list'] = plot_df['title'].tolist() if not plot_df.empty else []

    st.markdown("#### Research Landscape")
    
    col1_heat, col2_heat = st.columns([2, 1])
    
    with col1_heat:
        # UMAP visualization
        if not plot_df.empty:
            with st.spinner("Generating research landscape visualization..."):
                heatmap_fig = heatMap.heatmap(plot_df, topics_df)
                st.plotly_chart(heatmap_fig.draw(), use_container_width=True)
        else:
            st.write("No data available for heatmap with current filters.")
        
        # Main explanation with better terminology
        st.markdown("""            
        This visualization creates a **living map** of academic research, where similar studies naturally cluster together like neighborhoods in a city. 
        Watch how knowledge evolves, new ideas emerge, and research communities form over time.
        """)
        
        # Enhanced instruction columns
        col4, col5 = st.columns(2)
        
        with col4:
            st.markdown("""
            #### **Reading the Map**
            
            **Research Papers (White Dots)**
            - Each dot represents one published study
            - Hover to see paper title and details
            - Position shows content similarity to other papers
            
            **Research Intensity (Color Heat)**
            - **Purple areas**: Sparse research, unexplored territories
            - **Green-blue areas**: Moderate research activity  
            - **Bright yellow peaks**: High-activity research hotspots
            
            **Topic Labels (White Text)**
            - Show major research themes and communities
            - Positioned at the center of each research cluster
            """)
        
        with col5:
            st.markdown("""
            #### **Interactive Controls**
            
            **Time Slider (Bottom)**
            - Drag to travel through research history
            - Watch clusters form, grow, split, and merge
            - Observe how topics gain or lose momentum
            
            **Exploration Tips**
            - **Identify trends**: Look for growing yellow areas
            - **Find gaps**: Purple spaces = research opportunities  
            - **Track evolution**: Follow clusters across years
            - **Spot emergence**: New clusters appearing at edges
            - **See convergence**: Separate topics moving together
            """)
        
        # Advanced insights section
        with st.expander("Advanced Insights", expanded=False):
            st.markdown("""

            #### **Advanced Insights**
            
            **Strategic Research Planning**
            - **Hot zones** (yellow): Competitive, well-established areas
            - **Transition zones** (green): Emerging opportunities with moderate competition
            - **Frontier zones** (purple): High-risk, high-reward unexplored areas
            
            **Temporal Patterns to Watch**
            - **Cluster growth**: Topics gaining academic attention
            - **Cluster migration**: Research focus shifting direction  
            - **Cluster fragmentation**: Fields becoming more specialized
            - **Cluster convergence**: Interdisciplinary collaboration increasing
            
            **Research Discovery**
            - Papers at cluster edges often represent innovative boundary work
            - Isolated papers may be ahead of their time or highly specialized
            - Dense cluster centers represent well-established, foundational work
            """)
        
        # Technical note
        with st.expander("Technical Details"):
            st.markdown("""
            **How it works:**
            - Papers are positioned using **UMAP** (Uniform Manifold Approximation and Projection)
            - Similar research content creates natural clustering patterns
            - Density estimation reveals research concentration patterns
            - Time animation shows cumulative research up to each year
            
            **Data processing:**
            - Each paper's abstract and metadata are converted to mathematical vectors
            - Dimensionality reduction projects high-dimensional similarity into 2D space
            - Gaussian density estimation creates smooth intensity surfaces
            """)

    with col2_heat:
        paper_details_fragment()
    
    # Clean up local variables to prevent memory leaks
    del umap_data, plot_df, topics_df, working_df_exploded
    gc.collect()
    log_memory("after_fragment_cleanup")

        
# Add this function to help debug cache issues
def debug_cache_sizes():
    """Debug function to check cache sizes"""
    try:
        # This will help us see what's in cache
        logger.debug("[CACHE_DEBUG] Checking cache contents...")
        
        # Check memory by object type
        import sys
        object_counts = {}
        total_size = 0
        
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            if obj_type not in object_counts:
                object_counts[obj_type] = {'count': 0, 'size': 0}
            object_counts[obj_type]['count'] += 1
            try:
                obj_size = sys.getsizeof(obj)
                object_counts[obj_type]['size'] += obj_size
                total_size += obj_size
            except:
                pass
        
        # Show top memory consumers
        sorted_objects = sorted(object_counts.items(), 
                            key=lambda x: x[1]['size'], reverse=True)[:10]
        
        logger.info(f"[MEMORY_DEBUG] Total tracked size: {total_size / 1024 / 1024:.1f} MB")
        logger.debug("[MEMORY_DEBUG] Top 10 memory consumers:")
        for obj_type, stats in sorted_objects:
            size_mb = stats['size'] / 1024 / 1024
            logger.debug(f"  {obj_type}: {stats['count']} objects, {size_mb:.1f} MB")
        
        # Force garbage collection and check memory
        collected = gc.collect()
        logger.debug(f"[CACHE_DEBUG] Garbage collected {collected} objects")
        log_memory("after_gc_debug")
        
    except Exception as e:
        logger.error(f"[CACHE_DEBUG] Error: {e}")

# Add cleanup function for main()
def cleanup_dataframes():
    """Clean up DataFrame variables to prevent memory leaks"""
    import gc
    import sys
    
    # Get current frame
    frame = sys._getframe(1)  # Get caller's frame
    local_vars = frame.f_locals
    
    # Find DataFrame variables and delete them
    to_delete = []
    for name, obj in local_vars.items():
        if hasattr(obj, 'dtypes') and hasattr(obj, 'columns'):  # It's a DataFrame
            to_delete.append(name)
    
    for name in to_delete:
        if name in local_vars:
            del local_vars[name]
    
    # Force garbage collection
    collected = gc.collect()
    logger.info(f"[CLEANUP] Deleted {len(to_delete)} DataFrames, collected {collected} objects")
    log_memory("after_dataframe_cleanup")

# Simple periodic memory check
def check_memory():
    """Quick memory check - call this occasionally"""
    log_memory("memory_check")
    debug_cache_sizes()  # Add cache debugging
    monitor_and_clear_cache()

# App start with aggressive cache clearing
logger.info("=" * 50)
logger.info("[APP] Streamlit application starting")

# Clear all caches at startup to prevent memory buildup
try:
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("[APP] Cleared all caches at startup")
except Exception as e:
    logger.error(f"[APP] Cache clear error: {e}")

# Force garbage collection at startup
collected = gc.collect()
logger.info(f"[APP] Garbage collected {collected} objects at startup")

log_memory("app_start_after_cleanup")

def main():  

    # with st.sidebar:
    #     st.markdown("### System Status")
        
    #     # Memory Usage
    #     try:
    #         process = psutil.Process()
    #         memory_mb = process.memory_info().rss / 1024 / 1024
            
    #         if memory_mb < 1500:
    #             st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="Healthy")
    #         else:
    #             st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="High", delta_color="inverse")
                
    #     except ImportError:
    #         st.metric("Memory Usage", "N/A", delta="Install psutil")

    monitor_and_clear_cache()

    try: 
        if 'connection_tested' not in st.session_state:
            client = get_healthy_bigquery_client()
            st.session_state.connection_tested = True
    except Exception as e:
        st.error("Database connection failed. Please refresh the page.")
        st.stop()


    # if st.sidebar.button("Test DB Connection"):
    #     client = get_healthy_bigquery_client()
    #     if client._is_client_healthy():
    #         st.sidebar.success("Database connection healthy")
    #     else:
    #         st.sidebar.error("Database connection failed")

  

    # Initialize persistent UI state in session state (small objects only!)
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = {
            'selected_country': 'All',
            'selected_institution': 'All',
            'sankey_filters': {
                'contexts': [],
                'study_types': [],
                'mechanisms': [], 
                'behaviors': []
            },
            'selected_paper': None,
            'stats_computed': False,
            'paper_details': pd.DataFrame()
        }

    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.title(" Psychology and Economics of Poverty Literature Review Dashboard")
    with col2:
        st.image("streamlit/logo.png", use_container_width=False)

    # Create tabs for workspace functionalities
    tab1, tab2= st.tabs(["Dashboard", "About" ])

    # Introduction Tab
    with tab2:
       
       st.markdown("""
                   
                    Research on the psychological consequences of poverty has emerged from diverse academic disciplines, each offering distinct perspectives and methodologies. However, this research has often developed in isolation, leading to fragmented insights and limited interdisciplinary dialogue. This lack of integration makes it challenging to develop a comprehensive understanding of poverty and its multifaceted effects.

                    To address this issue, we have developed an interactive literature review tool that enables researchers, policymakers, practitioners, and others interested to explore, analyze, and synthesize knowledge from this growing body of research.

                    ### Key Features of Our Tool

                    - **Research Exploration**: An interactive platform that allows users to navigate research findings in an intuitive manner.
                    - **Data Visualization**: Visualizations that help users identify trends, patterns, and gaps in the literature.
                    - **Continuous Updates**: Designed to evolve, incorporating new studies and findings as they emerge.

                    ### Our Mission

                    We aim to enhance the understanding of the psychological impacts of poverty and, when available, subsequent downstream impacts on behaviors in the real world. We seek to support evidence-based decision-making and provide a robust foundation for researchers and practitioners working to design effective interventions.

                    This project is being developed by the [Psychology and Economics of Poverty Initiative (PEP)](https://cega.berkeley.edu/collection/psychology-and-economics-of-poverty/) at the **Center for Effective Global Action (CEGA)** at Berkeley.

                    📥 [Click here to download the data](https://docs.google.com/spreadsheets/d/1npkoU3RmhnKTSKsk_BbXerrSrHtbmyaPXcjO8YSJlUI/edit?gid=1950861456#gid=1950861456)

                    ---

                    ### Research Team and Contact Info

                    ##### Faculty
                    - Supreet Kaur  
                    - Mahesh Srinivasan  
                    - Jiaying Zhao  
                    - Ye Rang Park  
                    - Yuen Ho  

                    ##### Postdocs
                    - Aarti Malik

                    ##### Research Assistants
                    - Jaysan Shah  
                    - Mangai Sundaram  
                    - Swathi Natarajan  

                    ##### Dashboard Visualization Team
                    - Muhammad Mudhar  
                    - Shufan Pan  
                    - Kristina Hallez  

                    Special thanks to **Kellie Hogue** at UC Berkeley's D-Lab and **Van Tran**.

                    ---
                    If you are a researcher or project manager interested in adapting this tool for your field, visit the **Documentation** tab to learn about the technology behind it and how to implement it in your research domain.

                    Comments, questions, ideas: Kristina Hallez, Senior Research Manager, khallez@berkeley.edu
                    
                   

         """)
       
    # Understanding the Field Tab
    with tab1:

        # # Get base data (cached - loads once)
        # if 'papers_df' and 'preprocessed_papers' not in st.session_state:
        #     papers_df = get_papers_data()
        #     preprocessed_papers = get_preprocessed_papers()
        #     st.session_state.papers_df = 'present'
        #     st.session_state.preprocessed_papers = 'present'

        papers_df = get_papers_data()
        preprocessed_papers = get_preprocessed_papers()
        log_memory("after_loading_base_data")
        check_memory()


        # Introduction
        st.markdown("""
            In the research landscape below, we offer a guided visual exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
            """)

        st.markdown("##### Data at a Glance")

        # Get countries/institutions (cached)
        countries, all_institutions = process_countries_and_institutions(preprocessed_papers)

        # Create layout with appropriate column widths
        col1, col2 = st.columns([1, 1.4])

        with col1:
            # Filtering expander
            with st.expander("Filter by Country and Institution", expanded=True):
                col3, col4 = st.columns(2)
                
                # Country filter dropdown
                with col3:
                    current_country_index = 0
                    if st.session_state.ui_state['selected_country'] in (['All'] + countries):
                        current_country_index = (['All'] + countries).index(st.session_state.ui_state['selected_country'])
                    
                    selected_country = st.selectbox(
                        "Filter by Country", 
                        ['All'] + countries,
                        index=current_country_index,
                        key="country_selector"
                    )
                
                # Only update if changed
                if selected_country != st.session_state.ui_state['selected_country']:
                    st.session_state.ui_state['selected_country'] = selected_country
                    st.session_state.ui_state['stats_computed'] = False  # Flag for recomputation
                
                # Get institutions for current country (this part can be cached too)
                temp_filtered_df = get_filtered_data(selected_country, 'All')
                
                # Institution filter dropdown (dependent on country selection)
                with col4:
                    # Get institutions for the selected country filter
                    if selected_country != 'All':
                        filtered_institutions = []
                        for inst_str in temp_filtered_df['institutions_list']:
                            inst_list = ast.literal_eval(inst_str)  # Convert string back to list
                            filtered_institutions.extend(inst_list)
                        institutions_list = sorted(list(set(filtered_institutions)))
                    else:
                        institutions_list = all_institutions
                    
                    # Ensure current selection is valid
                    current_institution = st.session_state.ui_state['selected_institution']
                    if current_institution not in (['All'] + institutions_list):
                        current_institution = 'All'
                        st.session_state.ui_state['selected_institution'] = 'All'
                    
                    current_institution_index = 0
                    if current_institution in (['All'] + institutions_list):
                        current_institution_index = (['All'] + institutions_list).index(current_institution)
                    
                    selected_institution = st.selectbox(
                        "Filter by Institution", 
                        ['All'] + institutions_list,
                        index=current_institution_index,
                        key="institution_selector"
                    )
                
                # Only update if changed
                if selected_institution != st.session_state.ui_state['selected_institution']:
                    st.session_state.ui_state['selected_institution'] = selected_institution
                    st.session_state.ui_state['stats_computed'] = False
            
            # Get working data (cached based on filters)
            working_df = get_filtered_data(
                st.session_state.ui_state['selected_country'], 
                st.session_state.ui_state['selected_institution']
            )

            if 'temp_filtered_df' in locals():
                del temp_filtered_df
                gc.collect()
                log_memory("after_temp_filter_cleanup")
            
            # Calculate stats (only if needed)
            if not st.session_state.ui_state['stats_computed']:
                st.session_state.ui_state['current_stats'] = calculate_statistics(
                    working_df, papers_df, 
                    st.session_state.ui_state['selected_country'], 
                    st.session_state.ui_state['selected_institution']
                )
                st.session_state.ui_state['stats_computed'] = True
            
            stats = st.session_state.ui_state['current_stats']
            
            # Display statistics (these won't reload unless filters change)
            with st.expander("Total Number of Papers", expanded=True):
                st.markdown(f"**{stats['total_papers']:,}**")
            
            with st.expander("Research Time Span", expanded=True):
                st.markdown(f"**{stats['date_range']}**")
            
            # Display countries and institutions statistics in two columns
            col3, col4 = st.columns(2)
            
            with col3:
                with st.expander("Countries ", expanded=True):
                    st.markdown(f"**{stats['countries_count']}**")
            
            with col4:
                with st.expander("Institutions", expanded=True):
                    st.markdown(f"**{stats['institutions_count']}**")

                    
        with col2:
            # Bar chart (only recomputes if working_df changes)
            with st.spinner("Generating Bar Chart..."):
                with st.expander(" ", expanded=True):
                    # Use preprocessed institutions list for counting
                    all_institutions_in_filtered = []
                    for inst_str in working_df['institutions_list']:
                        inst_list = ast.literal_eval(inst_str)  # Convert string back to list
                        all_institutions_in_filtered.extend(inst_list)
                    
                    # Count institutions
                    from collections import Counter
                    institution_counts = Counter(all_institutions_in_filtered)
                    
                    # Convert to DataFrame for plotting
                    top_institutions = pd.DataFrame([
                        {'institution': inst, 'count': count} 
                        for inst, count in institution_counts.most_common(10)
                    ])

                    st.markdown("###### Research Institutions -  Number of Publications")
                    if not top_institutions.empty:
                        institution_figure = bar.create(top_institutions, x_column='institution', y_column='count', title=None, coord_flip=True, height= 345)
                        st.plotly_chart(institution_figure, use_container_width=True)
                    else:
                        st.write("No data available for selected filters.")

        st.markdown("#### Connecting Poverty Context, Psychological Mechanisms and Behavior")
        st.markdown("""
        Use the filters below to customize the Sankey diagram. 
        Performance decreases when visualizing a large number of papers.
        """)
            
        # Filter and Sankey diagram fragment
        sankey_heatmap_fragment()
        check_memory()

        st.markdown("#### Research Landscape")
        col1, col2 = st.columns([2, 1])

    
        

        


        dataframes_to_cleanup = ['papers_df', 'preprocessed_papers', 'working_df_exploded', 
                           'plot_df', 'topics_df', 'umap_data']
        cleaned_count = 0
        for df_name in dataframes_to_cleanup:
            if df_name in locals():
                del locals()[df_name]
                cleaned_count += 1
        
        gc.collect()
        print(f"[MAIN_CLEANUP] Cleaned {cleaned_count} DataFrames at end of main()")
        check_memory()
        log_memory("main_function_exit")
            

if __name__ == "__main__":
    main()