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
import hashlib

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
    """Memory tracking utility"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        if hasattr(memory_info, 'uss'):
            memory_mb = memory_info.uss / 1024 / 1024
            metric = "USS"
        else:
            memory_mb = memory_info.rss / 1024 / 1024
            metric = "RSS"
        
        logger.info(f"[MEMORY] {step_name}: {memory_mb:.1f} MB ({metric})")
        
        if memory_mb > 2000:
            logger.warning(f"[MEMORY_HIGH] High memory usage detected: {memory_mb:.1f} MB")
        
        return memory_mb
        
    except Exception as e:
        logger.error(f"[MEMORY_ERROR] Failed to measure memory: {e}")
        return 0.0

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
        
        if memory_mb > 900:
            logger.warning(f"[CACHE] Clearing cache - memory at {memory_mb:.1f}MB")
            st.cache_data.clear()
            st.cache_resource.clear()
            # clear session state to free up memory
            if hasattr(st.session_state, 'ui_state'):
                st.session_state.ui_state.clear()
            if hasattr(st.session_state, 'cached_working_df'):
                del st.session_state['cached_working_df']
            if hasattr(st.session_state, 'cached_sankey_fig'):
                del st.session_state['cached_sankey_fig']
            if hasattr(st.session_state, 'cached_heatmap_data'):
                del st.session_state['cached_heatmap_data']
            gc.collect()
            log_memory("after_cache_clear")
            return True
        return False
    except Exception:
        return False

# ============================================================================
# CONFIGURATION AND CONNECTION SETUP
# ============================================================================

def load_environment_variables():
    """Load environment variables with fallback to Streamlit secrets"""
    load_dotenv()
    
    def get_env_var(key, default=None):
        value = os.getenv(key)
        if value is not None:
            return value
        
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
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
    
    api_key = get_env_var("GEMINI_API_KEY")
    
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

api_key, credentials, email, password = get_configuration()

@st.cache_resource(show_spinner="Connecting to Database...")
def get_bigquery_client():
    try:
        client = Client(credentials, 'literature-452020')
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
    client.get_healthy_client()
    return client

# ============================================================================
# CENTRALIZED QUERY EXECUTION FUNCTION
# ============================================================================

def execute_bigquery(sql_query, description="query", log_rows=True, show_progress=False, batch_info=None):
    """
    Centralized BigQuery execution with error handling, logging, and progress tracking.
    All database queries go through this function.
    """
    log_data_op(f"{description}_start")
    
    try:
        client = get_healthy_bigquery_client()
        
        # Show progress for batch operations
        if show_progress and batch_info:
            # progress_text = f"Processing batch {batch_info['current']}/{batch_info['total']} ({batch_info['count']} items)"
            with st.spinner(' '):
                result = client.execute_query(sql_query)
        else:
            result = client.execute_query(sql_query)
        
        if log_rows:
            log_data_op(f"{description}_complete", len(result))
        else:
            log_data_op(f"{description}_complete")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to execute {description}: {str(e)}")
        st.error(f"Failed to execute {description}: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# ON-DEMAND QUERY FUNCTIONS - REPLACE LARGE CACHED DATASETS
# ============================================================================

def query_geography_data(selected_country='All', selected_institution='All'):
    """Query geography data on-demand instead of loading entire dataset"""
    
    with st.spinner("Loading Statistics..."):
        # Build dynamic WHERE clause
        where_conditions = []
        if selected_country != 'All':
            safe_country = selected_country.replace("'", "\\'")
            where_conditions.append(f"REGEXP_CONTAINS(country_of_study, r'\\b{safe_country}\\b')")
        
        if selected_institution != 'All':
            safe_institution = selected_institution.replace("'", "\\'")
            where_conditions.append(f"REGEXP_CONTAINS(institution, r'\\b{safe_institution}\\b')")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query = f"""
            SELECT doi, country, date, institution, country_of_study
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE {where_clause}
        """
        
        return execute_bigquery(query, f"geography_data_{selected_country}_{selected_institution}")

def query_available_filters():
    """Query unique countries and institutions for filter dropdowns"""
    with st.spinner("Loading filters..."):
        query = """
            SELECT DISTINCT 
                country_of_study,
                institution
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE country_of_study IS NOT NULL 
              AND institution IS NOT NULL
              AND country_of_study != ''
              AND institution != ''
        """
        
        result = execute_bigquery(query, "available_filters")
        
        # Extract unique countries and institutions
        countries = set()
        institutions = set()
        
        for _, row in result.iterrows():
            # Parse countries
            country_list = [c.strip() for c in str(row['country_of_study']).split(',') 
                           if c.strip() and c.strip().lower() != 'nan']
            countries.update(country_list)
            
            # Parse institutions
            try:
                inst_list = ast.literal_eval(str(row['institution']))
                if isinstance(inst_list, list):
                    institutions.update([str(i) for i in inst_list if i])
            except:
                # Fallback for non-list format
                inst_list = [i.strip() for i in str(row['institution']).split(',')]
                institutions.update([i for i in inst_list if i])
        
        return sorted(list(countries)), sorted(list(institutions))

def query_umap_data(doi_list):
    """Query UMAP data for specific DOIs only"""
    if not doi_list:
        return pd.DataFrame()
    
    # Dynamic batch size calculation based on query complexity
    base_query = """
        SELECT title, doi, UMAP1, UMAP2, date
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE doi IN ('PLACEHOLDER')
          AND UMAP1 IS NOT NULL 
          AND UMAP2 IS NOT NULL
    """
    
    # Calculate safe batch size
    max_query_size = 1024 * 1000  # 1MB limit
    base_query_size = len(base_query)
    safety_margin = 5000  # 5KB safety margin for UMAP (simpler query)
    available_chars_for_dois = max_query_size - base_query_size - safety_margin
    
    # Sample DOIs to estimate character usage
    sample_dois = doi_list[:min(50, len(doi_list))]
    sample_doi_str = "', '".join([str(doi).replace("'", "\\'") for doi in sample_dois])
    chars_per_doi = len(sample_doi_str) / len(sample_dois) if sample_dois else 25
    
    # Calculate dynamic batch size
    batch_size = int(available_chars_for_dois / chars_per_doi)
    batch_size = max(2000, min(75000, batch_size))  # Between 2K and 75K for UMAP
    
    logger.info(f"[UMAP_BATCHING] Using batch size: {batch_size} for {len(doi_list)} DOIs")
    
    # Calculate total batches for progress tracking
    total_batches = (len(doi_list) + batch_size - 1) // batch_size
    
    # Create progress tracking for multi-batch operations only
    if total_batches > 1:
        st.markdown(" ")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    all_results = []
    
    for i in range(0, len(doi_list), batch_size):
        batch_dois = doi_list[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"[UMAP_BATCH] Processing batch {batch_num}/{total_batches} ({len(batch_dois)} DOIs)")
        
        # Update progress for multi-batch operations
        if total_batches > 1:
            progress = batch_num / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Loading Research Landscape Data")
        
        doi_list_str = "', '".join([str(doi).replace("'", "\\'") for doi in batch_dois])
        
        query = f"""
            SELECT title, doi, UMAP1, UMAP2, date
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE doi IN ('{doi_list_str}')
              AND UMAP1 IS NOT NULL 
              AND UMAP2 IS NOT NULL
        """
        
        try:
            batch_info = {'current': batch_num, 'total': total_batches, 'count': len(batch_dois)}
            batch_result = execute_bigquery(
                query, 
                f"umap_data_batch_{batch_num}",
                show_progress=(total_batches == 1),  # Only show spinner for single batch
                batch_info=batch_info
            )
            if not batch_result.empty:
                all_results.append(batch_result)
            logger.info(f"[UMAP_BATCH] Batch {batch_num} completed: {len(batch_result)} rows")
            
        except Exception as e:
            logger.error(f"[UMAP_BATCH] Batch {batch_num} failed: {str(e)}")
            continue
    
    # Clear progress indicators for multi-batch operations
    if total_batches > 1:
        progress_bar.progress(1.0)
        
        # Clean up progress UI after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    if all_results:
        result = pd.concat(all_results, ignore_index=True)
        # Convert numeric columns
        result['UMAP1'] = pd.to_numeric(result['UMAP1'], errors='coerce')
        result['UMAP2'] = pd.to_numeric(result['UMAP2'], errors='coerce')
        logger.info(f"[UMAP_BATCHING] Combined {len(all_results)} batches into {len(result)} total rows")
        return result
    
    return pd.DataFrame()

@st.cache_data(ttl=300, max_entries=1, show_spinner="Loading Topics...")
def query_topics_data():
    """Query topics data (can stay cached as it's small and static)"""
    query = "SELECT * FROM `literature-452020.psychology_of_poverty_literature.topics`"
    result = execute_bigquery(query, "topics_data")
    
    if not result.empty:
        result['umap_1_mean'] = pd.to_numeric(result['umap_1_mean'], errors='coerce')
        result['umap_2_mean'] = pd.to_numeric(result['umap_2_mean'], errors='coerce')
    
    return result

def query_sankey_data(doi_list, filter_contexts=None, filter_study_types=None, 
                     filter_mechanisms=None, filter_behaviors=None):
    """Query exploded Sankey data for specific DOIs with filters"""
    if not doi_list:
        return pd.DataFrame()
    
    # Build filter conditions
    filter_clauses = []
    
    if filter_contexts:
        context_list = "', '".join([ctx.replace("'", "\\'") for ctx in filter_contexts])
        filter_clauses.append(f"TRIM(poverty_context_item) IN ('{context_list}')")
    
    if filter_study_types:
        study_list = "', '".join([st.replace("'", "\\'") for st in filter_study_types])
        filter_clauses.append(f"TRIM(study_type_item) IN ('{study_list}')")
        
    if filter_mechanisms:
        mech_list = "', '".join([mech.replace("'", "\\'") for mech in filter_mechanisms])
        filter_clauses.append(f"TRIM(mechanism_item) IN ('{mech_list}')")
        
    if filter_behaviors:
        behavior_list = "', '".join([beh.replace("'", "\\'") for beh in filter_behaviors])
        filter_clauses.append(f"TRIM(behavior_item) IN ('{behavior_list}')")
    
    where_clause = " AND ".join(filter_clauses) if filter_clauses else "1=1"
    
    # Calculate maximum safe batch size automatically (your original dynamic logic)
    max_query_size = 1024 * 1000  # 1MB limit in characters
    
    # Calculate base query size (everything except DOI list)
    base_query_template = f"""
        SELECT 
          doi,
          TRIM(poverty_context_item) as poverty_context,
          TRIM(study_type_item) as study_type,
          TRIM(mechanism_item) as mechanism,
          TRIM(behavior_item) as behavior
        FROM (
          SELECT 
            doi,
            poverty_context,
            study_type,
            mechanism,
            behavior
          FROM `literature-452020.psychology_of_poverty_literature.papers`
          WHERE doi IN ('PLACEHOLDER')
        ),
        UNNEST(SPLIT(poverty_context, ',')) as poverty_context_item,
        UNNEST(SPLIT(study_type, ',')) as study_type_item,
        UNNEST(SPLIT(mechanism, ',')) as mechanism_item,
        UNNEST(SPLIT(behavior, ',')) as behavior_item
        WHERE poverty_context_item IS NOT NULL 
          AND study_type_item IS NOT NULL
          AND mechanism_item IS NOT NULL  
          AND behavior_item IS NOT NULL
          AND TRIM(poverty_context_item) != ''
          AND TRIM(study_type_item) != ''
          AND TRIM(mechanism_item) != ''
          AND TRIM(behavior_item) != ''
          AND TRIM(poverty_context_item) != 'Insufficient info'
          AND TRIM(study_type_item) != 'Insufficient info'
          AND TRIM(mechanism_item) != 'Insufficient info'
          AND TRIM(behavior_item) != 'Insufficient info'
          AND {where_clause}
        """
    
    base_query_size = len(base_query_template)
    safety_margin = 10000  # 10KB safety margin
    available_chars_for_dois = max_query_size - base_query_size - safety_margin
    
    # Sample a few DOIs to get accurate character count
    sample_dois = doi_list[:min(100, len(doi_list))]
    sample_doi_list = "', '".join([str(doi).replace("'", "\\'") for doi in sample_dois])
    chars_per_doi = len(sample_doi_list) / len(sample_dois) if sample_dois else 30
    
    # Calculate maximum batch size
    batch_size = int(available_chars_for_dois / chars_per_doi)
    
    # Apply reasonable bounds
    batch_size = max(1000, min(50000, batch_size))  # Between 1K and 50K
    
    logger.info(f"[SANKEY_BATCHING] Using batch size: {batch_size} for {len(doi_list)} DOIs")
    
    # Calculate total batches for progress tracking
    total_batches = (len(doi_list) + batch_size - 1) // batch_size
    
    # Create progress tracking for multi-batch operations only
    if total_batches > 1:
        st.markdown(" ")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Split DOIs into batches
    all_results = []
    
    for i in range(0, len(doi_list), batch_size):
        batch_dois = doi_list[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        logger.info(f"[SANKEY_BATCH] Processing batch {batch_num}/{total_batches} ({len(batch_dois)} DOIs)")
        
        # Update progress for multi-batch operations
        if total_batches > 1:
            progress = batch_num / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Loading Sankey Data")
        
        # Convert batch DOI list to SQL-friendly format
        doi_list_str = "', '".join([str(doi).replace("'", "\\'") for doi in batch_dois])
        
        query = f"""
        SELECT 
          doi,
          TRIM(poverty_context_item) as poverty_context,
          TRIM(study_type_item) as study_type,
          TRIM(mechanism_item) as mechanism,
          TRIM(behavior_item) as behavior
        FROM (
          SELECT 
            doi,
            poverty_context,
            study_type,
            mechanism,
            behavior
          FROM `literature-452020.psychology_of_poverty_literature.papers`
          WHERE doi IN ('{doi_list_str}')
        ),
        UNNEST(SPLIT(poverty_context, ',')) as poverty_context_item,
        UNNEST(SPLIT(study_type, ',')) as study_type_item,
        UNNEST(SPLIT(mechanism, ',')) as mechanism_item,
        UNNEST(SPLIT(behavior, ',')) as behavior_item
        WHERE poverty_context_item IS NOT NULL 
          AND study_type_item IS NOT NULL
          AND mechanism_item IS NOT NULL  
          AND behavior_item IS NOT NULL
          AND TRIM(poverty_context_item) != ''
          AND TRIM(study_type_item) != ''
          AND TRIM(mechanism_item) != ''
          AND TRIM(behavior_item) != ''
          AND TRIM(poverty_context_item) != 'Insufficient info'
          AND TRIM(study_type_item) != 'Insufficient info'
          AND TRIM(mechanism_item) != 'Insufficient info'
          AND TRIM(behavior_item) != 'Insufficient info'
          AND {where_clause}
        """
        
        try:
            batch_result = execute_bigquery(query, f"sankey_data_batch_{batch_num}")
            if not batch_result.empty:
                all_results.append(batch_result)
            logger.info(f"[SANKEY_BATCH] Batch {batch_num} completed: {len(batch_result)} rows")
            
        except Exception as e:
            logger.error(f"[SANKEY_BATCH] Batch {batch_num} failed: {str(e)}")
            # Continue with other batches instead of failing completely
            continue
    
    # Clear progress indicators for multi-batch operations
    if total_batches > 1:
        progress_bar.progress(1.0)
        
        # Clean up progress UI after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
    
    if all_results:
        final_result = pd.concat(all_results, ignore_index=True)
        logger.info(f"[SANKEY_BATCHING] Combined {len(all_results)} batches into {len(final_result)} total rows")
        return final_result
    
    return pd.DataFrame()

@st.cache_data(ttl=300, max_entries=10, show_spinner="Loading Paper Details...")
def query_paper_details(title):
    """Query individual paper details by title"""
    safe_title = title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    
    query = f"""
        SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE title = '{safe_title}'
        LIMIT 1
    """
    
    return execute_bigquery(query, "paper_details")

# ============================================================================
# FILTER AND UI HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300, max_entries=1)
def load_filters_json():
    """Load filters JSON"""
    with open('data/trainingData/labels.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_and_process_filters():
    """Cache the filter loading and tree building"""
    filters = load_filters_json()
    
    study_types_tree = build_tree_optimized(filters['study_types'])
    mechanisms_tree = build_tree_optimized(filters['mechanisms']) 
    behaviors_tree = build_tree_optimized(filters['Behaviors'])
    
    return filters, study_types_tree, mechanisms_tree, behaviors_tree

def build_tree_optimized(data, path=""):
    """Optimized tree builder"""
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
    """Process tree selections"""
    if not selections or not selections.get('checked'):
        return []
    
    result = []
    for value in selections['checked']:
        parts = value.split(' > ')
        if len(parts) >= min_depth:
            result.append(parts[min_depth - 1])
    return result

# ============================================================================
# FRAGMENT FUNCTIONS
# ============================================================================

@st.fragment
def render_filters():
    """Renders the filter UI and processes selections"""
    filters, study_types_tree, mechanisms_tree, behaviors_tree = load_and_process_filters()
    
    _render_filter_ui(filters, study_types_tree, mechanisms_tree, behaviors_tree)
    
    processed_selections = _process_current_selections(filters)
    _update_session_state(processed_selections)

def _render_filter_ui(filters, study_types_tree, mechanisms_tree, behaviors_tree):
    """Renders all filter UI components"""
    st.markdown("###### Poverty Contexts")
    st.multiselect(
        "Select",
        list(filters['poverty_contexts'].keys()),
        key="sankey_contexts"
    )
    
    st.markdown("###### Study Types")
    tree_select(study_types_tree, key="sankey_study_types")
    
    st.markdown("###### Psychological Mechanisms")
    tree_select(mechanisms_tree, key="sankey_mechanisms")
    
    st.markdown("###### Behavioral Outcomes")
    tree_select(behaviors_tree, key="sankey_behaviors")

def _process_current_selections(filters):
    """Processes current UI selections into usable format"""
    selected_contexts = st.session_state.get("sankey_contexts", [])
    selected_study_types = st.session_state.get("sankey_study_types", {})
    selected_mechanisms = st.session_state.get("sankey_mechanisms", {})
    selected_behaviors = st.session_state.get("sankey_behaviors", {})
    
    all_selected_context = []
    for context_key in selected_contexts:
        all_selected_context.extend(filters['poverty_contexts'][context_key])
    
    all_selected_study_types = process_tree_selections(selected_study_types, min_depth=3)
    all_selected_mechanisms = process_tree_selections(selected_mechanisms, min_depth=2)
    all_selected_behaviors = process_tree_selections(selected_behaviors, min_depth=2)
    
    return {
        'contexts': all_selected_context,
        'study_types': all_selected_study_types,
        'mechanisms': all_selected_mechanisms,
        'behaviors': all_selected_behaviors
    }

def _update_session_state(processed_selections):
    """Updates session state with new selections"""
    for key, value in processed_selections.items():
        st.session_state.ui_state[key] = value

def _create_sankey_signature():
    data = {
        'contexts': sorted(st.session_state.ui_state.get('contexts', [])),
        'study_types': sorted(st.session_state.ui_state.get('study_types', [])),
        'mechanisms': sorted(st.session_state.ui_state.get('mechanisms', [])),
        'behaviors': sorted(st.session_state.ui_state.get('behaviors', [])),
        'country': st.session_state.ui_state.get('selected_country', 'All'),
        'institution': st.session_state.ui_state.get('selected_institution', 'All')
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

# ============================================================================
# UPDATED SANKEY FRAGMENT WITH SEQUENCING
# ============================================================================

@st.fragment(run_every=10)
def load_sankey():
    """Loads and displays the Sankey diagram - RUNS FIRST"""
    current_signature = _create_sankey_signature()
    previous_signature = st.session_state.get('sankey_data_signature')
    
    data_changed = current_signature != previous_signature
    if data_changed:
        st.session_state['sankey_data_signature'] = current_signature
        # Mark that sankey is processing
        st.session_state['sankey_processing'] = True
        st.session_state['sankey_ready'] = False
        logger.info("[SEQUENCE] Sankey processing started")

    # Memory cleanup only if data changed
    if data_changed:
        gc.collect()
        log_memory("after_sankey_cleanup")

    # Show spinner only if data changed
    spinner_context = st.spinner("Loading Sankey Data...") if data_changed else st.empty()
    
    with spinner_context:   
        # Toggle controls - only render if data changed or first time
        controls_changed = False
        
        # Check if controls have changed
        current_controls = {
            'context': st.session_state.get("node_context", True),
            'study': st.session_state.get("node_study", True), 
            'mechanism': st.session_state.get("node_mechanism", True),
            'behavior': st.session_state.get("node_behavior", True)
        }
        
        previous_controls = st.session_state.get('sankey_controls_state', {})
        controls_changed = current_controls != previous_controls
        
        if data_changed or controls_changed or 'sankey_controls_rendered' not in st.session_state:
            col1_inner, col2_inner, col3_inner, col4_inner = st.columns(4)

            with col1_inner:
                show_context = st.checkbox("Poverty Context", value=True, key="node_context")
            with col2_inner:
                show_study = st.checkbox("Study Type", value=True, key="node_study")
            with col3_inner:
                show_mechanism = st.checkbox("Psychological Mechanism", value=True, key="node_mechanism")
            with col4_inner:
                show_behavior = st.checkbox("Behavioral Outcomes", value=True, key="node_behavior")
            
            st.session_state['sankey_controls_state'] = current_controls
            st.session_state['sankey_controls_rendered'] = True
        else:
            # Use cached control values
            show_context = st.session_state.get("node_context", True)
            show_study = st.session_state.get("node_study", True)
            show_mechanism = st.session_state.get("node_mechanism", True)
            show_behavior = st.session_state.get("node_behavior", True)

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
        
        if not selected_nodes:
            selected_nodes = ['poverty_context', 'study_type', 'mechanism', 'behavior']

        # Get filter selections
        all_selected_contexts = st.session_state.ui_state.get('contexts', [])
        all_selected_study_types = st.session_state.ui_state.get('study_types', [])
        all_selected_mechanisms = st.session_state.ui_state.get('mechanisms', [])
        all_selected_behaviors = st.session_state.ui_state.get('behaviors', [])

        active_filters = {
            'contexts': all_selected_contexts,
            'study_types': all_selected_study_types,
            'mechanisms': all_selected_mechanisms,
            'behaviors': all_selected_behaviors
        }

        # Check if we need to regenerate the chart
        chart_signature = {
            'data_signature': current_signature,
            'selected_nodes': selected_nodes,
            'active_filters': active_filters
        }
        
        previous_chart_signature = st.session_state.get('sankey_chart_signature')
        chart_needs_update = chart_signature != previous_chart_signature

        # Only regenerate data if signature changed
        if data_changed:
            logger.info("[SEQUENCE] Sankey data processing...")
            # Get geography data first
            geography_df = query_geography_data(
                st.session_state.ui_state['selected_country'],
                st.session_state.ui_state['selected_institution']
            )
            
            # Get sankey data for those DOIs
            working_df_exploded = query_sankey_data(
                geography_df['doi'].tolist(),
                all_selected_contexts,
                all_selected_study_types,
                all_selected_mechanisms,
                all_selected_behaviors
            )
            
            st.session_state['cached_working_df'] = working_df_exploded
            # Mark sankey processing as complete
            st.session_state['sankey_processing'] = False
            st.session_state['sankey_ready'] = True
            logger.info("[SEQUENCE] Sankey processing completed")
        else:
            working_df_exploded = st.session_state.get('cached_working_df', pd.DataFrame())
            # Ensure sankey is marked as ready even when data hasn't changed
            st.session_state['sankey_ready'] = True
            st.session_state['sankey_processing'] = False

        # Create and display Sankey diagram - only if chart needs update
        if not working_df_exploded.empty:
            if chart_needs_update or 'cached_sankey_fig' not in st.session_state:
                logger.info("[SANKEY] Regenerating chart due to changes")
                filters, _, _, _ = load_and_process_filters()
                sankey_diagram = sankey.Sankey(filters_json=filters)
                
                sankey_fig = sankey_diagram.draw(
                    working_df_exploded, 
                    active_filters=active_filters,
                    columns_to_show=selected_nodes
                )
                st.session_state['cached_sankey_fig'] = sankey_fig
                st.session_state['sankey_chart_signature'] = chart_signature
            
            # Always use the cached figure with a stable key
            st.plotly_chart(
                st.session_state['cached_sankey_fig'], 
                use_container_width=True, 
                key="sankey_chart_stable"
            )
            st.session_state['current_working_df'] = working_df_exploded
        else:
            st.write("No data available for selected filters.")
            st.session_state['current_working_df'] = None

# ============================================================================
# UPDATED HEATMAP FRAGMENT WITH DEPENDENCY WAITING
# ============================================================================

@st.fragment(run_every=10)
def render_heatmap():
    """Renders the heatmap section - WAITS FOR SANKEY TO COMPLETE"""
    
    # Wait for sankey to be ready
    if not st.session_state.get('sankey_ready', False):
        st.info("🔄 Preparing data filters...")
        return
        
    # Check if sankey is still processing
    if st.session_state.get('sankey_processing', False):
        st.info("🔄 Processing filters...")
        return
    
    logger.info("[SEQUENCE] Heatmap processing started")
    
    working_df_exploded = st.session_state.get('current_working_df', None)
    
    if working_df_exploded is None or working_df_exploded.empty:
        st.info("No data available for visualization with current filters.")
        return

    # Check for changes
    current_signature = _create_sankey_signature()
    previous_signature = st.session_state.get('heatmap_data_signature')
    
    data_changed = current_signature != previous_signature
    if data_changed:
        st.session_state['heatmap_data_signature'] = current_signature
        logger.info("[SEQUENCE] Heatmap data changed, regenerating...")

    # Get heatmap data only if changed
    if data_changed or 'cached_heatmap_data' not in st.session_state:
        # Check if any Sankey filters are applied
        has_sankey_filters = (
            st.session_state.ui_state.get('contexts', []) or
            st.session_state.ui_state.get('study_types', []) or
            st.session_state.ui_state.get('mechanisms', []) or
            st.session_state.ui_state.get('behaviors', [])
        )
        
        if has_sankey_filters:
            # Use filtered Sankey data
            unique_dois = working_df_exploded['doi'].unique().tolist()
            logger.info(f"[SEQUENCE] Using Sankey-filtered DOIs: {len(unique_dois)}")
        else:
            # Use all DOIs from geography filters (country/institution) directly
            geography_df = query_geography_data(
                st.session_state.ui_state['selected_country'],
                st.session_state.ui_state['selected_institution']
            )
            unique_dois = geography_df['doi'].tolist()
            logger.info(f"[SEQUENCE] Using geography-filtered DOIs: {len(unique_dois)}")
        
        plot_df = query_umap_data(unique_dois)
        topics_df = query_topics_data()
        
        st.session_state['cached_heatmap_data'] = {
            'plot_df': plot_df,
            'topics_df': topics_df
        }
        
        if 'cached_heatmap_fig' in st.session_state:
            del st.session_state['cached_heatmap_fig']
    else:
        cached_data = st.session_state.get('cached_heatmap_data', {})
        plot_df = cached_data.get('plot_df', pd.DataFrame())
        topics_df = cached_data.get('topics_df', pd.DataFrame())

    st.session_state.ui_state['current_papers_list'] = plot_df['title'].tolist() if not plot_df.empty else []

    st.markdown("#### Research Landscape")
    
    col1_heat, col2_heat = st.columns([2, 1])
    
    with col1_heat:
        if not plot_df.empty:
            if data_changed or 'cached_heatmap_fig' not in st.session_state:
                with st.spinner("Generating research landscape visualization..."):
                    heatmap_fig = heatMap.heatmap(plot_df, topics_df)
                    st.session_state['cached_heatmap_fig'] = heatmap_fig.draw()
            
            st.plotly_chart(st.session_state['cached_heatmap_fig'], use_container_width=True)
        else:
            st.write("No data available for heatmap with current filters.")
        
        st.markdown("""            
        This visualization creates a **living map** of academic research, where similar studies naturally cluster together like neighborhoods in a city. 
        Watch how knowledge evolves, new ideas emerge, and research communities form over time.
        """)
        
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
    
    logger.info("[SEQUENCE] Heatmap processing completed")

@st.fragment
def paper_details_fragment():
    """Paper selection and details display fragment"""
    
    with st.expander("Number of Papers Visualized", expanded=True):
        paper_count = len(st.session_state.ui_state.get('current_papers_list', []))
        st.markdown(f"**{paper_count:,}**")

    with st.expander("Select Paper", expanded=True):
        available_papers = st.session_state.ui_state.get('current_papers_list', [])
        
        if available_papers:
            selected_paper = st.selectbox(
                "Select Paper", 
                available_papers,
                key="isolated_paper_selector",
                help="Select a paper to view detailed information"
            )

            if selected_paper != st.session_state.ui_state.get('current_selected_paper'):
                logger.info(f"[PAPER] Selected: {selected_paper[:50]}...")
                st.session_state.ui_state['current_selected_paper'] = selected_paper
                with st.spinner("Loading paper details..."):
                    st.session_state.ui_state['current_paper_details'] = query_paper_details(selected_paper)
                log_memory("after_paper_selection")
            
            filtered_data = st.session_state.ui_state.get('current_paper_details', pd.DataFrame())

            if not filtered_data.empty:
                st.markdown("###### Paper Details")

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

                mechanisms = filtered_data['mechanism'].to_list()
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

# ============================================================================
# MAIN APPLICATION WITH SEQUENCING INITIALIZATION
# ============================================================================

def main():  
    monitor_and_clear_cache()

    # Test database connection
    try: 
        if 'connection_tested' not in st.session_state:
            client = get_healthy_bigquery_client()
            st.session_state.connection_tested = True
    except Exception as e:
        st.error("Database connection failed. Please refresh the page.")
        st.stop()

    # Initialize UI state
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = {
            'selected_country': 'All',
            'selected_institution': 'All',
            'contexts': [],
            'study_types': [],
            'mechanisms': [],
            'behaviors': [],
            'selected_paper': None,
            'stats_computed': False,
            'current_stats': {},
            'paper_details': pd.DataFrame(),
            'current_papers_list': [],
            'current_selected_paper': None,
            'current_paper_details': pd.DataFrame()
        }

    # Initialize sequencing state
    if 'sankey_ready' not in st.session_state:
        st.session_state['sankey_ready'] = False
        st.session_state['sankey_processing'] = False
        logger.info("[SEQUENCE] Sequencing state initialized")

    # Header
    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.title("Psychology and Economics of Poverty Literature Review Dashboard")
    with col2:
        st.image("streamlit/logo.png", use_container_width=False)

    # Create tabs
    tab1, tab2 = st.tabs(["Dashboard", "About"])

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

    with tab1:
        st.markdown("""
        In the research landscape below, we offer a guided visual exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
        """)

        st.markdown("##### Data at a Glance")

        # Get available filters for dropdowns
        countries, all_institutions = query_available_filters()

        # Ensure stats_computed is initialized
        if 'stats_computed' not in st.session_state.ui_state:
            st.session_state.ui_state['stats_computed'] = False

        col1, col2 = st.columns([1, 1.4])

        with col1:
            # Geography filters
            with st.expander("Filter by Country and Institution", expanded=True):
                col3, col4 = st.columns(2)
                
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
                
                if selected_country != st.session_state.ui_state['selected_country']:
                    st.session_state.ui_state['selected_country'] = selected_country
                    st.session_state.ui_state['stats_computed'] = False
                    # Reset sequencing when geography changes
                    st.session_state['sankey_ready'] = False
                    logger.info("[SEQUENCE] Geography filter changed - resetting sequence")

                with col4:
                    # Filter institutions based on selected country
                    if selected_country != 'All':
                        temp_geo_data = query_geography_data(selected_country, 'All')
                        filtered_institutions = set()
                        for _, row in temp_geo_data.iterrows():
                            try:
                                inst_list = ast.literal_eval(str(row['institution']))
                                if isinstance(inst_list, list):
                                    filtered_institutions.update([str(i) for i in inst_list if i])
                            except:
                                inst_list = [i.strip() for i in str(row['institution']).split(',')]
                                filtered_institutions.update([i for i in inst_list if i])
                        institutions_list = sorted(list(filtered_institutions))
                    else:
                        institutions_list = all_institutions
                    
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
                
                if selected_institution != st.session_state.ui_state['selected_institution']:
                    st.session_state.ui_state['selected_institution'] = selected_institution
                    st.session_state.ui_state['stats_computed'] = False
                    # Reset sequencing when geography changes
                    st.session_state['sankey_ready'] = False
                    logger.info("[SEQUENCE] Institution filter changed - resetting sequence")
            
            # Get working data for statistics
            if not st.session_state.ui_state['stats_computed']:
                working_df = query_geography_data(
                    st.session_state.ui_state['selected_country'], 
                    st.session_state.ui_state['selected_institution']
                )
                
                # Calculate statistics
                total_papers = len(working_df)
                
                if not working_df.empty:
                    min_date = working_df['date'].min()
                    max_date = working_df['date'].max()
                    date_range = f"{min_date} – {max_date}"
                else:
                    date_range = "No data available"
                
                # Count countries and institutions
                countries_count = 1 if st.session_state.ui_state['selected_country'] != 'All' else len(countries)
                institutions_count = 1 if st.session_state.ui_state['selected_institution'] != 'All' else len(all_institutions)
                
                st.session_state.ui_state['current_stats'] = {
                    'total_papers': total_papers,
                    'date_range': date_range,
                    'countries_count': countries_count,
                    'institutions_count': institutions_count
                }
                st.session_state.ui_state['stats_computed'] = True
            
            stats = st.session_state.ui_state['current_stats']
            
            # Display statistics
            with st.expander("Total Number of Papers", expanded=True):
                st.markdown(f"**{stats['total_papers']:,}**")
            
            with st.expander("Research Time Span", expanded=True):
                st.markdown(f"**{stats['date_range']}**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                with st.expander("Countries ", expanded=True):
                    st.markdown(f"**{stats['countries_count']}**")
            
            with col4:
                with st.expander("Institutions", expanded=True):
                    st.markdown(f"**{stats['institutions_count']}**")

        with col2:
            # Bar chart of top institutions
            with st.spinner("Generating Bar Chart..."):
                with st.expander(" ", expanded=True):
                    # Get current working data for bar chart
                    working_df = query_geography_data(
                        st.session_state.ui_state['selected_country'], 
                        st.session_state.ui_state['selected_institution']
                    )
                    
                    # Count institutions
                    all_institutions_in_filtered = []
                    for _, row in working_df.iterrows():
                        try:
                            inst_list = ast.literal_eval(str(row['institution']))
                            if isinstance(inst_list, list):
                                all_institutions_in_filtered.extend([str(i) for i in inst_list if i])
                        except:
                            inst_list = [i.strip() for i in str(row['institution']).split(',')]
                            all_institutions_in_filtered.extend([i for i in inst_list if i])
                    
                    from collections import Counter
                    institution_counts = Counter(all_institutions_in_filtered)
                    
                    top_institutions = pd.DataFrame([
                        {'institution': inst, 'count': count} 
                        for inst, count in institution_counts.most_common(10)
                    ])

                    st.markdown("###### Research Institutions - Number of Publications")
                    if not top_institutions.empty:
                        institution_figure = bar.create(
                            top_institutions, 
                            x_column='institution', 
                            y_column='count', 
                            title=None, 
                            coord_flip=True, 
                            height=345
                        )
                        st.plotly_chart(institution_figure, use_container_width=True)
                    else:
                        st.write("No data available for selected filters.")

        st.markdown("#### Connecting Poverty Context, Psychological Mechanisms and Behavior")
        st.markdown("""
        Use the filters below to customize the Sankey diagram. 
        Performance decreases when visualizing a large number of papers.
        """)

        # Sankey section - RUNS FIRST
        col1, col2 = st.columns([1, 6])

        with col1:
            render_filters()

        with col2:
            load_sankey()

        # Heatmap section - WAITS FOR SANKEY TO COMPLETE
        render_heatmap()

        # Cleanup
        gc.collect()
        log_memory("main_function_exit")

# ============================================================================
# APP STARTUP
# ============================================================================

# App startup with cache clearing
logger.info("=" * 50)
logger.info("[APP] Streamlit application starting")

try:
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("[APP] Cleared all caches at startup")
except Exception as e:
    logger.error(f"[APP] Cache clear error: {e}")

collected = gc.collect()
logger.info(f"[APP] Garbage collected {collected} objects at startup")
log_memory("app_start_after_cleanup")

if __name__ == "__main__":
    main()