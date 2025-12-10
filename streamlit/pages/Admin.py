import streamlit as st
import os
import sys 
parent_dir = os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(parent_dir)
import pandas as pd
from data.ETL import main
from datetime import datetime
from dotenv import load_dotenv
from data.bigQuery import Client
import json
import time
import psutil
import gc
import logging

# ============================================================================
# LOGGING SYSTEM - Comprehensive Memory and Operation Tracking
# ============================================================================

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_memory(step_name, include_details=False):
    """Enhanced memory logging for admin page"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_vms_mb = memory_info.vms / 1024 / 1024
        
        # Log basic memory info
        logger.info(f"MEMORY [{step_name}]: RSS={memory_mb:.1f}MB, VMS={memory_vms_mb:.1f}MB")
        
        if include_details:
            # Get object counts by type
            object_counts = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            
            # Log top 5 object types
            top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"OBJECTS [{step_name}]: Top objects: {top_objects}")
        
        # Log session state size if available
        if hasattr(st, 'session_state') and st.session_state:
            session_size = len(st.session_state)
            logger.info(f"SESSION [{step_name}]: {session_size} items")
        
        print(f"[ADMIN_MEMORY] {step_name}: {memory_mb:.1f} MB")
        return memory_mb
    except Exception as e:
        logger.error(f"Failed to log memory for {step_name}: {e}")
        return 0

def log_operation(operation_name, *args, **kwargs):
    """Log admin operations with parameters"""
    args_str = ', '.join([str(arg)[:50] for arg in args])  # Truncate long args
    kwargs_str = ', '.join([f"{k}={str(v)[:50]}" for k, v in kwargs.items()])
    logger.info(f"OPERATION [{operation_name}]: args=({args_str}), kwargs=({kwargs_str})")
    log_memory(f"start_{operation_name}")

def log_operation_complete(operation_name, success=True, result=None):
    """Log operation completion"""
    status = "SUCCESS" if success else "FAILED"
    result_str = str(result)[:100] if result else "None"
    logger.info(f"OPERATION_COMPLETE [{operation_name}]: {status}, result={result_str}")
    log_memory(f"end_{operation_name}")

def log_database_operation(operation_type, query_preview=None, rows_affected=None):
    """Log database operations"""
    query_str = query_preview[:100] if query_preview else "No query"
    logger.info(f"DATABASE [{operation_type}]: query='{query_str}...', rows={rows_affected}")
    log_memory(f"db_{operation_type}")

def log_user_action(action, details=None):
    """Log user actions"""
    details_str = f", details={details}" if details else ""
    logger.info(f"USER_ACTION [{action}]{details_str}")
    log_memory(f"action_{action}")

def log_cache_operation(operation, cache_type, hit_or_miss=""):
    """Log cache operations"""
    logger.info(f"CACHE [{operation}] {cache_type}: {hit_or_miss}")
    log_memory(f"cache_{operation}")

def log_performance_metrics():
    """Log detailed performance metrics"""
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        # Get cache sizes
        cache_info = {
            'session_state_items': len(st.session_state) if hasattr(st, 'session_state') else 0,
        }
        
        logger.info(f"PERFORMANCE: CPU={cpu_percent}%, Memory={memory_info.rss/1024/1024:.1f}MB, Cache={cache_info}")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cache_items': cache_info['session_state_items']
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {}

def monitor_and_clear_cache():
    """Enhanced memory monitoring with aggressive cleanup for admin page"""
    log_operation("monitor_and_clear_cache")
    
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        logger.info(f"CACHE_MONITOR: Current memory {memory_mb:.1f}MB")
        
        # Lower threshold for admin page since it's not the main app
        if memory_mb > 1500:
            logger.warning(f"CACHE_CLEAR: Memory threshold exceeded ({memory_mb:.1f}MB > 1500MB)")
            st.warning(f"Memory usage is high: {memory_mb:.0f} MB. Clearing cache to free up memory.")
            
            # Clear all caches
            st.cache_data.clear()
            st.cache_resource.clear()
            log_cache_operation("CLEAR", "all_caches")
            
            # Clear session state except critical data
            critical_keys = ['authenticated']
            temp_storage = {key: st.session_state.get(key) for key in critical_keys if key in st.session_state}
            cleared_items = len(st.session_state)
            st.session_state.clear()
            st.session_state.update(temp_storage)
            
            logger.info(f"SESSION_CLEAR: Cleared {cleared_items} items, kept {len(temp_storage)} critical")
            
            # Force garbage collection
            collected = gc.collect()
            logger.info(f"GC_COLLECT: Collected {collected} objects")
            
            log_operation_complete("monitor_and_clear_cache", True, f"cleared_{cleared_items}_items")
            return True
        
        log_operation_complete("monitor_and_clear_cache", True, "no_clear_needed")
        return False
    except Exception as e:
        logger.error(f"CACHE_ERROR: {e}")
        log_operation_complete("monitor_and_clear_cache", False, str(e))
        return False

def cleanup_session_data():
    """Clean up temporary session data after operations"""
    log_operation("cleanup_session_data")
    
    cleanup_keys = [
        'papers', 'data_fetched', 'selected_paper_temp',
        'processing_complete', 'last_fetch_time'
    ]
    
    cleaned_count = 0
    for key in cleanup_keys:
        if key in st.session_state:
            del st.session_state[key]
            cleaned_count += 1
    
    logger.info(f"SESSION_CLEANUP: Cleaned {cleaned_count} temporary items")
    
    # Force garbage collection after cleanup
    collected = gc.collect()
    logger.info(f"GC_AFTER_CLEANUP: Collected {collected} objects")
    
    log_operation_complete("cleanup_session_data", True, f"cleaned_{cleaned_count}_items")

# App initialization logging
logger.info("=" * 50)
logger.info("ADMIN_APP_START: Streamlit admin application starting")
log_memory("app_initialization")

# Run memory monitoring at the start
monitor_and_clear_cache()

# Function to load environment variables with Streamlit compatibility
def load_environment_variables():
    """
    Load environment variables with fallback to Streamlit secrets
    """
    log_operation("load_environment_variables")
    
    # Try to load from .env file first (for local development)
    load_dotenv()
    
    def get_env_var(key, default=None):
        """Get environment variable with fallback to Streamlit secrets"""
        # First try regular environment variables
        value = os.getenv(key)
        if value is not None:
            logger.info(f"ENV_VAR: Found {key} in environment")
            return value
        
        # Then try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                logger.info(f"ENV_VAR: Found {key} in Streamlit secrets")
                return st.secrets[key]
        except Exception as e:
            logger.error(f"ENV_VAR: Error accessing Streamlit secrets for {key}: {e}")
        
        # Return default or raise error
        if default is not None:
            logger.warning(f"ENV_VAR: Using default value for {key}")
            return default
        else:
            logger.error(f"ENV_VAR: {key} not found anywhere")
            st.error(f"Environment variable '{key}' not found. Please set it in your .env file or Streamlit secrets.")
            st.stop()
    
    log_operation_complete("load_environment_variables", True)
    return get_env_var

# Initialize environment variable getter
get_env_var = load_environment_variables()

# Set page config
st.set_page_config(page_title="Database Updater", layout="wide")

# Load all environment variables using the new function with TTL
@st.cache_data(ttl=1800, max_entries=1)  # 30 minute TTL, max 1 entry
def get_configuration():
    """Load and cache configuration from environment variables"""
    log_operation("get_configuration")
    log_cache_operation("ACCESS", "configuration")
    
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

    # Load Google Sheets IDs from environment variables
    spreadsheet_ids = {
        "papers": get_env_var("papers_spreadsheet_id"),
        "density": get_env_var("density"),
        "density_X": get_env_var("X"),
        "density_Y": get_env_var("Y"),
        "topics": get_env_var("topics_spreadsheet_id")
    }

    email = get_env_var("USERNAME")
    password = get_env_var("PASSWORD")
    
    log_operation_complete("get_configuration", True, "all_config_loaded")
    return api_key, credentials, spreadsheet_ids, email, password

def get_categories_and_keys(data):
    log_operation("get_categories_and_keys", f"data_keys={len(data) if data else 0}")
    
    result = {}

    def recursive_search(current_data):
        if isinstance(current_data, dict):
            innermost_keys = []
            for key, value in current_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    innermost_keys.extend(recursive_search(value))
                else:
                    innermost_keys.append(value)
            return innermost_keys
        elif isinstance(current_data, list):
            return current_data
        else:
            return [current_data]

    for broad_category, sub_data in data.items():
        result[broad_category] = recursive_search(sub_data)

    log_operation_complete("get_categories_and_keys", True, f"categories={len(result)}")
    return result

# Get configuration
api_key, credentials, spreadsheet_ids , email, password = get_configuration()
log_memory("after_configuration_load")

# Load log data with caching and TTL
@st.cache_data(ttl=300, max_entries=1)  # 5 minute TTL for log data
def load_log_data():
    log_operation("load_log_data")
    log_cache_operation("ACCESS", "log_data")
    
    try:
        with open('data/log.json', 'r') as f:
            data = json.load(f)
        log_operation_complete("load_log_data", True, f"entries={len(data)}")
        return data
    except Exception as e:
        logger.error(f"LOG_DATA_ERROR: {e}")
        log_operation_complete("load_log_data", False, str(e))
        return {}

# Load labels with caching and TTL  
@st.cache_data(ttl=1800, max_entries=1)  # 30 minute TTL for labels
def load_labels():
    log_operation("load_labels")
    log_cache_operation("ACCESS", "labels")
    
    try:
        with open('data/trainingData/labels.json', 'r') as f:
            data = json.load(f)
        log_operation_complete("load_labels", True, f"categories={len(data)}")
        return data
    except Exception as e:
        logger.error(f"LABELS_ERROR: {e}")
        log_operation_complete("load_labels", False, str(e))
        return {}

log_data = load_log_data()
labels = load_labels()
log_memory("after_data_load")

# enter username password otherwise page isnt authorized
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    log_user_action("login_attempt", "showing_login_form")
    
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        with st.form("Login"):
            st.markdown("  ")
            st.markdown("  ")
            
            st.write("This page is protected. Please enter credentials to access.")
            st.markdown("  ")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            st.markdown("  ")

            submitted = st.form_submit_button("Login")
            if submitted:
                log_user_action("login_submit", f"username={username[:3]}***")
                
                # Get credentials from environment/secrets
                expected_username = get_env_var("USERNAME")
                expected_password = get_env_var("PASSWORD")
                
                if username == expected_username and password == expected_password:
                    st.session_state['authenticated'] = True
                    log_user_action("login_success", f"user={username}")
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    log_user_action("login_failed", f"user={username}")
                    st.error("Invalid username or password")

            st.markdown("  ")
            st.markdown("  ")
    
    st.stop()

# User successfully authenticated
log_user_action("authenticated_access", f"session_authenticated=True")

# Initialize ETL Pipeline with TTL and memory optimization
@st.cache_resource(ttl=1800, max_entries=1)  # 30 minute TTL, max 1 entry
def initialize_pipeline():
    """Initialize and cache the ETL pipeline with automatic cleanup"""
    log_operation("initialize_pipeline")
    log_cache_operation("ACCESS", "etl_pipeline")
    
    try:
        pipeline = main.ETLPipeline(
            api_key=api_key,
            credentials_json=credentials,
            project_id='literature-452020',
        )
        log_operation_complete("initialize_pipeline", True, "pipeline_created")
        return pipeline
    except Exception as e:
        logger.error(f"ETL_PIPELINE_ERROR: {e}")
        st.error(f"Failed to initialize ETL pipeline: {str(e)}")
        log_operation_complete("initialize_pipeline", False, str(e))
        return None

@st.dialog("Confirm Deletion")
def confirm_delete_dialog(paper_title, papers_df, client, key):
    """
    Confirmation dialog for paper deletion
    """
    log_user_action("delete_dialog_shown", f"paper={paper_title[:50]}...")
    
    st.write(f"Are you sure you want to delete this paper?")
    st.markdown(f"**Title:** {paper_title}")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{key}", use_container_width=True):
            log_user_action("delete_cancelled", f"paper={paper_title[:50]}...")
            st.rerun()
    with col3:
        if st.button("🗑️ Delete", key=f"confirm_{key}", type="primary", use_container_width=True):
            log_user_action("delete_confirmed", f"paper={paper_title[:50]}...")
            
            # Show spinner during deletion
            with st.spinner("Deleting paper from database..."):
                try:
                    # Delete the paper from BigQuery using parameterized query
                    paper_data = papers_df.iloc[0]
                    title = paper_data["title"].replace("'", "''")  # Escape single quotes
                    doi = paper_data.get("doi", "").replace("'", "''")  # Escape single quotes
                    
                    delete_query = f"""
                        DELETE FROM `literature-452020.psychology_of_poverty_literature.papers`
                        WHERE title = '{title}' AND doi = '{doi}'
                    """
                    
                    log_database_operation("DELETE", delete_query[:100], "1_paper")
                    
                    client.execute_query(delete_query)
                    
                    log_user_action("paper_deleted_success", f"title={title[:50]}...")
                    st.success("Paper deleted successfully!")
                    
                    # Clean up session state after successful deletion
                    cleanup_session_data()
                    st.rerun()
                except Exception as e:
                    logger.error(f"DELETE_PAPER_ERROR: {e}")
                    log_user_action("paper_delete_failed", f"error={str(e)[:100]}")
                    st.error(f"Error deleting paper: {str(e)}")

def display_paper_details_optimized(papers_df, key=None, show_delete=False, client=None):
    """
    Optimized paper details display with memory management
    """
    log_operation("display_paper_details", f"papers_count={len(papers_df)}", f"show_delete={show_delete}")
    
    # Limit papers displayed to prevent memory issues
    if len(papers_df) > 50:
        papers_df = papers_df.head(50)
        logger.warning(f"PAPERS_DISPLAY: Limited to 50 papers from {len(papers_df)} total")
        st.warning("Showing first 50 papers only to prevent memory issues.")
    
    # Drop down to select paper
    with st.expander("Human Review", expanded=True):
        # Delete button at the top if enabled
        if show_delete:
            # Create columns for spacing and delete button
            delete_col, spacer_col = st.columns([4, 1])
            with delete_col:
                if st.button("🗑️ Delete", type="secondary", key=f"delete_{key}"):
                    # We need to get the selected paper first
                    if 'selected_paper_temp' in st.session_state:
                        selected_paper = st.session_state['selected_paper_temp']
                        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
                        log_user_action("delete_button_clicked", f"paper={selected_paper[:50]}...")
                        confirm_delete_dialog(
                            selected_paper,
                            selected_paper_details,
                            client,
                            key
                        )
        
        # Select paper dropdown with improved styling
        selected_paper = st.selectbox(" ", papers_df['title'].unique(), key=f"select_{key}")
        # Store selected paper in session state for delete function
        st.session_state['selected_paper_temp'] = selected_paper
        log_user_action("paper_selected", f"paper={selected_paper[:50]}...")
        
        # Display the selected paper's details
        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
        
        # Efficient formatting without creating copies
        paper_data = selected_paper_details.iloc[0]
        
        def format_list_field(field_value):
            if isinstance(field_value, list):
                return ', '.join(str(x) for x in field_value)
            return str(field_value)
        
        # Display paper information efficiently
        st.markdown(f"**Title:** {paper_data.get('title', 'N/A')}")
        st.markdown(f"**Study Type:** {format_list_field(paper_data.get('study_type', 'N/A'))}")
        st.markdown(f"**Context:** {format_list_field(paper_data.get('poverty_context', 'N/A'))}")
        st.markdown(f"**Mechanism:** {format_list_field(paper_data.get('mechanism', 'N/A'))}")
        st.markdown(f"**Behavior:** {format_list_field(paper_data.get('behavior', 'N/A'))}")
        st.markdown(f"**Authors:** {format_list_field(paper_data.get('authors', 'N/A'))}")
        
        # Truncate abstract if too long to save memory
        abstract = paper_data.get('abstract', 'N/A')
        if len(str(abstract)) > 1000:
            abstract = str(abstract)[:1000] + "..."
        st.markdown(f"**Abstract:** {abstract}")
    
    log_operation_complete("display_paper_details", True, f"displayed_paper={selected_paper[:50]}...")

# UPDATED: Cache BigQuery client with health check capability and TTL
@st.cache_resource(show_spinner='Connecting to database...', ttl=900, max_entries=1)  # 15 minute TTL
def get_bigquery_client():
    log_operation("get_bigquery_client")
    log_cache_operation("ACCESS", "bigquery_client")
    
    try:
        client = Client(credentials, 'literature-452020')
        log_memory("after_client_creation")
        
        # Test the connection immediately
        if not client._is_client_healthy():
            logger.error("BIGQUERY: Client health check failed")
            st.error("Failed to establish healthy database connection")
            st.stop()
        
        log_operation_complete("get_bigquery_client", True, "healthy_client")
        return client
    except Exception as e:
        logger.error(f"BIGQUERY_ERROR: {e}")
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# NEW: Add healthy client function for automatic refresh
def get_healthy_bigquery_client():
    """Get a healthy BigQuery client, handling refresh automatically"""
    log_operation("get_healthy_bigquery_client")
    
    client = get_bigquery_client()
    
    # This will automatically refresh the connection if unhealthy
    client.get_healthy_client()
    
    log_operation_complete("get_healthy_bigquery_client", True, "client_refreshed")
    return client

# NEW: Safe query execution with better error handling
def execute_safe_query(query_description, query):
    """Execute a query safely with proper error handling and user feedback"""
    log_operation("execute_safe_query", query_description, f"query_length={len(query)}")
    log_database_operation("EXECUTE", query[:100])
    
    try:
        client = get_healthy_bigquery_client()
        result = client.execute_query(query)
        
        result_info = f"rows={len(result)}" if hasattr(result, '__len__') else "query_executed"
        log_database_operation("SUCCESS", query[:100], result_info)
        log_operation_complete("execute_safe_query", True, result_info)
        return result, None
    except Exception as e:
        error_msg = f"Error {query_description}: {str(e)}"
        logger.error(f"QUERY_ERROR: {error_msg}")
        st.error(error_msg)
        log_operation_complete("execute_safe_query", False, str(e))
        return None, error_msg

# NEW: Escape function for SQL injection protection
def escape_sql_string(value):
    """Escape single quotes in SQL strings to prevent injection"""
    if value is None:
        return ""
    escaped = str(value).replace("'", "''")
    logger.debug(f"SQL_ESCAPE: {str(value)[:50]} -> {escaped[:50]}")
    return escaped

# Initialize connection and health monitoring
if 'connection_tested' not in st.session_state:
    log_user_action("db_connection_test", "initial_test")
    get_bigquery_client()
    st.session_state.connection_tested = True
    st.session_state.last_health_check = time.time()
    log_memory("after_initial_db_connection")

# Enhanced System Monitor Section
with st.sidebar:
    st.markdown("### System Status")
    
    # Memory Usage with lower thresholds for admin
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb < 800:
            st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="Healthy")
        elif memory_mb < 1200:
            st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="Moderate", delta_color="off")
        else:
            st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="High", delta_color="inverse")
            
    except ImportError:
        st.metric("Memory Usage", "N/A", delta="Install psutil")

    # Database Status
    current_time = time.time()
    if ('last_health_check' not in st.session_state or
        current_time - st.session_state.last_health_check > 300):
        
        log_user_action("db_health_check", "periodic_check")
        try:
            client = get_healthy_bigquery_client()
            is_healthy = client._is_client_healthy()
            st.session_state.db_status = "Connected" if is_healthy else "Reconnecting"
            st.session_state.last_health_check = current_time
            log_database_operation("HEALTH_CHECK", "connection_test", "healthy" if is_healthy else "unhealthy")
        except Exception as e:
            logger.error(f"DB_HEALTH_CHECK_ERROR: {e}")
            st.session_state.db_status = "Error"
    
    status = st.session_state.get('db_status', 'Checking')
    
    if status == "Connected":
        st.metric("Database", status, delta="Active")
    elif status == "Error":
        st.metric("Database", status, delta="Failed", delta_color="inverse")
    else:
        st.metric("Database", status, delta="Working", delta_color="off")
    
    # Enhanced Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache"):
            log_user_action("clear_cache_button", "manual_clear")
            st.cache_data.clear()
            st.cache_resource.clear()
            cleanup_session_data()
            log_cache_operation("CLEAR", "manual_clear")
            st.success("Cache cleared")
    
    with col2:
        if st.button("Test DB"):
            log_user_action("test_db_button", "manual_test")
            try:
                client = get_healthy_bigquery_client()
                client.execute_query("SELECT 1")
                log_database_operation("TEST", "SELECT 1", "success")
                st.success("DB OK")
            except Exception as e:
                logger.error(f"DB_TEST_ERROR: {e}")
                st.error("DB Failed")
    
    
    # Technical Details Expander
    with st.expander("Technical Details"):
        st.markdown("""
        **Admin Page Memory Optimization:**
        - Lower memory thresholds (800MB healthy, 1200MB high)
        - Automatic session cleanup after operations
        - TTL on all cached resources (15-30 minutes)
        - Limited paper display (max 50 papers)
        - Aggressive garbage collection
        
        **Memory Monitor:**
        - Tracks RAM usage to prevent crashes
        - "High" warning appears above 1.2GB usage for admin
        
        **Database Monitor:**
        - Tests BigQuery connection every 5 minutes
        
        **Cleanup Features:**
        - "Clear Cache": Standard cache clearing
        - Automatic cleanup after database operations
        """)
        
        # Show memory details
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            st.caption(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
            st.caption(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
        except:
            pass
        
        # Show last check time
        if 'last_health_check' in st.session_state:
            last_check = datetime.fromtimestamp(st.session_state.last_health_check)
            st.caption(f"Last database check: {last_check.strftime('%H:%M:%S')}")
        
        # Show performance metrics
        metrics = log_performance_metrics()
        if metrics:
            st.caption(f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
            st.caption(f"Cache items: {metrics.get('cache_items', 0)}")

# Periodic memory monitoring
if st.session_state.get('last_memory_check', 0) + 60 < time.time():
    logger.info("PERIODIC_MEMORY_CHECK: Running scheduled memory check")
    monitor_and_clear_cache()
    st.session_state.last_memory_check = time.time()

# Main content
log_user_action("main_content_render", "rendering_tabs")
tab1, tab3, tab4 = st.tabs(["Update Database", "Edit Database", "Logs"])

with tab1:
    log_user_action("tab_access", "update_database")
    st.header("Update Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = log_data["last_updated"]
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        start_date=start_date.strftime('%Y-%m-%d')
        end_date = datetime.today()
        end_date=end_date.strftime('%Y-%m-%d')
        days_since = (datetime.now().date() - datetime.strptime(start_date, '%Y-%m-%d').date()).days
        
        logger.info(f"UPDATE_TAB: Days since last update: {days_since}")
        st.write(f"It has been {days_since} days since the last update.")

        if st.button("Fetch Data From API", type="primary"):
            log_user_action("fetch_data_button", f"date_range={start_date}_to_{end_date}")
            warning_placeholder = st.empty()
            
            # Clear any existing data first
            cleanup_session_data()
            
            warning_placeholder.warning("While fetching data, please do not close the tab or refresh the page.")
            
            try:
                log_operation("etl_pipeline_run", f"start_date={start_date}", f"end_date={end_date}")
                
                data_pipeline = initialize_pipeline()
                if data_pipeline is None:
                    logger.error("ETL_PIPELINE: Failed to initialize")
                    st.error("Failed to initialize ETL pipeline")
                    warning_placeholder.empty()
                    st.stop()
                
                with st.spinner("Processing papers..."):
                    papers = data_pipeline.run(start_date, end_date)
                    log_memory("after_etl_pipeline_run")
                    
                    # Process papers immediately without storing in session state
                    if papers is not None and not papers.empty:
                        papers_count = len(papers)
                        logger.info(f"ETL_SUCCESS: Processed {papers_count} papers")
                        
                        # Log the successful data fetch
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        log_data["last_updated"] = current_date
                        log_data["updates"].append({"date": current_date})
                        
                        with open("data/log.json", "w") as file:
                            json.dump(log_data, file, indent=2)
                        
                        logger.info(f"LOG_UPDATE: Updated log file with date {current_date}")
                        
                        # Store only essential info, not the full dataset
                        st.session_state.fetch_success = True
                        st.session_state.papers_count = papers_count
                        st.session_state.sample_papers = papers.head(10)  # Only keep 10 for preview
                        
                        log_user_action("data_fetch_success", f"papers_processed={papers_count}")
                        
                        # Clear the warning and show success
                        warning_placeholder.empty()
                        st.success(f"Data fetched successfully! {papers_count} papers processed. You can validate a sample below.")
                        
                        # Force cleanup of the large dataset
                        del papers
                        collected = gc.collect()
                        logger.info(f"ETL_CLEANUP: Deleted large dataset, collected {collected} objects")
                        log_memory("after_etl_cleanup")
                    else:
                        logger.info("ETL_NO_DATA: No new papers found")
                        warning_placeholder.empty()
                        st.info("No new papers found for the specified date range.")
                        log_user_action("data_fetch_no_results", f"date_range={start_date}_to_{end_date}")
                        
            except Exception as e:
                logger.error(f"ETL_ERROR: {e}")
                st.error(f"Error fetching data: {str(e)}")
                warning_placeholder.empty()
                cleanup_session_data()
                log_user_action("data_fetch_error", f"error={str(e)[:100]}")

    with col2:
        # Show sample papers for validation if fetch was successful
        if st.session_state.get('fetch_success', False) and 'sample_papers' in st.session_state:
            papers_count = st.session_state.papers_count
            logger.info(f"SAMPLE_DISPLAY: Showing sample from {papers_count} papers")
            st.info(f"Showing sample from {papers_count} processed papers")
            
            display_paper_details_optimized(
                st.session_state.sample_papers,
                key="loading_papers_sample",
                show_delete=True,
                client=get_healthy_bigquery_client()
            )
            
            # Add button to clear the sample data
            if st.button("Clear Sample Data", type="secondary"):
                log_user_action("clear_sample_data", f"papers_count={papers_count}")
                cleanup_session_data()
                st.rerun()

with tab3:
    log_user_action("tab_access", "edit_database")
    st.header("Edit Database")
    col3, col4 = st.columns(2)
    with col3:
        doi = st.text_input("Enter the DOI or title:", "")
    
    label_data = get_categories_and_keys(labels)
    log_memory("after_label_processing")

    if doi and doi.strip():
        log_user_action("paper_search", f"query={doi[:50]}...")
        
        # Query BigQuery for paper data using safe query execution
        escaped_doi = escape_sql_string(doi.strip())
        query = f"""
            SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior, doi
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE doi = '{escaped_doi}' OR title = '{escaped_doi}'
            LIMIT 1
        """
        
        paper_data, error = execute_safe_query("searching for paper", query)
        
        # Now create the main content columns
        col1, col2 = st.columns(2)
        
        with col1:
            if paper_data is not None and not paper_data.empty:
                logger.info(f"PAPER_FOUND: Found paper for query '{doi[:50]}...'")
                display_paper_details_optimized(
                    paper_data, 
                    key="edit_database",
                    show_delete=True,
                    client=get_healthy_bigquery_client()
                )
            elif error is None:
                logger.info(f"PAPER_NOT_FOUND: No paper found for query '{doi[:50]}...'")
                st.info("Paper not found in database")
        
        with col2:
            # Edit existing paper
            if paper_data is not None and not paper_data.empty:
                log_user_action("edit_form_render", f"paper={paper_data.iloc[0].get('title', '')[:50]}...")
                
                with st.form("edit_form"):
                    st.markdown("### Edit Paper Details")
                    
                    # Get current values for default selection
                    current_paper = paper_data.iloc[0]
                    
                    # Convert current values to lists for multi-select
                    current_study_types = [current_paper.get('study_type', '')] if current_paper.get('study_type', '') else []
                    current_poverty_contexts = [current_paper.get('poverty_context', '')] if current_paper.get('poverty_context', '') else []
                    current_mechanisms = [current_paper.get('mechanism', '')] if current_paper.get('mechanism', '') else []
                    current_behaviors = [current_paper.get('behavior', '')] if current_paper.get('behavior', '') else []
                    
                    # Handle comma-separated values if they exist
                    if isinstance(current_paper.get('study_type', ''), str) and ',' in current_paper.get('study_type', ''):
                        current_study_types = [x.strip() for x in current_paper.get('study_type', '').split(',')]
                    if isinstance(current_paper.get('poverty_context', ''), str) and ',' in current_paper.get('poverty_context', ''):
                        current_poverty_contexts = [x.strip() for x in current_paper.get('poverty_context', '').split(',')]
                    if isinstance(current_paper.get('mechanism', ''), str) and ',' in current_paper.get('mechanism', ''):
                        current_mechanisms = [x.strip() for x in current_paper.get('mechanism', '').split(',')]
                    if isinstance(current_paper.get('behavior', ''), str) and ',' in current_paper.get('behavior', ''):
                        current_behaviors = [x.strip() for x in current_paper.get('behavior', '').split(',')]
                    
                    # Store current form data in session state for persistence
                    form_key = f"edit_form_{current_paper.get('doi', 'unknown')}"
                    if form_key not in st.session_state:
                        st.session_state[form_key] = {
                            'study_types': [x for x in current_study_types if x in label_data['study_types']],
                            'poverty_contexts': [x for x in current_poverty_contexts if x in label_data['poverty_contexts']],
                            'mechanisms': [x for x in current_mechanisms if x in label_data['mechanisms']],
                            'behaviors': [x for x in current_behaviors if x in label_data['Behaviors']]
                        }
                    
                    # Multi-select fields for editing
                    study_types = st.multiselect(
                        "Study Types", 
                        options=label_data['study_types'], 
                        default=st.session_state[form_key]['study_types'],
                        key=f"study_types_{form_key}"
                    )
                    poverty_contexts = st.multiselect(
                        "Poverty Contexts", 
                        options=label_data['poverty_contexts'], 
                        default=st.session_state[form_key]['poverty_contexts'],
                        key=f"poverty_contexts_{form_key}"
                    )
                    mechanisms = st.multiselect(
                        "Mechanisms", 
                        options=label_data['mechanisms'], 
                        default=st.session_state[form_key]['mechanisms'],
                        key=f"mechanisms_{form_key}"
                    )
                    behaviors = st.multiselect(
                        "Behaviors", 
                        options=label_data['Behaviors'], 
                        default=st.session_state[form_key]['behaviors'],
                        key=f"behaviors_{form_key}"
                    )
                    
                    submit_button = st.form_submit_button(label="Update Paper")
                    
                    if submit_button:
                        log_user_action("paper_update_submit", f"paper={current_paper.get('title', '')[:50]}...")
                        
                        with st.spinner("Updating paper in database..."):
                            try:
                                # Escape all values for SQL safety
                                escaped_study_types = escape_sql_string(", ".join(study_types) if study_types else "")
                                escaped_poverty_contexts = escape_sql_string(", ".join(poverty_contexts) if poverty_contexts else "")
                                escaped_mechanisms = escape_sql_string(", ".join(mechanisms) if mechanisms else "")
                                escaped_behaviors = escape_sql_string(", ".join(behaviors) if behaviors else "")
                                escaped_doi = escape_sql_string(current_paper["doi"])
                                escaped_title = escape_sql_string(current_paper["title"])
                                
                                # Update the paper in BigQuery
                                update_query = f"""
                                    UPDATE `literature-452020.psychology_of_poverty_literature.papers`
                                    SET 
                                        study_type = '{escaped_study_types}',
                                        poverty_context = '{escaped_poverty_contexts}',
                                        mechanism = '{escaped_mechanisms}',
                                        behavior = '{escaped_behaviors}'
                                    WHERE doi = '{escaped_doi}' OR title = '{escaped_title}'
                                """
                                
                                result, error = execute_safe_query("updating paper", update_query)
                                if error is None:
                                    log_user_action("paper_update_success", f"paper={current_paper.get('title', '')[:50]}...")
                                    st.success("Paper details updated successfully!")
                                    # Clear form data from session state and cleanup
                                    if form_key in st.session_state:
                                        del st.session_state[form_key]
                                    cleanup_session_data()
                                    st.rerun()
                                    
                            except Exception as e:
                                logger.error(f"PAPER_UPDATE_ERROR: {e}")
                                log_user_action("paper_update_error", f"error={str(e)[:100]}")
                                st.error(f"Unexpected error updating paper: {str(e)}")
            
            # Add New Paper Manually section
            else:
                log_user_action("add_paper_form_render", f"query={doi[:50]}...")
                
                with st.form("add_form"):
                    st.markdown("### Add New Paper Manually")
                    
                    # Input fields for new paper
                    new_title = st.text_input("Title", value=doi if doi else "")
                    new_doi = st.text_input("DOI", value=doi if doi.startswith('10.') else "")
                    new_authors = st.text_input("Authors")
                    new_year = st.number_input("Year", min_value=1900, max_value=2030, value=2024)
                    new_journal = st.text_input("Journal")
                    new_abstract = st.text_area("Abstract", max_chars=2000)  # Limit abstract length
                    
                    # Multi-select category selections
                    new_study_types = st.multiselect("Study Types", options=label_data['study_types'])
                    new_poverty_contexts = st.multiselect("Poverty Contexts", options=label_data['poverty_contexts'])
                    new_mechanisms = st.multiselect("Mechanisms", options=label_data['mechanisms'])
                    new_behaviors = st.multiselect("Behaviors", options=label_data['Behaviors'])
                    
                    add_button = st.form_submit_button(label="Add New Paper")
                    
                    if add_button:
                        log_user_action("paper_add_submit", f"title={new_title[:50]}...")
                        
                        with st.spinner("Adding new paper to database..."):
                            try:
                                # Escape all values for SQL safety
                                escaped_title = escape_sql_string(new_title)
                                escaped_doi = escape_sql_string(new_doi)
                                escaped_authors = escape_sql_string(new_authors)
                                escaped_journal = escape_sql_string(new_journal)
                                escaped_abstract = escape_sql_string(new_abstract)
                                escaped_study_types = escape_sql_string(", ".join(new_study_types) if new_study_types else "")
                                escaped_poverty_contexts = escape_sql_string(", ".join(new_poverty_contexts) if new_poverty_contexts else "")
                                escaped_mechanisms = escape_sql_string(", ".join(new_mechanisms) if new_mechanisms else "")
                                escaped_behaviors = escape_sql_string(", ".join(new_behaviors) if new_behaviors else "")
                                
                                # Insert new paper into BigQuery
                                insert_query = f"""
                                    INSERT INTO `literature-452020.psychology_of_poverty_literature.papers`
                                    (title, doi, authors, year, journal, abstract, study_type, poverty_context, mechanism, behavior)
                                    VALUES (
                                        '{escaped_title}',
                                        '{escaped_doi}',
                                        '{escaped_authors}',
                                        {new_year},
                                        '{escaped_journal}',
                                        '{escaped_abstract}',
                                        '{escaped_study_types}',
                                        '{escaped_poverty_contexts}',
                                        '{escaped_mechanisms}',
                                        '{escaped_behaviors}'
                                    )
                                """
                                
                                result, error = execute_safe_query("adding new paper", insert_query)
                                if error is None:
                                    log_user_action("paper_add_success", f"title={new_title[:50]}...")
                                    st.success("New paper added successfully!")
                                    cleanup_session_data()
                                    st.rerun()
                                    
                            except Exception as e:
                                logger.error(f"PAPER_ADD_ERROR: {e}")
                                log_user_action("paper_add_error", f"error={str(e)[:100]}")
                                st.error(f"Unexpected error adding paper: {str(e)}")

    else:
        st.info("Enter a DOI or title to search for papers")
        
        # Option to Add New Paper Manually without searching
        with st.expander("➕ Add New Paper Manually"):
            log_user_action("add_paper_expander_open", "manual_add_form")
            
            with st.form("add_new_form"):
                st.markdown("### Add New Paper Manually")
                
                # Input fields for new paper
                new_title = st.text_input("Title")
                new_doi = st.text_input("DOI")
                new_authors = st.text_input("Authors")
                new_year = st.number_input("Year", min_value=1900, max_value=2030, value=2024)
                new_journal = st.text_input("Journal")
                new_abstract = st.text_area("Abstract", max_chars=2000)  # Limit abstract length
                
                # Multi-select category selections
                new_study_types = st.multiselect("Study Types", options=label_data['study_types'])
                new_poverty_contexts = st.multiselect("Poverty Contexts", options=label_data['poverty_contexts'])
                new_mechanisms = st.multiselect("Mechanisms", options=label_data['mechanisms'])
                new_behaviors = st.multiselect("Behaviors", options=label_data['Behaviors'])
                
                add_button = st.form_submit_button(label="Add New Paper")
                
                if add_button and new_title:  # Require at least a title
                    log_user_action("manual_paper_add_submit", f"title={new_title[:50]}...")
                    
                    with st.spinner("Adding new paper to database..."):
                        try:
                            # Escape all values for SQL safety
                            escaped_title = escape_sql_string(new_title)
                            escaped_doi = escape_sql_string(new_doi)
                            escaped_authors = escape_sql_string(new_authors)
                            escaped_journal = escape_sql_string(new_journal)
                            escaped_abstract = escape_sql_string(new_abstract)
                            escaped_study_types = escape_sql_string(", ".join(new_study_types) if new_study_types else "")
                            escaped_poverty_contexts = escape_sql_string(", ".join(new_poverty_contexts) if new_poverty_contexts else "")
                            escaped_mechanisms = escape_sql_string(", ".join(new_mechanisms) if new_mechanisms else "")
                            escaped_behaviors = escape_sql_string(", ".join(new_behaviors) if new_behaviors else "")
                            
                            # Insert new paper into BigQuery
                            insert_query = f"""
                                INSERT INTO `literature-452020.psychology_of_poverty_literature.papers`
                                (title, doi, authors, year, journal, abstract, study_type, poverty_context, mechanism, behavior)
                                VALUES (
                                    '{escaped_title}',
                                    '{escaped_doi}',
                                    '{escaped_authors}',
                                    {new_year},
                                    '{escaped_journal}',
                                    '{escaped_abstract}',
                                    '{escaped_study_types}',
                                    '{escaped_poverty_contexts}',
                                    '{escaped_mechanisms}',
                                    '{escaped_behaviors}'
                                )
                            """
                            
                            result, error = execute_safe_query("adding new paper", insert_query)
                            if error is None:
                                log_user_action("manual_paper_add_success", f"title={new_title[:50]}...")
                                st.success("New paper added successfully!")
                                cleanup_session_data()
                                st.rerun()
                                
                        except Exception as e:
                            logger.error(f"MANUAL_PAPER_ADD_ERROR: {e}")
                            log_user_action("manual_paper_add_error", f"error={str(e)[:100]}")
                            st.error(f"Unexpected error adding paper: {str(e)}")
            
with tab4:
    log_user_action("tab_access", "logs")
    st.header("Operation Logs")
    
    # Display logs efficiently without storing large datasets
    logs_df = pd.DataFrame(log_data)
    st.write(log_data, use_container_width=True)
    


# Final memory cleanup and logging
log_memory("app_end", include_details=True)
logger.info("ADMIN_APP_END: Application cycle complete")

# Cleanup any temporary variables at the end
cleanup_keys = ['temp_data', 'processing_data', 'large_objects']
for key in cleanup_keys:
    if key in locals():
        del locals()[key]

final_collected = gc.collect()
if final_collected > 0:
    logger.info(f"FINAL_CLEANUP: Collected {final_collected} objects at app end")