import streamlit as st
import os
import sys
import json
import time
import gc
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from core.config import get_configuration, get_healthy_bigquery_client
from core.memory import log_memory, monitor_and_clear_cache
from core.auth import check_authentication
from ui.admin import render_system_sidebar, render_update_tab, render_edit_tab, render_logs_tab

st.set_page_config(page_title="Database Updater", layout="wide")

logger.info("=" * 50)
logger.info("ADMIN_APP_START: Streamlit admin application starting")
log_memory("app_initialization")

# Run memory monitoring at the start
monitor_and_clear_cache()

api_key, credentials, email, password = get_configuration()
log_memory("after_configuration_load")


def load_environment_variables():
    """Load environment variables with fallback to Streamlit secrets"""
    from dotenv import load_dotenv
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
        st.error(f"Environment variable '{key}' not found.")
        st.stop()

    return get_env_var


get_env_var = load_environment_variables()


@st.cache_data(ttl=300, max_entries=1)
def load_log_data():
    """Load log data from file"""
    try:
        with open('data/log.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"LOG_DATA_ERROR: {e}")
        return {}


@st.cache_data(ttl=1800, max_entries=1)
def load_labels():
    """Load labels from file"""
    try:
        with open('data/trainingData/labels.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"LABELS_ERROR: {e}")
        return {}


def get_categories_and_keys(data):
    """Extract categories and their keys from nested data structure"""
    result = {}

    def recursive_search(current_data):
        if isinstance(current_data, dict):
            innermost_keys = []
            for key, value in current_data.items():
                if isinstance(value, (dict, list)):
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

    return result


log_data = load_log_data()
labels = load_labels()
label_data = get_categories_and_keys(labels)
log_memory("after_data_load")


if not check_authentication(get_env_var):
    st.stop()


@st.cache_resource(ttl=1800, max_entries=1)
def initialize_pipeline():
    """Initialize and cache the ETL pipeline (lazy loads TensorFlow)"""
    from data.ETL import main
    try:
        pipeline = main.ETLPipeline(
            api_key=api_key,
            credentials_json=credentials,
            project_id='literature-452020',
        )
        return pipeline
    except Exception as e:
        logger.error(f"ETL_PIPELINE_ERROR: {e}")
        st.error(f"Failed to initialize ETL pipeline: {str(e)}")
        return None

def execute_safe_query(query_description, query):
    """Execute a query safely with proper error handling"""
    logger.info(f"OPERATION [execute_safe_query]: {query_description}, query_length={len(query)}")

    try:
        client = get_healthy_bigquery_client()
        result = client.execute_query(query)
        logger.info(f"DATABASE [SUCCESS]: {query_description}")
        return result, None
    except Exception as e:
        error_msg = f"Error {query_description}: {str(e)}"
        logger.error(f"QUERY_ERROR: {error_msg}")
        st.error(error_msg)
        return None, error_msg


def cleanup_session_data():
    """Clean up temporary session data after operations"""
    cleanup_keys = [
        'papers', 'data_fetched', 'selected_paper_temp',
        'processing_complete', 'last_fetch_time'
    ]

    for key in cleanup_keys:
        if key in st.session_state:
            del st.session_state[key]

    gc.collect()
    logger.info("SESSION_CLEANUP: Cleaned temporary items")


if 'connection_tested' not in st.session_state:
    logger.info("USER_ACTION [db_connection_test]: initial_test")
    get_healthy_bigquery_client()
    st.session_state.connection_tested = True
    st.session_state.last_health_check = time.time()
    log_memory("after_initial_db_connection")

render_system_sidebar(get_healthy_bigquery_client, cleanup_session_data)


if st.session_state.get('last_memory_check', 0) + 60 < time.time():
    logger.info("PERIODIC_MEMORY_CHECK: Running scheduled memory check")
    monitor_and_clear_cache()
    st.session_state.last_memory_check = time.time()


logger.info("USER_ACTION [main_content_render]: rendering_tabs")
tab1, tab2, tab3 = st.tabs(["Update Database", "Edit Database", "Logs"])

with tab1:
    render_update_tab(log_data, get_env_var)

with tab2:
    render_edit_tab(
        label_data,
        get_healthy_bigquery_client,
        execute_safe_query,
        cleanup_session_data
    )

with tab3:
    render_logs_tab(log_data)


log_memory("app_end")
logger.info("ADMIN_APP_END: Application cycle complete")

final_collected = gc.collect()
if final_collected > 0:
    logger.info(f"FINAL_CLEANUP: Collected {final_collected} objects at app end")
