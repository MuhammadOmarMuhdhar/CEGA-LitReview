import streamlit as st
import psutil
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def render_system_sidebar(get_healthy_client_fn, cleanup_callback=None):
    """Render the system status sidebar with memory and database monitoring"""
    with st.sidebar:
        st.markdown("### System Status")

        # Memory Usage
        _render_memory_status()

        # Database Status
        _render_database_status(get_healthy_client_fn)

        # Quick Actions
        _render_quick_actions(get_healthy_client_fn, cleanup_callback)

        # Technical Details
        _render_technical_details()


def _render_memory_status():
    """Render memory usage metrics"""
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


def _render_database_status(get_healthy_client_fn):
    """Render database connection status"""
    current_time = time.time()
    if ('last_health_check' not in st.session_state or
            current_time - st.session_state.last_health_check > 300):

        logger.info("USER_ACTION [db_health_check]: periodic_check")
        try:
            client = get_healthy_client_fn()
            is_healthy = client._is_client_healthy()
            st.session_state.db_status = "Connected" if is_healthy else "Reconnecting"
            st.session_state.last_health_check = current_time
            logger.info(f"DATABASE [HEALTH_CHECK]: connection_test, {'healthy' if is_healthy else 'unhealthy'}")
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


def _render_quick_actions(get_healthy_client_fn, cleanup_callback):
    """Render quick action buttons"""
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Cache"):
            logger.info("USER_ACTION [clear_cache_button]: manual_clear")
            st.cache_data.clear()
            st.cache_resource.clear()
            if cleanup_callback:
                cleanup_callback()
            logger.info("CACHE [CLEAR] manual_clear:")
            st.success("Cache cleared")

    with col2:
        if st.button("Test DB"):
            logger.info("USER_ACTION [test_db_button]: manual_test")
            try:
                client = get_healthy_client_fn()
                client.execute_query("SELECT 1")
                logger.info("DATABASE [TEST]: SELECT 1, success")
                st.success("DB OK")
            except Exception as e:
                logger.error(f"DB_TEST_ERROR: {e}")
                st.error("DB Failed")


def _render_technical_details():
    """Render technical details expander"""
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
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            cache_items = len(st.session_state) if hasattr(st, 'session_state') else 0
            st.caption(f"CPU: {cpu_percent:.1f}%")
            st.caption(f"Cache items: {cache_items}")
        except:
            pass
