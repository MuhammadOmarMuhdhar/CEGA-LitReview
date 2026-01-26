import logging
import psutil
import sys
import streamlit as st
import gc

logger = logging.getLogger(__name__)

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

            # Clear session state data to free up memory
            cache_keys_to_clear = [
                'cached_working_df', 'cached_sankey_fig', 'cached_heatmap_data',
                'cached_heatmap_fig', 'current_working_df', 'current_paper_details'
            ]

            for key in cache_keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

            if hasattr(st.session_state, 'ui_state'):
                ui_keys_to_clear = [
                    'paper_details', 'current_papers_list', 'current_paper_details'
                ]
                for key in ui_keys_to_clear:
                    if key in st.session_state.ui_state:
                        del st.session_state.ui_state[key]

                st.session_state.ui_state['stats_computed'] = False

            gc.collect()
            log_memory("after_enhanced_cache_clear")
            return True
        return False
    except Exception:
        return False
