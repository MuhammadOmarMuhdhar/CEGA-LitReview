import streamlit as st
import sys
import logging
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

from core.config import get_healthy_bigquery_client
from core.memory import monitor_and_clear_cache, log_memory
from core.database import query_available_filters
from ui.filters import render_filters
from ui.statistics import render_geography_filters, render_statistics, render_bar_chart
from ui.visualizations import load_visualizations

st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')


def render_about_tab():
    """About tab content"""
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

def render_dashboard_tab():
    """Dashboard tab orchestration"""
    st.markdown("""
    In the research landscape below, we offer a guided visual exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
    """)

    st.markdown("##### Data at a Glance")

    countries, all_institutions = query_available_filters()

    if 'stats_computed' not in st.session_state.ui_state:
        st.session_state.ui_state['stats_computed'] = False

    col1, col2 = st.columns([1, 1.4])

    with col1:
        render_geography_filters(countries, all_institutions)
        render_statistics(countries, all_institutions)

    with col2:
        render_bar_chart()

    st.markdown("#### Connecting Poverty Context, Psychological Mechanisms and Behavior")
    st.markdown("""
    Use the filters below to customize the Sankey diagram.
    Performance decreases when visualizing a large number of papers.
    """)

    # col1, col2 = st.columns([1, 6])

    # with col1:
    #     render_filters()

    load_visualizations()

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
            'paper_details': st.session_state.get('paper_details', {}),
            'current_papers_list': [],
            'current_selected_paper': None,
            'current_paper_details': st.session_state.get('current_paper_details', {})
        }

    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.title("Psychology and Economics of Poverty Literature Review Dashboard")
    with col2:
        st.image("streamlit/logo.png", use_container_width=False)

    tab1, tab2 = st.tabs(["Dashboard", "About"])

    with tab2:
        render_about_tab()

    with tab1:
        render_dashboard_tab()

    gc.collect()
    log_memory("main_function_exit")

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
