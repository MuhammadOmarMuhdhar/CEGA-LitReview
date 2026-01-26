import streamlit as st
import pandas as pd
import logging
import gc
import hashlib
import ast
import os
import sys
from core.database import (
    query_geography_data,
    query_sankey_data,
    query_sankey_aggregated,
    query_umap_data,
    query_umap_data_optimized,
    query_topics_data,
    query_paper_details
)
from core.memory import log_memory
from ui.filters import load_and_process_filters, _create_sankey_signature, render_filters

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from visuals import sankey, heatMap

logger = logging.getLogger(__name__)


@st.fragment(run_every=5)
def load_visualizations():
    """Loads and displays both Sankey diagram and Research Landscape sequentially"""
    current_signature = _create_sankey_signature()

    
    previous_signature = st.session_state.get('sankey_data_signature')

    data_changed = current_signature != previous_signature
    if data_changed:
        st.session_state['sankey_data_signature'] = current_signature
        logger.info("[SANKEY] Processing started")

    if data_changed:
        gc.collect()
        log_memory("after_sankey_cleanup")

    col1, col2 = st.columns([1, 6])

    with col1: 
        render_filters()
    with col2:
        controls_changed = False

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
            show_context = st.session_state.get("node_context", True)
            show_study = st.session_state.get("node_study", True)
            show_mechanism = st.session_state.get("node_mechanism", True)
            show_behavior = st.session_state.get("node_behavior", True)

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

        chart_signature = {
            'data_signature': current_signature,
            'selected_nodes': selected_nodes,
            'active_filters': active_filters
        }

        previous_chart_signature = st.session_state.get('sankey_chart_signature')
        chart_needs_update = chart_signature != previous_chart_signature

        if data_changed:
            with st.spinner("Loading Sankey Data..."):
                logger.info("[SANKEY] Loading pre-aggregated data...")
                working_df_exploded = query_sankey_aggregated(
                    st.session_state.ui_state['selected_country'],
                    st.session_state.ui_state['selected_institution'],
                    all_selected_contexts,
                    all_selected_study_types,
                    all_selected_mechanisms,
                    all_selected_behaviors
                )

                st.session_state['cached_working_df'] = working_df_exploded

                # DEBUG: Check data size
                logger.info(f"[SANKEY DEBUG] Rows returned: {len(working_df_exploded)}")
                if 'count' in working_df_exploded.columns:
                    logger.info(f"[SANKEY DEBUG] Total count sum: {working_df_exploded['count'].sum()}")
                logger.info(f"[SANKEY DEBUG] Columns: {working_df_exploded.columns.tolist()}")

                gc.collect()
                log_memory("after_sankey_intermediate_cleanup")
        else:
            working_df_exploded = st.session_state.get('cached_working_df', pd.DataFrame())

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

            st.plotly_chart(
                st.session_state['cached_sankey_fig'],
                use_container_width=True,
                key="sankey_chart_stable"
            )
            st.session_state['current_working_df'] = working_df_exploded
            logger.info("[SANKEY] Processing completed")
        else:
            st.write("No data available for selected filters.")
            st.session_state['current_working_df'] = None

    

    logger.info("[HEATMAP] Processing started")

    previous_heatmap_signature = st.session_state.get('heatmap_cache_signature')

    heatmap_data_changed = current_signature != previous_heatmap_signature
    if heatmap_data_changed:
        st.session_state['heatmap_cache_signature'] = current_signature
        logger.info("[HEATMAP] Data changed, regenerating...")

    if heatmap_data_changed or 'cached_heatmap_data' not in st.session_state:
        with st.spinner("Loading Research Landscape Data..."):
            plot_df = query_umap_data_optimized(
                selected_country=st.session_state.ui_state.get('selected_country', 'All'),
                selected_institution=st.session_state.ui_state.get('selected_institution', 'All'),
                filter_contexts=st.session_state.ui_state.get('contexts', []),
                filter_study_types=st.session_state.ui_state.get('study_types', []),
                filter_mechanisms=st.session_state.ui_state.get('mechanisms', []),
                filter_behaviors=st.session_state.ui_state.get('behaviors', [])
            )
            topics_df = query_topics_data()

        st.session_state['cached_heatmap_data'] = {
            'plot_df': plot_df,
            'topics_df': topics_df
        }

        gc.collect()
        log_memory("after_heatmap_data_cleanup")

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
            if heatmap_data_changed or 'cached_heatmap_fig' not in st.session_state:
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

    logger.info("[HEATMAP] Processing completed")

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
