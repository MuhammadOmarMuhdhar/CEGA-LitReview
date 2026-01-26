import streamlit as st
import pandas as pd
import ast
from collections import Counter
import logging
import os
import sys
from core.database import query_geography_data, query_available_filters
from core.cache_loader import load_statistics_cache, get_institutions_for_country, get_statistics_from_cache

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from visuals import bar

logger = logging.getLogger(__name__)

def render_geography_filters(countries, all_institutions):
    """Renders geography filter UI"""
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
            st.session_state['sankey_ready'] = False
            st.session_state['sankey_display_complete'] = False
            logger.info("[SEQUENCE] Geography filter changed - resetting sequence")

        with col4:
            if selected_country != 'All':
                try:
                    cache = load_statistics_cache()
                    if cache is not None:
                        institutions_list = get_institutions_for_country(cache, selected_country)
                        logger.info("[CACHE] Loaded institutions from cache")
                    else:
                        raise ValueError("Cache not available")
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to load institutions from cache: {e}")
                    # Fallback to existing BigQuery logic
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
            st.session_state['sankey_ready'] = False
            st.session_state['sankey_display_complete'] = False
            logger.info("[SEQUENCE] Institution filter changed - resetting sequence")

def render_statistics(countries, all_institutions):
    """Calculates and displays statistics"""
    if not st.session_state.ui_state['stats_computed']:
        # Try cache first
        try:
            cache = load_statistics_cache()

            if cache is not None:
                cached_stats = get_statistics_from_cache(
                    cache,
                    st.session_state.ui_state['selected_country'],
                    st.session_state.ui_state['selected_institution']
                )

                if cached_stats:
                    st.session_state.ui_state['current_stats'] = {
                        'total_papers': cached_stats['total_papers'],
                        'date_range': cached_stats['date_range'],
                        'countries_count': cached_stats['countries_count'],
                        'institutions_count': cached_stats['institutions_count']
                    }
                    # Store top institutions for bar chart
                    st.session_state.ui_state['cached_top_institutions'] = cached_stats['top_institutions']
                    st.session_state.ui_state['stats_computed'] = True
                    logger.info("[CACHE] Loaded statistics from cache")
                else:
                    logger.warning("[CACHE] Stats not in cache, falling back to BigQuery")
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load stats from cache: {e}")

        if not st.session_state.ui_state['stats_computed']:
            working_df = query_geography_data(
                st.session_state.ui_state['selected_country'],
                st.session_state.ui_state['selected_institution']
            )

            total_papers = len(working_df)

            if not working_df.empty:
                min_date = working_df['date'].min()
                max_date = working_df['date'].max()
                date_range = f"{min_date} – {max_date}"
            else:
                date_range = "No data available"

            countries_count = 1 if st.session_state.ui_state['selected_country'] != 'All' else len(countries)
            institutions_count = 1 if st.session_state.ui_state['selected_institution'] != 'All' else len(all_institutions)

            st.session_state.ui_state['current_stats'] = {
                'total_papers': total_papers,
                'date_range': date_range,
                'countries_count': countries_count,
                'institutions_count': institutions_count
            }
            st.session_state.ui_state['cached_top_institutions'] = None
            st.session_state.ui_state['stats_computed'] = True

    stats = st.session_state.ui_state['current_stats']

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

def render_bar_chart():
    """Renders the bar chart of top institutions"""
    with st.spinner("Generating Bar Chart..."):
        with st.expander(" ", expanded=True):
            top_institutions = None

            # Try cache first - check if we have cached top institutions from render_statistics
            if 'cached_top_institutions' in st.session_state.ui_state and st.session_state.ui_state['cached_top_institutions']:
                top_institutions = pd.DataFrame(st.session_state.ui_state['cached_top_institutions'])
                logger.info("[CACHE] Using cached bar chart data from statistics")
            else:
                try:
                   
                    cache = load_statistics_cache()
                    if cache is not None:
                        cached_stats = get_statistics_from_cache(
                            cache,
                            st.session_state.ui_state['selected_country'],
                            st.session_state.ui_state['selected_institution']
                        )
                        if cached_stats and cached_stats.get('top_institutions'):
                            top_institutions = pd.DataFrame(cached_stats['top_institutions'])
                            logger.info("[CACHE] Loaded bar chart data from cache")
                except Exception as e:
                    logger.warning(f"[CACHE] Failed to load bar chart from cache: {e}")

            # Fallback: 
            if top_institutions is None:
                logger.warning("[CACHE] Bar chart not in cache, computing from BigQuery")
                working_df = query_geography_data(
                    st.session_state.ui_state['selected_country'],
                    st.session_state.ui_state['selected_institution']
                )

                all_institutions_in_filtered = []
                for _, row in working_df.iterrows():
                    try:
                        inst_list = ast.literal_eval(str(row['institution']))
                        if isinstance(inst_list, list):
                            all_institutions_in_filtered.extend([str(i) for i in inst_list if i])
                    except:
                        inst_list = [i.strip() for i in str(row['institution']).split(',')]
                        all_institutions_in_filtered.extend([i for i in inst_list if i])

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
