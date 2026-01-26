import pandas as pd
import streamlit as st
import logging
import gc
import time
import ast
from core.config import get_healthy_bigquery_client
from core.memory import log_data_op, log_memory
from core.cache_loader import load_umap_cache, load_topics_cache, load_statistics_cache, get_filters_from_cache, load_sankey_cache

logger = logging.getLogger(__name__)

def execute_bigquery(sql_query, description="query", log_rows=True, show_progress=False, batch_info=None):
    """
    Centralized BigQuery execution with error handling, logging, and progress tracking.
    All database queries go through this function.
    """
    log_data_op(f"{description}_start")

    try:
        client = get_healthy_bigquery_client()

        if show_progress and batch_info:
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


def _build_filter_clauses(selected_country='All', selected_institution='All',
                         filter_contexts=None, filter_study_types=None,
                         filter_mechanisms=None, filter_behaviors=None):
    """
    Build SQL WHERE clause for filtering data based on geography and Sankey categories.
    Shared by both Sankey and UMAP queries.

    Returns:
        str: SQL WHERE clause (e.g., "country = 'Albania' AND study_type IN ('RCT')")
             Returns "1=1" if no filters are applied
    """
    if isinstance(selected_country, list):
        selected_country = selected_country[0] if selected_country else 'All'
    if isinstance(selected_institution, list):
        selected_institution = selected_institution[0] if selected_institution else 'All'

    filter_clauses = []

    if selected_country != 'All':
        safe_country = selected_country.replace("'", "\\'")
        filter_clauses.append(f"REGEXP_CONTAINS(country_of_study, r'\\b{safe_country}\\b')")

    if selected_institution != 'All':
        safe_institution = selected_institution.replace("'", "\\'")
        filter_clauses.append(f"REGEXP_CONTAINS(institution, r'\\b{safe_institution}\\b')")

    if filter_contexts:
        context_list = "', '".join([ctx.replace("'", "\\'") for ctx in filter_contexts])
        filter_clauses.append(f"poverty_context IN ('{context_list}')")

    if filter_study_types:
        study_list = "', '".join([st.replace("'", "\\'") for st in filter_study_types])
        filter_clauses.append(f"study_type IN ('{study_list}')")

    if filter_mechanisms:
        mech_list = "', '".join([mech.replace("'", "\\'") for mech in filter_mechanisms])
        filter_clauses.append(f"mechanism IN ('{mech_list}')")

    if filter_behaviors:
        behavior_list = "', '".join([beh.replace("'", "\\'") for beh in filter_behaviors])
        filter_clauses.append(f"behavior IN ('{behavior_list}')")

    return " AND ".join(filter_clauses) if filter_clauses else "1=1"


@st.cache_data(ttl=300, max_entries=50)
def query_geography_data(selected_country='All', selected_institution='All'):
    """Query geography data"""

    if isinstance(selected_country, list):
        selected_country = selected_country[0] if selected_country else 'All'
    if isinstance(selected_institution, list):
        selected_institution = selected_institution[0] if selected_institution else 'All'

    with st.spinner("Loading Statistics..."):
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

@st.cache_data(ttl=600, max_entries=1)
def query_available_filters():
    """Query unique countries and institutions for filter dropdowns"""
    # try loading from cache first
    try:
        cache = load_statistics_cache()
        if cache is not None:
            countries, institutions, _ = get_filters_from_cache(cache)
            if countries and institutions:
                logger.info("[CACHE] Loaded filters from cache")
                return countries, institutions
    except Exception as e:
        logger.warning(f"[CACHE] Failed to load filters from cache: {e}")

    logger.info("[CACHE] Cache unavailable, querying BigQuery for filters")
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

        countries = set()
        institutions = set()

        for _, row in result.iterrows():
            country_list = [c.strip() for c in str(row['country_of_study']).split(',')
                           if c.strip() and c.strip().lower() != 'nan']
            countries.update(country_list)

            try:
                inst_list = ast.literal_eval(str(row['institution']))
                if isinstance(inst_list, list):
                    institutions.update([str(i) for i in inst_list if i])
            except:
                inst_list = [i.strip() for i in str(row['institution']).split(',')]
                institutions.update([i for i in inst_list if i])

        return sorted(list(countries)), sorted(list(institutions))

def query_umap_data(doi_list):
    """Query UMAP data for specific DOIs only"""
    if not doi_list:
        return pd.DataFrame()

    base_query = """
        SELECT title, doi, UMAP1, UMAP2, date
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE doi IN ('PLACEHOLDER')
          AND UMAP1 IS NOT NULL
          AND UMAP2 IS NOT NULL
    """

    max_query_size = 1024 * 1000  # 1MB limit
    base_query_size = len(base_query)
    safety_margin = 5000  # 5KB safety margin for UMAP (simpler query)
    available_chars_for_dois = max_query_size - base_query_size - safety_margin

    # Sample DOIs to estimate character usage
    sample_dois = doi_list[:min(50, len(doi_list))]
    sample_doi_str = "', '".join([str(doi).replace("'", "\\'") for doi in sample_dois])
    chars_per_doi = len(sample_doi_str) / len(sample_dois) if sample_dois else 25
    batch_size = int(available_chars_for_dois / chars_per_doi)
    batch_size = max(2000, min(75000, batch_size))  # Between 2K and 75K for UMAP

    logger.info(f"[UMAP_BATCHING] Using batch size: {batch_size} for {len(doi_list)} DOIs")

    total_batches = (len(doi_list) + batch_size - 1) // batch_size

    if total_batches > 1:
        st.markdown(" ")
        progress_bar = st.progress(0)
        status_text = st.empty()

    all_results = []

    for i in range(0, len(doi_list), batch_size):
        batch_dois = doi_list[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        logger.info(f"[UMAP_BATCH] Processing batch {batch_num}/{total_batches} ({len(batch_dois)} DOIs)")

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
                show_progress=(total_batches == 1),  
                batch_info=batch_info
            )
            if not batch_result.empty:
                all_results.append(batch_result)
            logger.info(f"[UMAP_BATCH] Batch {batch_num} completed: {len(batch_result)} rows")

        except Exception as e:
            logger.error(f"[UMAP_BATCH] Batch {batch_num} failed: {str(e)}")
            continue

    if total_batches > 1:
        progress_bar.progress(1.0)

        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

    if all_results:
        result = pd.concat(all_results, ignore_index=True)
        result['UMAP1'] = pd.to_numeric(result['UMAP1'], errors='coerce')
        result['UMAP2'] = pd.to_numeric(result['UMAP2'], errors='coerce')
        logger.info(f"[UMAP_BATCHING] Combined {len(all_results)} batches into {len(result)} total rows")

        # DELETE BATCHES AFTER CONCATENATION
        del all_results
        gc.collect()
        log_memory("after_umap_batch_cleanup")

        return result

    return pd.DataFrame()

@st.cache_data(ttl=300, max_entries=50)
def query_umap_data_optimized(selected_country='All', selected_institution='All',
                              filter_contexts=None, filter_study_types=None,
                              filter_mechanisms=None, filter_behaviors=None):
    """
    Optimized UMAP query using cached parquet file with in-memory filtering.
    Falls back to BigQuery if cache is unavailable.
    """
    # Try loading from cache first
    try:
        umap_cache = load_umap_cache()
        if umap_cache is not None:
            logger.info("[UMAP_CACHE] Using cached UMAP data, filtering in-memory...")

            result = umap_cache.copy()

            has_sankey_filters = (filter_contexts or filter_study_types or
                                 filter_mechanisms or filter_behaviors)

            if has_sankey_filters:
                # Get DOIs that match Sankey filters from BigQuery
                where_clause = _build_filter_clauses(
                    selected_country, selected_institution,
                    filter_contexts, filter_study_types,
                    filter_mechanisms, filter_behaviors
                )

                doi_query = f"""
                    SELECT DISTINCT doi
                    FROM `literature-452020.psychology_of_poverty_literature.fctSankeyLinks`
                    WHERE {where_clause}
                """

                doi_df = execute_bigquery(doi_query, "umap_filter_dois", show_progress=False)
                valid_dois = set(doi_df['doi'].tolist())

                result = result[result['doi'].isin(valid_dois)]
                logger.info(f"[UMAP_CACHE] Filtered to {len(result)} papers matching Sankey filters")

            elif selected_country != 'All' or selected_institution != 'All':
                where_clause = _build_filter_clauses(
                    selected_country, selected_institution,
                    None, None, None, None
                )

                doi_query = f"""
                    SELECT DISTINCT doi
                    FROM `literature-452020.psychology_of_poverty_literature.papers`
                    WHERE {where_clause}
                """

                doi_df = execute_bigquery(doi_query, "umap_filter_geography", show_progress=False)
                valid_dois = set(doi_df['doi'].tolist())

                result = result[result['doi'].isin(valid_dois)]
                logger.info(f"[UMAP_CACHE] Filtered to {len(result)} papers matching geography filters")

            result['UMAP1'] = pd.to_numeric(result['UMAP1'], errors='coerce')
            result['UMAP2'] = pd.to_numeric(result['UMAP2'], errors='coerce')

            logger.info(f"[UMAP_CACHE] Returning {len(result)} rows from cache")
            return result

    except Exception as e:
        logger.warning(f"[UMAP_CACHE] Cache unavailable, falling back to BigQuery: {e}")

    logger.info("[UMAP_FALLBACK] Using BigQuery...")
    has_sankey_filters = (filter_contexts or filter_study_types or
                         filter_mechanisms or filter_behaviors)

    if has_sankey_filters:
        where_clause = _build_filter_clauses(
            selected_country, selected_institution,
            filter_contexts, filter_study_types,
            filter_mechanisms, filter_behaviors
        )

        query = f"""
            SELECT DISTINCT p.title, p.doi, p.UMAP1, p.UMAP2, p.date
            FROM `literature-452020.psychology_of_poverty_literature.papers` p
            WHERE p.doi IN (
                SELECT DISTINCT doi
                FROM `literature-452020.psychology_of_poverty_literature.fctSankeyLinks`
                WHERE {where_clause}
            )
            AND p.UMAP1 IS NOT NULL
            AND p.UMAP2 IS NOT NULL
        """
        description = "umap_data_optimized_with_sankey_filters"

    else:
        where_clause = _build_filter_clauses(
            selected_country, selected_institution,
            None, None, None, None
        )

        query = f"""
            SELECT title, doi, UMAP1, UMAP2, date
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE {where_clause}
            AND UMAP1 IS NOT NULL
            AND UMAP2 IS NOT NULL
        """
        description = "umap_data_optimized_geography_only"

    result = execute_bigquery(query, description, show_progress=True)

    if not result.empty:
        result['UMAP1'] = pd.to_numeric(result['UMAP1'], errors='coerce')
        result['UMAP2'] = pd.to_numeric(result['UMAP2'], errors='coerce')
        logger.info(f"[UMAP_FALLBACK] Retrieved {len(result)} rows from BigQuery")

    return result

@st.cache_data(ttl=300, max_entries=1, show_spinner="Loading Topics...")
def query_topics_data():
    """Query topics data (can stay cached as it's small and static)"""

    try:
        topics_cache = load_topics_cache()
        # display the topics dataframe 
        # st.write(topics_cache)
        if topics_cache is not None:
            logger.info("[TOPICS_CACHE] Using cached Topics data")

            if not result.empty:
                topics_cache['umap_1_mean'] = pd.to_numeric(topics_cache['umap_1_mean'], errors='coerce')
                topics_cache['umap_2_mean'] = pd.to_numeric(topics_cache['umap_2_mean'], errors='coerce')
            return topics_cache
        
    except Exception as e:
        logger.warning(f"[TOPICS_CACHE] Failed to load Topics cache: {e}")

    logger.info("[TOPICS_CACHE] Cache unavailable, querying BigQuery for Topics data")
    query = "SELECT * FROM `literature-452020.psychology_of_poverty_literature.topics`"
    result = execute_bigquery(query, "topics_data")

    if not result.empty:
        result['umap_1_mean'] = pd.to_numeric(result['umap_1_mean'], errors='coerce')
        result['umap_2_mean'] = pd.to_numeric(result['umap_2_mean'], errors='coerce')

    return result

@st.cache_data(ttl=300, max_entries=50)
def query_sankey_data(selected_country='All', selected_institution='All',
                     filter_contexts=None, filter_study_types=None,
                     filter_mechanisms=None, filter_behaviors=None):
    """Query pre-exploded Sankey data from dbt table with direct filtering"""

    where_clause = _build_filter_clauses(
        selected_country, selected_institution,
        filter_contexts, filter_study_types,
        filter_mechanisms, filter_behaviors
    )

    query = f"""
        SELECT doi, poverty_context, study_type, mechanism, behavior
        FROM `literature-452020.psychology_of_poverty_literature.fctSankeyLinks`
        WHERE {where_clause}
    """

    return execute_bigquery(query, "sankey_data")

@st.cache_data(ttl=300, max_entries=50)
def query_sankey_aggregated(selected_country='All', selected_institution='All',
                           filter_contexts=None, filter_study_types=None,
                           filter_mechanisms=None, filter_behaviors=None):
    """
    Query pre-aggregated Sankey data for fast visualization rendering.
    Uses cached parquet file when no filters applied, otherwise queries BigQuery.
    """
    no_filters_applied = (
        selected_country == 'All' and
        selected_institution == 'All' and
        not filter_contexts and
        not filter_study_types and
        not filter_mechanisms and
        not filter_behaviors
    )

    # Try loading from cache if no filters
    if no_filters_applied:
        try:
            sankey_cache = load_sankey_cache()
            if sankey_cache is not None:
                logger.info(f"[SANKEY_CACHE] Using cached Sankey data ({len(sankey_cache)} rows)")
                return sankey_cache
        except Exception as e:
            logger.warning(f"[SANKEY_CACHE] Cache unavailable, falling back to BigQuery: {e}")

    where_clause = _build_filter_clauses(
        selected_country, selected_institution,
        filter_contexts, filter_study_types,
        filter_mechanisms, filter_behaviors
    )

    query = f"""
        SELECT
            poverty_context,
            study_type,
            mechanism,
            behavior,
            paper_count as count
        FROM `literature-452020.psychology_of_poverty_literature.fctSankeyAggregated`
        WHERE {where_clause}
    """

    logger.info(f"[SANKEY_AGGREGATED] Query WHERE clause: {where_clause}")
    result = execute_bigquery(query, "sankey_aggregated")
    logger.info(f"[SANKEY_AGGREGATED] Retrieved {len(result)} aggregated rows")
    return result

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
