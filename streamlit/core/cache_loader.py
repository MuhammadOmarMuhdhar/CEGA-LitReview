import json
import pandas as pd
import streamlit as st
import logging
import os

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, max_entries=1)
def load_statistics_cache():
    """
    Load pre-computed statistics cache (cached for 1 hour)

    Returns:
        dict: Cache data or None if unavailable
    """
    cache_path = "data/cache/statistics_cache.json"

    if not os.path.exists(cache_path):
        logger.warning(f"Statistics cache not found at {cache_path}")
        return None

    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        logger.info(f"Loaded statistics cache (generated {cache['metadata']['generated_at']})")
        return cache
    except Exception as e:
        logger.error(f"Failed to load statistics cache: {e}")
        return None


def get_filters_from_cache(cache):
    """
    Extract filter lists from cache

    Args:
        cache: Cache data dictionary

    Returns:
        tuple: (countries_list, institutions_list, institutions_by_country_dict)
    """
    if cache is None:
        return None, None, None

    return (
        cache['filters']['countries'],
        cache['filters']['institutions'],
        cache['filters']['institutions_by_country']
    )


def get_statistics_from_cache(cache, country='All', institution='All'):
    """
    Get pre-computed statistics for a filter combination

    Args:
        cache: Cache data dictionary
        country: Country filter ('All' or specific country)
        institution: Institution filter ('All' or specific institution)

    Returns:
        dict: Statistics for the filter combination or None if not found
    """
    if cache is None:
        return None

    key = f"{country}__{institution}"
    return cache['statistics'].get(key)


def get_institutions_for_country(cache, country):
    """
    Get institutions list for a specific country

    Args:
        cache: Cache data dictionary
        country: Country name or 'All'

    Returns:
        list: Institutions list (all institutions if country='All' or cache missing)
    """
    if cache is None:
        return []

    if country == 'All':
        return cache['filters']['institutions']

    return cache['filters']['institutions_by_country'].get(country, [])


@st.cache_data(ttl=3600, max_entries=1)
def load_umap_cache():
    """
    Load pre-computed UMAP base data cache 

    Returns:
        pd.DataFrame: UMAP data with columns [title, doi, UMAP1, UMAP2, date]
                     or None if unavailable
    """
    cache_path = "data/cache/umap_base_data.parquet"

    if not os.path.exists(cache_path):
        logger.warning(f"UMAP cache not found at {cache_path}")
        return None

    try:
        umap_df = pd.read_parquet(cache_path)
        logger.info(f"Loaded UMAP cache with {len(umap_df)} papers")
        return umap_df
    except Exception as e:
        logger.error(f"Failed to load UMAP cache: {e}")
        return None

@st.cache_data(ttl=3600, max_entries=1)
def load_topics_cache():
    """
    Load pre-computed Topics data cache

    Returns:
        pd.DataFrame: Topics data with columns [doi, topic, topic_probability]
                     or None if unavailable
    """
    cache_path = "data/cache/topics_data.parquet"

    if not os.path.exists(cache_path):
        logger.warning(f"Topics cache not found at {cache_path}")
        return None

    try:
        topics_df = pd.read_parquet(cache_path)
        logger.info(f"Loaded Topics cache with {len(topics_df)} records")
        return topics_df
    except Exception as e:
        logger.error(f"Failed to load Topics cache: {e}")
        return None


@st.cache_data(ttl=3600, max_entries=1)
def load_sankey_cache():
    """
    Load pre-computed Sankey aggregated data cache (cached for 1 hour)

    Returns:
        pd.DataFrame: Sankey data with columns [poverty_context, study_type, mechanism, behavior, count]
                     or None if unavailable
    """
    cache_path = "data/cache/sankey_base_data.parquet"

    if not os.path.exists(cache_path):
        logger.warning(f"Sankey cache not found at {cache_path}")
        return None

    try:
        sankey_df = pd.read_parquet(cache_path)
        logger.info(f"Loaded Sankey cache with {len(sankey_df)} aggregated records")
        return sankey_df
    except Exception as e:
        logger.error(f"Failed to load Sankey cache: {e}")
        return None

