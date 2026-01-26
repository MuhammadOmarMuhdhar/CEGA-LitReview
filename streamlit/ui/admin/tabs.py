import streamlit as st
import pandas as pd
import json
import gc
import logging
import requests
import os
from datetime import datetime

from .forms import (
    display_paper_details,
    render_paper_edit_form,
    render_paper_add_form,
    escape_sql_string,
)

logger = logging.getLogger(__name__)

GITHUB_REPO_OWNER = "MuhammadOmarMuhdhar"
GITHUB_REPO_NAME = "CEGA-LitReview"
GITHUB_WORKFLOW_FILE = "etl.yml"


def render_update_tab(log_data, get_env_var_fn=None):
    """Render the Update Database tab content"""
    logger.info("USER_ACTION [tab_access]: update_database")
    st.header("Update Database")

    start_date = log_data.get("last_updated", "Unknown")
    if start_date != "Unknown":
        start_date_parsed = datetime.strptime(start_date, '%Y-%m-%d')
        days_since = (datetime.now().date() - start_date_parsed.date()).days
        st.write(f"It has been **{days_since} days** since the last update.")
    else:
        st.write("Last update date unknown.")

    st.write(f"Last updated: **{start_date}**")

    st.markdown("---")

    if st.button("Run ETL Pipeline", type="primary"):
        _trigger_github_action(get_env_var_fn)

    st.caption("This triggers a GitHub Action that scrapes new papers, processes them, and updates the database.")

    # Link to view the workflow run
    st.markdown(f"[View workflow runs on GitHub](https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/actions/workflows/{GITHUB_WORKFLOW_FILE})")


def render_edit_tab(label_data, get_healthy_client_fn, execute_query_fn, cleanup_callback=None):
    """Render the Edit Database tab content"""
    logger.info("USER_ACTION [tab_access]: edit_database")
    st.header("Edit Database")

    col3, col4 = st.columns(2)
    with col3:
        doi = st.text_input("Enter the DOI or title:", "")

    if doi and doi.strip():
        _render_search_results(
            doi, label_data, get_healthy_client_fn,
            execute_query_fn, cleanup_callback
        )
    else:
        st.info("Enter a DOI or title to search for papers")
        _render_manual_add_expander(label_data, execute_query_fn, cleanup_callback)


def render_logs_tab(log_data):
    """Render the Logs tab content"""
    logger.info("USER_ACTION [tab_access]: logs")
    st.header("Operation Logs")

    logs_df = pd.DataFrame(log_data)
    st.write(log_data, use_container_width=True)


def _trigger_github_action(get_env_var_fn=None):
    """Trigger the ETL GitHub Action via API"""
    logger.info("USER_ACTION [trigger_github_action]: attempting to trigger workflow")

    # Get GitHub token
    github_token = None
    if get_env_var_fn:
        github_token = get_env_var_fn("GITHUB_TOKEN", None)
    if not github_token:
        github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        st.error("GITHUB_TOKEN not found. Please add it to your environment variables or Streamlit secrets.")
        return

    # GitHub API endpoint
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches"

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    payload = {
        "ref": "main"  # or your default branch
    }

    try:
        with st.spinner("Triggering ETL pipeline..."):
            response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 204:
            logger.info("USER_ACTION [github_action_triggered]: success")
            st.success("ETL pipeline triggered successfully! Check GitHub Actions for progress.")
            st.balloons()
        elif response.status_code == 404:
            logger.error("GITHUB_ACTION_ERROR: Workflow not found")
            st.error("Workflow not found. Make sure the workflow file exists and is pushed to GitHub.")
        elif response.status_code == 401:
            logger.error("GITHUB_ACTION_ERROR: Unauthorized")
            st.error("Unauthorized. Check that your GITHUB_TOKEN has the correct permissions.")
        else:
            logger.error(f"GITHUB_ACTION_ERROR: {response.status_code} - {response.text}")
            st.error(f"Failed to trigger workflow: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"GITHUB_ACTION_ERROR: {e}")
        st.error(f"Error triggering workflow: {str(e)}")


def _render_search_results(doi, label_data, get_healthy_client_fn, execute_query_fn, cleanup_callback):
    """Render search results for a DOI/title query"""
    logger.info(f"USER_ACTION [paper_search]: query={doi[:50]}...")

    escaped_doi = escape_sql_string(doi.strip())
    query = f"""
        SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior, doi
        FROM `literature-452020.psychology_of_poverty_literature.papers`
        WHERE doi = '{escaped_doi}' OR title = '{escaped_doi}'
        LIMIT 1
    """

    paper_data, error = execute_query_fn("searching for paper", query)

    col1, col2 = st.columns(2)

    with col1:
        if paper_data is not None and not paper_data.empty:
            logger.info(f"PAPER_FOUND: Found paper for query '{doi[:50]}...'")
            display_paper_details(
                paper_data,
                key="edit_database",
                show_delete=True,
                client=get_healthy_client_fn(),
                cleanup_callback=cleanup_callback
            )
        elif error is None:
            logger.info(f"PAPER_NOT_FOUND: No paper found for query '{doi[:50]}...'")
            st.info("Paper not found in database")

    with col2:
        if paper_data is not None and not paper_data.empty:
            render_paper_edit_form(
                paper_data.iloc[0],
                label_data,
                execute_query_fn,
                cleanup_callback
            )
        else:
            render_paper_add_form(
                label_data,
                execute_query_fn,
                cleanup_callback,
                prefill_value=doi,
                form_key_suffix=""
            )


def _render_manual_add_expander(label_data, execute_query_fn, cleanup_callback):
    """Render the manual add paper expander"""
    with st.expander("Add New Paper Manually"):
        logger.info("USER_ACTION [add_paper_expander_open]: manual_add_form")
        render_paper_add_form(
            label_data,
            execute_query_fn,
            cleanup_callback,
            prefill_value="",
            form_key_suffix="_new"
        )
