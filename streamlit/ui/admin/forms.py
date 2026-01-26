import streamlit as st
import logging

logger = logging.getLogger(__name__)


def format_list_field(field_value):
    """Format a field value that may be a list into a comma-separated string"""
    if isinstance(field_value, list):
        return ', '.join(str(x) for x in field_value)
    return str(field_value)


def escape_sql_string(value):
    """Escape single quotes in SQL strings to prevent injection"""
    if value is None:
        return ""
    escaped = str(value).replace("'", "''")
    logger.debug(f"SQL_ESCAPE: {str(value)[:50]} -> {escaped[:50]}")
    return escaped


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(paper_title, papers_df, client, key, cleanup_callback=None):
    """Confirmation dialog for paper deletion"""
    logger.info(f"USER_ACTION [delete_dialog_shown]: paper={paper_title[:50]}...")

    st.write(f"Are you sure you want to delete this paper?")
    st.markdown(f"**Title:** {paper_title}")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("Cancel", key=f"cancel_{key}", use_container_width=True):
            logger.info(f"USER_ACTION [delete_cancelled]: paper={paper_title[:50]}...")
            st.rerun()

    with col3:
        if st.button("Delete", key=f"confirm_{key}", type="primary", use_container_width=True):
            logger.info(f"USER_ACTION [delete_confirmed]: paper={paper_title[:50]}...")

            with st.spinner("Deleting paper from database..."):
                try:
                    paper_data = papers_df.iloc[0]
                    title = escape_sql_string(paper_data["title"])
                    doi = escape_sql_string(paper_data.get("doi", ""))

                    delete_query = f"""
                        DELETE FROM `literature-452020.psychology_of_poverty_literature.papers`
                        WHERE title = '{title}' AND doi = '{doi}'
                    """

                    logger.info(f"DATABASE [DELETE]: query='{delete_query[:100]}...', rows=1_paper")

                    client.execute_query(delete_query)

                    logger.info(f"USER_ACTION [paper_deleted_success]: title={title[:50]}...")
                    st.success("Paper deleted successfully!")

                    if cleanup_callback:
                        cleanup_callback()
                    st.rerun()
                except Exception as e:
                    logger.error(f"DELETE_PAPER_ERROR: {e}")
                    logger.info(f"USER_ACTION [paper_delete_failed]: error={str(e)[:100]}")
                    st.error(f"Error deleting paper: {str(e)}")


def display_paper_details(papers_df, key=None, show_delete=False, client=None, cleanup_callback=None):
    """Display paper details with optional delete button"""
    logger.info(f"OPERATION [display_paper_details]: papers_count={len(papers_df)}, show_delete={show_delete}")

    if len(papers_df) > 50:
        papers_df = papers_df.head(50)
        logger.warning(f"PAPERS_DISPLAY: Limited to 50 papers from {len(papers_df)} total")
        st.warning("Showing first 50 papers only to prevent memory issues.")

    with st.expander("Human Review", expanded=True):
        if show_delete:
            delete_col, spacer_col = st.columns([4, 1])
            with delete_col:
                if st.button("Delete", type="secondary", key=f"delete_{key}"):
                    if 'selected_paper_temp' in st.session_state:
                        selected_paper = st.session_state['selected_paper_temp']
                        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
                        logger.info(f"USER_ACTION [delete_button_clicked]: paper={selected_paper[:50]}...")
                        confirm_delete_dialog(
                            selected_paper,
                            selected_paper_details,
                            client,
                            key,
                            cleanup_callback
                        )

        selected_paper = st.selectbox(" ", papers_df['title'].unique(), key=f"select_{key}")
        st.session_state['selected_paper_temp'] = selected_paper
        logger.info(f"USER_ACTION [paper_selected]: paper={selected_paper[:50]}...")

        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
        paper_data = selected_paper_details.iloc[0]

        st.markdown(f"**Title:** {paper_data.get('title', 'N/A')}")
        st.markdown(f"**Study Type:** {format_list_field(paper_data.get('study_type', 'N/A'))}")
        st.markdown(f"**Context:** {format_list_field(paper_data.get('poverty_context', 'N/A'))}")
        st.markdown(f"**Mechanism:** {format_list_field(paper_data.get('mechanism', 'N/A'))}")
        st.markdown(f"**Behavior:** {format_list_field(paper_data.get('behavior', 'N/A'))}")
        st.markdown(f"**Authors:** {format_list_field(paper_data.get('authors', 'N/A'))}")

        abstract = paper_data.get('abstract', 'N/A')
        if len(str(abstract)) > 1000:
            abstract = str(abstract)[:1000] + "..."
        st.markdown(f"**Abstract:** {abstract}")

    logger.info(f"OPERATION_COMPLETE [display_paper_details]: displayed_paper={selected_paper[:50]}...")


def render_paper_edit_form(paper_data, label_data, execute_query_fn, cleanup_callback=None):
    """Render form for editing an existing paper"""
    logger.info(f"USER_ACTION [edit_form_render]: paper={paper_data.get('title', '')[:50]}...")

    with st.form("edit_form"):
        st.markdown("### Edit Paper Details")

        current_paper = paper_data

        current_study_types = _parse_field_to_list(current_paper.get('study_type', ''))
        current_poverty_contexts = _parse_field_to_list(current_paper.get('poverty_context', ''))
        current_mechanisms = _parse_field_to_list(current_paper.get('mechanism', ''))
        current_behaviors = _parse_field_to_list(current_paper.get('behavior', ''))

        form_key = f"edit_form_{current_paper.get('doi', 'unknown')}"
        if form_key not in st.session_state:
            st.session_state[form_key] = {
                'study_types': [x for x in current_study_types if x in label_data['study_types']],
                'poverty_contexts': [x for x in current_poverty_contexts if x in label_data['poverty_contexts']],
                'mechanisms': [x for x in current_mechanisms if x in label_data['mechanisms']],
                'behaviors': [x for x in current_behaviors if x in label_data['Behaviors']]
            }

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
            _handle_paper_update(
                current_paper, study_types, poverty_contexts, mechanisms, behaviors,
                form_key, execute_query_fn, cleanup_callback
            )


def render_paper_add_form(label_data, execute_query_fn, cleanup_callback=None, prefill_value="", form_key_suffix=""):
    """Render form for adding a new paper"""
    logger.info(f"USER_ACTION [add_paper_form_render]: prefill={prefill_value[:50] if prefill_value else ''}...")

    form_key = f"add_form{form_key_suffix}"

    with st.form(form_key):
        st.markdown("### Add New Paper Manually")

        new_title = st.text_input("Title", value=prefill_value if prefill_value else "")
        new_doi = st.text_input("DOI", value=prefill_value if prefill_value.startswith('10.') else "")
        new_authors = st.text_input("Authors")
        new_date = st.text_input("Date (YYYY-MM-DD)", value="")
        new_publication = st.text_input("Publication")
        new_abstract = st.text_area("Abstract", max_chars=2000)

        new_study_types = st.multiselect("Study Types", options=label_data['study_types'])
        new_poverty_contexts = st.multiselect("Poverty Contexts", options=label_data['poverty_contexts'])
        new_mechanisms = st.multiselect("Mechanisms", options=label_data['mechanisms'])
        new_behaviors = st.multiselect("Behaviors", options=label_data['Behaviors'])

        add_button = st.form_submit_button(label="Add New Paper")

        if add_button and new_title:
            _handle_paper_add(
                new_title, new_doi, new_authors, new_date, new_publication, new_abstract,
                new_study_types, new_poverty_contexts, new_mechanisms, new_behaviors,
                execute_query_fn, cleanup_callback
            )


def _parse_field_to_list(field_value):
    """Parse a field value (string or list) into a list of strings"""
    if not field_value:
        return []
    if isinstance(field_value, list):
        return field_value
    if isinstance(field_value, str) and ',' in field_value:
        return [x.strip() for x in field_value.split(',')]
    return [field_value] if field_value else []


def _handle_paper_update(current_paper, study_types, poverty_contexts, mechanisms, behaviors,
                         form_key, execute_query_fn, cleanup_callback):
    """Handle paper update form submission"""
    logger.info(f"USER_ACTION [paper_update_submit]: paper={current_paper.get('title', '')[:50]}...")

    with st.spinner("Updating paper in database..."):
        try:
            escaped_study_types = escape_sql_string(", ".join(study_types) if study_types else "")
            escaped_poverty_contexts = escape_sql_string(", ".join(poverty_contexts) if poverty_contexts else "")
            escaped_mechanisms = escape_sql_string(", ".join(mechanisms) if mechanisms else "")
            escaped_behaviors = escape_sql_string(", ".join(behaviors) if behaviors else "")
            escaped_doi = escape_sql_string(current_paper["doi"])
            escaped_title = escape_sql_string(current_paper["title"])

            update_query = f"""
                UPDATE `literature-452020.psychology_of_poverty_literature.papers`
                SET
                    study_type = '{escaped_study_types}',
                    poverty_context = '{escaped_poverty_contexts}',
                    mechanism = '{escaped_mechanisms}',
                    behavior = '{escaped_behaviors}'
                WHERE doi = '{escaped_doi}' OR title = '{escaped_title}'
            """

            result, error = execute_query_fn("updating paper", update_query)
            if error is None:
                logger.info(f"USER_ACTION [paper_update_success]: paper={current_paper.get('title', '')[:50]}...")
                st.success("Paper details updated successfully!")
                if form_key in st.session_state:
                    del st.session_state[form_key]
                if cleanup_callback:
                    cleanup_callback()
                st.rerun()

        except Exception as e:
            logger.error(f"PAPER_UPDATE_ERROR: {e}")
            logger.info(f"USER_ACTION [paper_update_error]: error={str(e)[:100]}")
            st.error(f"Unexpected error updating paper: {str(e)}")


def _handle_paper_add(new_title, new_doi, new_authors, new_date, new_publication, new_abstract,
                      new_study_types, new_poverty_contexts, new_mechanisms, new_behaviors,
                      execute_query_fn, cleanup_callback):
    """Handle paper add form submission"""
    logger.info(f"USER_ACTION [paper_add_submit]: title={new_title[:50]}...")

    with st.spinner("Adding new paper to database..."):
        try:
            escaped_title = escape_sql_string(new_title)
            escaped_doi = escape_sql_string(new_doi)
            escaped_authors = escape_sql_string(new_authors)
            escaped_date = escape_sql_string(new_date)
            escaped_publication = escape_sql_string(new_publication)
            escaped_abstract = escape_sql_string(new_abstract)
            escaped_study_types = escape_sql_string(", ".join(new_study_types) if new_study_types else "")
            escaped_poverty_contexts = escape_sql_string(", ".join(new_poverty_contexts) if new_poverty_contexts else "")
            escaped_mechanisms = escape_sql_string(", ".join(new_mechanisms) if new_mechanisms else "")
            escaped_behaviors = escape_sql_string(", ".join(new_behaviors) if new_behaviors else "")

            insert_query = f"""
                INSERT INTO `literature-452020.psychology_of_poverty_literature.papers`
                (title, doi, authors, date, publication, abstract, study_type, poverty_context, mechanism, behavior)
                VALUES (
                    '{escaped_title}',
                    '{escaped_doi}',
                    '{escaped_authors}',
                    '{escaped_date}',
                    '{escaped_publication}',
                    '{escaped_abstract}',
                    '{escaped_study_types}',
                    '{escaped_poverty_contexts}',
                    '{escaped_mechanisms}',
                    '{escaped_behaviors}'
                )
            """

            result, error = execute_query_fn("adding new paper", insert_query)
            if error is None:
                logger.info(f"USER_ACTION [paper_add_success]: title={new_title[:50]}...")
                st.success("New paper added successfully!")
                if cleanup_callback:
                    cleanup_callback()
                st.rerun()

        except Exception as e:
            logger.error(f"PAPER_ADD_ERROR: {e}")
            logger.info(f"USER_ACTION [paper_add_error]: error={str(e)[:100]}")
            st.error(f"Unexpected error adding paper: {str(e)}")
