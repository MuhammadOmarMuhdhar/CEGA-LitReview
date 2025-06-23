import streamlit as st
import os
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import pandas as pd
from data.ETL import main
from datetime import datetime
from dotenv import load_dotenv
from streamlit_gsheets import GSheetsConnection
from data.bigQuery import Client
import json
import time
import psutil

def monitor_and_clear_cache():
    """Monitor memory usage and clear cache if needed"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Clear cache if memory usage exceeds 2GB
        if memory_mb > 1024:
            st.cache_data.clear()
            st.cache_resource.clear()
            return True
        return False
    except Exception:
        return False
        
monitor_and_clear_cache()

# Function to load environment variables with Streamlit compatibility
def load_environment_variables():
    """
    Load environment variables with fallback to Streamlit secrets
    """
    # Try to load from .env file first (for local development)
    load_dotenv()
    
    def get_env_var(key, default=None):
        """Get environment variable with fallback to Streamlit secrets"""
        # First try regular environment variables
        value = os.getenv(key)
        if value is not None:
            return value
        
        # Then try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
        # Return default or raise error
        if default is not None:
            return default
        else:
            st.error(f"Environment variable '{key}' not found. Please set it in your .env file or Streamlit secrets.")
            st.stop()
    
    return get_env_var

# Initialize environment variable getter
get_env_var = load_environment_variables()

# Set page config
st.set_page_config(page_title="Database Updater", layout="wide")

# Load all environment variables using the new function
@st.cache_data
def get_configuration():
    """Load and cache configuration from environment variables"""
    
    # Fetch API key from environment variables
    api_key = get_env_var("GEMINI_API_KEY")

    # Load Google Sheets credentials from environment variables
    credentials = {
        "type": get_env_var("type"),
        "project_id": get_env_var("project_id"),
        "private_key_id": get_env_var("private_key_id"),
        "private_key": get_env_var("private_key").replace("\\n", "\n"),
        "client_email": get_env_var("client_email"),
        "client_id": get_env_var("client_id"),
        "auth_uri": get_env_var("auth_uri"),
        "token_uri": get_env_var("token_uri"),
        "auth_provider_x509_cert_url": get_env_var("auth_provider_x509_cert_url"),
        "client_x509_cert_url": get_env_var("client_x509_cert_url"),
        "universe_domain": get_env_var("universe_domain")
    }

    # Load Google Sheets IDs from environment variables
    spreadsheet_ids = {
        "papers": get_env_var("papers_spreadsheet_id"),
        "density": get_env_var("density"),
        "density_X": get_env_var("X"),
        "density_Y": get_env_var("Y"),
        "topics": get_env_var("topics_spreadsheet_id")
    }

    email = get_env_var("USERNAME")
    password = get_env_var("PASSWORD")
    
    return api_key, credentials, spreadsheet_ids, email, password

def get_categories_and_keys(data):
    result = {}

    def recursive_search(current_data):
        if isinstance(current_data, dict):
            innermost_keys = []
            for key, value in current_data.items():
                if isinstance(value, dict) or isinstance(value, list):
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

# Get configuration
api_key, credentials, spreadsheet_ids , email, password = get_configuration()

with open('data/log.json', 'r') as f:
    log_data = json.load(f)

with open('data/trainingData/labels.json', 'r') as f:
    labels = json.load(f)

# enter username password otherwise page isnt authorized
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        with st.form("Login"):
            st.markdown("  ")
            st.markdown("  ")
            
            st.write("This page is protected. Please enter credentials to access.")
            st.markdown("  ")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            st.markdown("  ")

            submitted = st.form_submit_button("Login")
            if submitted:
                # Get credentials from environment/secrets
                expected_username = get_env_var("USERNAME")
                expected_password = get_env_var("PASSWORD")
                
                if username == expected_username and password == expected_password:
                    st.session_state['authenticated'] = True
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            st.markdown("  ")
            st.markdown("  ")
    
    st.stop()

# Initialize ETL Pipeline - ONLY CACHE THE RESOURCE CREATION
@st.cache_resource
def initialize_pipeline():
    """Initialize and cache the ETL pipeline"""
    return main.ETLPipeline(
        api_key=api_key,
        credentials_json=credentials,
        project_id= 'literature-452020',
    )

data_pipeline = initialize_pipeline()

@st.dialog("Confirm Deletion")
def confirm_delete_dialog(paper_title, papers_df, client, key):
    """
    Confirmation dialog for paper deletion
    """
    st.write(f"Are you sure you want to delete this paper?")
    st.markdown(f"**Title:** {paper_title}")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("❌ Cancel", key=f"cancel_{key}", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🗑️ Delete", key=f"confirm_{key}", type="primary", use_container_width=True):
            # Show spinner during deletion
            with st.spinner("Deleting paper from database..."):
                try:
                    # Delete the paper from BigQuery using parameterized query
                    paper_data = papers_df.iloc[0]
                    title = paper_data["title"].replace("'", "''")  # Escape single quotes
                    doi = paper_data.get("doi", "").replace("'", "''")  # Escape single quotes
                    
                    client.execute_query(
                        f"""
                        DELETE FROM `literature-452020.psychology_of_poverty_literature.papers`
                        WHERE title = '{title}' AND doi = '{doi}'
                        """
                    )
                    st.success("Paper deleted successfully!")
                    # Clear any cached data that might be affected
                    if 'selected_paper_temp' in st.session_state:
                        del st.session_state['selected_paper_temp']
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting paper: {str(e)}")

def display_paper_details(papers_df, key=None, show_delete=False, client=None):
    """
    Display paper details with optional delete functionality
    """
    # Drop down to select paper
    with st.expander("Human Review", expanded=True):
        # Delete button at the top if enabled
        if show_delete:
            # Create columns for spacing and delete button
            delete_col, spacer_col = st.columns([4, 1])
            with delete_col:
                if st.button("🗑️ Delete", type="secondary", key=f"delete_{key}"):
                    # We need to get the selected paper first
                    if 'selected_paper_temp' in st.session_state:
                        selected_paper = st.session_state['selected_paper_temp']
                        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
                        confirm_delete_dialog(
                            selected_paper,
                            selected_paper_details,
                            client,
                            key
                        )
        # Select paper dropdown with improved styling
        selected_paper = st.selectbox(" ", papers_df['title'].unique(), key=f"select_{key}")
        # Store selected paper in session state for delete function
        st.session_state['selected_paper_temp'] = selected_paper
        # Display the selected paper's details
        selected_paper_details = papers_df[papers_df['title'] == selected_paper]
        # format all list values in the dataframe to string
        selected_paper_details = selected_paper_details.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x)
        # Display paper information
        st.markdown(f"**Title:** {selected_paper_details['title'].values[0]}")
        st.markdown(f"**Study Type:** {selected_paper_details['study_type'].values[0]}")
        st.markdown(f"**Context:** {selected_paper_details['poverty_context'].values[0]}")
        st.markdown(f"**Mechanism:** {selected_paper_details['mechanism'].values[0]}")
        st.markdown(f"**Behavior:** {selected_paper_details['behavior'].values[0]}")
        st.markdown(f"**Authors:** {selected_paper_details['authors'].values[0]}")
        st.markdown(f"**Abstract:** {selected_paper_details['abstract'].values[0]}")

# UPDATED: Cache BigQuery client with health check capability
@st.cache_resource
def get_bigquery_client():
    try:
        client = Client(credentials, 'literature-452020')
        # Test the connection immediately
        if not client._is_client_healthy():
            st.error("Failed to establish healthy database connection")
            st.stop()
        return client
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

# NEW: Add healthy client function for automatic refresh
def get_healthy_bigquery_client():
    """Get a healthy BigQuery client, handling refresh automatically"""
    client = get_bigquery_client()
    
    # This will automatically refresh the connection if unhealthy
    client.get_healthy_client()
    return client

# NEW: Safe query execution with better error handling
def execute_safe_query(query_description, query):
    """Execute a query safely with proper error handling and user feedback"""
    try:
        client = get_healthy_bigquery_client()
        result = client.execute_query(query)
        return result, None
    except Exception as e:
        error_msg = f"Error {query_description}: {str(e)}"
        st.error(error_msg)
        return None, error_msg

# NEW: Escape function for SQL injection protection
def escape_sql_string(value):
    """Escape single quotes in SQL strings to prevent injection"""
    if value is None:
        return ""
    return str(value).replace("'", "''")

# Initialize connection and health monitoring
if 'connection_tested' not in st.session_state:
    get_bigquery_client()
    st.session_state.connection_tested = True
    st.session_state.last_health_check = time.time()


# System Monitor Section - Clean Version with Metrics
with st.sidebar:
    st.markdown("### System Status")
    
    # Memory Usage
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb < 1024:
            st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="Healthy")
        else:
            st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="High", delta_color="inverse")
            
    except ImportError:
        st.metric("Memory Usage", "N/A", delta="Install psutil")

    # Database Status
    current_time = time.time()
    if ('last_health_check' not in st.session_state or
        current_time - st.session_state.last_health_check > 30):
        
        try:
            client = get_healthy_bigquery_client()
            is_healthy = client._is_client_healthy()
            st.session_state.db_status = "Connected" if is_healthy else "Reconnecting"
            st.session_state.last_health_check = current_time
        except Exception:
            st.session_state.db_status = "Error"
    
    status = st.session_state.get('db_status', 'Checking')
    
    if status == "Connected":
        st.metric("Database", status, delta="Active")
    elif status == "Error":
        st.metric("Database", status, delta="Failed", delta_color="inverse")
    else:
        st.metric("Database", status, delta="Working", delta_color="off")
    
    # Quick Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared")
    
    with col2:
        if st.button("Test DB"):
            try:
                client = get_healthy_bigquery_client()
                client.execute_query("SELECT 1")
                st.success("DB OK")
            except Exception:
                st.error("DB Failed")
    
    # Technical Details Expander
    with st.expander("Technical Details"):
        st.markdown("""
        **Memory Monitor:**
        - Tracks RAM usage to prevent crashes
        - "High" warning appears above 1GB usage
        
        **Database Monitor:**
        - Tests BigQuery connection every 30 seconds
        
        **Status Meanings:**
        - **Connected**: Active connection, queries working normally
        - **Reconnecting**: Stale connection detected, refreshing automatically
        - **Error**: Connection failed, check credentials or network
        
        **Troubleshooting:**
        - If memory stays high: Click "Clear Cache" or refresh page
        - If database shows "Error": Check BigQuery quotas and permissions
        - If frequent reconnects: May indicate network instability
        - Use "Test DB" button to manually verify connection
        """)
        
        # Show last check time
        if 'last_health_check' in st.session_state:
            last_check = datetime.fromtimestamp(st.session_state.last_health_check)
            st.caption(f"Last database check: {last_check.strftime('%H:%M:%S')}")
    

# Main content
tab1, tab3, tab4 = st.tabs(["Update Database", "Edit Database", "Logs"])

with tab1:
    st.header("Update Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
    
        start_date = log_data["last_updated"]
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        start_date=start_date.strftime('%Y-%m-%d')
        end_date = datetime.today()
        end_date=end_date.strftime('%Y-%m-%d')
        st.write(f"It has been {(datetime.now().date() - datetime.strptime(start_date, '%Y-%m-%d').date()).days} days since the last update.")

        if 'data_fetched' not in st.session_state:
            st.session_state.data_fetched = False

        if st.button("Fetch Data From API", type="primary"):
            # Show warning only during active fetching
            warning_placeholder = st.empty()
            
            if not st.session_state.data_fetched:
                warning_placeholder.warning("While fetching data, please do not close the tab or refresh the page.")
                
                try:
                    papers = data_pipeline.run(
                        start_date,
                        end_date)
                    st.session_state.papers = papers
                    st.session_state.data_fetched = True
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    warning_placeholder.empty()

            if st.session_state.data_fetched:
                # Log the successful data fetch
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_data["last_updated"] = current_date
                log_data["updates"].append({"date": current_date})
                with open("data/log.json", "w") as file:
                    json.dump(log_data, file, indent=2)

                # Clear the warning and show success
                warning_placeholder.empty()
                st.success("Data fetched successfully! You can validate the data to confirm accuracy of ETL processing results.")

    with col2:
        if st.session_state.data_fetched:
             display_paper_details(
                st.session_state.papers,
                key="loading_papers",
                show_delete=True,
                client=get_healthy_bigquery_client()
                )

with tab3:

    st.header("Edit Database")
    col3, col4 = st.columns(2)
    with col3:
        doi = st.text_input("Enter the DOI or title:", "")
    
    label_data = get_categories_and_keys(labels)

    if doi and doi.strip():
        # Query BigQuery for paper data using safe query execution
        escaped_doi = escape_sql_string(doi.strip())
        query = f"""
            SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior, doi
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE doi = '{escaped_doi}' OR title = '{escaped_doi}'
            LIMIT 1
        """
        
        paper_data, error = execute_safe_query("searching for paper", query)
        
        # Now create the main content columns
        col1, col2 = st.columns(2)
        
        with col1:
            if paper_data is not None and not paper_data.empty:
                display_paper_details(
                    paper_data, 
                    key="edit_database",
                    show_delete=True,
                    client=get_healthy_bigquery_client()
                )
            elif error is None:
                st.info("Paper not found in database")
        
        with col2:
            # Edit existing paper
            if paper_data is not None and not paper_data.empty:
                with st.form("edit_form"):
                    st.markdown("### Edit Paper Details")
                    
                    # Get current values for default selection
                    current_paper = paper_data.iloc[0]
                    
                    # Convert current values to lists for multi-select
                    current_study_types = [current_paper.get('study_type', '')] if current_paper.get('study_type', '') else []
                    current_poverty_contexts = [current_paper.get('poverty_context', '')] if current_paper.get('poverty_context', '') else []
                    current_mechanisms = [current_paper.get('mechanism', '')] if current_paper.get('mechanism', '') else []
                    current_behaviors = [current_paper.get('behavior', '')] if current_paper.get('behavior', '') else []
                    
                    # Handle comma-separated values if they exist
                    if isinstance(current_paper.get('study_type', ''), str) and ',' in current_paper.get('study_type', ''):
                        current_study_types = [x.strip() for x in current_paper.get('study_type', '').split(',')]
                    if isinstance(current_paper.get('poverty_context', ''), str) and ',' in current_paper.get('poverty_context', ''):
                        current_poverty_contexts = [x.strip() for x in current_paper.get('poverty_context', '').split(',')]
                    if isinstance(current_paper.get('mechanism', ''), str) and ',' in current_paper.get('mechanism', ''):
                        current_mechanisms = [x.strip() for x in current_paper.get('mechanism', '').split(',')]
                    if isinstance(current_paper.get('behavior', ''), str) and ',' in current_paper.get('behavior', ''):
                        current_behaviors = [x.strip() for x in current_paper.get('behavior', '').split(',')]
                    
                    # Store current form data in session state for persistence
                    form_key = f"edit_form_{current_paper.get('doi', 'unknown')}"
                    if form_key not in st.session_state:
                        st.session_state[form_key] = {
                            'study_types': [x for x in current_study_types if x in label_data['study_types']],
                            'poverty_contexts': [x for x in current_poverty_contexts if x in label_data['poverty_contexts']],
                            'mechanisms': [x for x in current_mechanisms if x in label_data['mechanisms']],
                            'behaviors': [x for x in current_behaviors if x in label_data['Behaviors']]
                        }
                    
                    # Multi-select fields for editing
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
                        with st.spinner("Updating paper in database..."):
                            try:
                                # Escape all values for SQL safety
                                escaped_study_types = escape_sql_string(", ".join(study_types) if study_types else "")
                                escaped_poverty_contexts = escape_sql_string(", ".join(poverty_contexts) if poverty_contexts else "")
                                escaped_mechanisms = escape_sql_string(", ".join(mechanisms) if mechanisms else "")
                                escaped_behaviors = escape_sql_string(", ".join(behaviors) if behaviors else "")
                                escaped_doi = escape_sql_string(current_paper["doi"])
                                escaped_title = escape_sql_string(current_paper["title"])
                                
                                # Update the paper in BigQuery
                                update_query = f"""
                                    UPDATE `literature-452020.psychology_of_poverty_literature.papers`
                                    SET 
                                        study_type = '{escaped_study_types}',
                                        poverty_context = '{escaped_poverty_contexts}',
                                        mechanism = '{escaped_mechanisms}',
                                        behavior = '{escaped_behaviors}'
                                    WHERE doi = '{escaped_doi}' OR title = '{escaped_title}'
                                """
                                
                                result, error = execute_safe_query("updating paper", update_query)
                                if error is None:
                                    st.success("Paper details updated successfully!")
                                    # Clear form data from session state
                                    if form_key in st.session_state:
                                        del st.session_state[form_key]
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Unexpected error updating paper: {str(e)}")
            
            # Add New Paper Manually section
            else:
                with st.form("add_form"):
                    st.markdown("### Add New Paper Manually")
                    
                    # Input fields for new paper
                    new_title = st.text_input("Title", value=doi if doi else "")
                    new_doi = st.text_input("DOI", value=doi if doi.startswith('10.') else "")
                    new_authors = st.text_input("Authors")
                    new_year = st.number_input("Year", min_value=1900, max_value=2030, value=2024)
                    new_journal = st.text_input("Journal")
                    new_abstract = st.text_area("Abstract")
                    
                    # Multi-select category selections
                    new_study_types = st.multiselect("Study Types", options=label_data['study_types'])
                    new_poverty_contexts = st.multiselect("Poverty Contexts", options=label_data['poverty_contexts'])
                    new_mechanisms = st.multiselect("Mechanisms", options=label_data['mechanisms'])
                    new_behaviors = st.multiselect("Behaviors", options=label_data['Behaviors'])
                    
                    add_button = st.form_submit_button(label="Add New Paper")
                    
                    if add_button:
                        with st.spinner("Adding new paper to database..."):
                            try:
                                # Escape all values for SQL safety
                                escaped_title = escape_sql_string(new_title)
                                escaped_doi = escape_sql_string(new_doi)
                                escaped_authors = escape_sql_string(new_authors)
                                escaped_journal = escape_sql_string(new_journal)
                                escaped_abstract = escape_sql_string(new_abstract)
                                escaped_study_types = escape_sql_string(", ".join(new_study_types) if new_study_types else "")
                                escaped_poverty_contexts = escape_sql_string(", ".join(new_poverty_contexts) if new_poverty_contexts else "")
                                escaped_mechanisms = escape_sql_string(", ".join(new_mechanisms) if new_mechanisms else "")
                                escaped_behaviors = escape_sql_string(", ".join(new_behaviors) if new_behaviors else "")
                                
                                # Insert new paper into BigQuery
                                insert_query = f"""
                                    INSERT INTO `literature-452020.psychology_of_poverty_literature.papers`
                                    (title, doi, authors, year, journal, abstract, study_type, poverty_context, mechanism, behavior)
                                    VALUES (
                                        '{escaped_title}',
                                        '{escaped_doi}',
                                        '{escaped_authors}',
                                        {new_year},
                                        '{escaped_journal}',
                                        '{escaped_abstract}',
                                        '{escaped_study_types}',
                                        '{escaped_poverty_contexts}',
                                        '{escaped_mechanisms}',
                                        '{escaped_behaviors}'
                                    )
                                """
                                
                                result, error = execute_safe_query("adding new paper", insert_query)
                                if error is None:
                                    st.success("New paper added successfully!")
                                    st.rerun()
                                    
                            except Exception as e:
                                st.error(f"Unexpected error adding paper: {str(e)}")

    else:
        st.info("Enter a DOI or title to search for papers")
        
        # Option to Add New Paper Manually without searching
        with st.expander("➕ Add New Paper Manually"):
            with st.form("add_new_form"):
                st.markdown("### Add New Paper Manually")
                
                # Input fields for new paper
                new_title = st.text_input("Title")
                new_doi = st.text_input("DOI")
                new_authors = st.text_input("Authors")
                new_year = st.number_input("Year", min_value=1900, max_value=2030, value=2024)
                new_journal = st.text_input("Journal")
                new_abstract = st.text_area("Abstract")
                
                # Multi-select category selections
                new_study_types = st.multiselect("Study Types", options=label_data['study_types'])
                new_poverty_contexts = st.multiselect("Poverty Contexts", options=label_data['poverty_contexts'])
                new_mechanisms = st.multiselect("Mechanisms", options=label_data['mechanisms'])
                new_behaviors = st.multiselect("Behaviors", options=label_data['Behaviors'])
                
                add_button = st.form_submit_button(label="Add New Paper")
                
                if add_button and new_title:  # Require at least a title
                    with st.spinner("Adding new paper to database..."):
                        try:
                            # Escape all values for SQL safety
                            escaped_title = escape_sql_string(new_title)
                            escaped_doi = escape_sql_string(new_doi)
                            escaped_authors = escape_sql_string(new_authors)
                            escaped_journal = escape_sql_string(new_journal)
                            escaped_abstract = escape_sql_string(new_abstract)
                            escaped_study_types = escape_sql_string(", ".join(new_study_types) if new_study_types else "")
                            escaped_poverty_contexts = escape_sql_string(", ".join(new_poverty_contexts) if new_poverty_contexts else "")
                            escaped_mechanisms = escape_sql_string(", ".join(new_mechanisms) if new_mechanisms else "")
                            escaped_behaviors = escape_sql_string(", ".join(new_behaviors) if new_behaviors else "")
                            
                            # Insert new paper into BigQuery
                            insert_query = f"""
                                INSERT INTO `literature-452020.psychology_of_poverty_literature.papers`
                                (title, doi, authors, year, journal, abstract, study_type, poverty_context, mechanism, behavior)
                                VALUES (
                                    '{escaped_title}',
                                    '{escaped_doi}',
                                    '{escaped_authors}',
                                    {new_year},
                                    '{escaped_journal}',
                                    '{escaped_abstract}',
                                    '{escaped_study_types}',
                                    '{escaped_poverty_contexts}',
                                    '{escaped_mechanisms}',
                                    '{escaped_behaviors}'
                                )
                            """
                            
                            result, error = execute_safe_query("adding new paper", insert_query)
                            if error is None:
                                st.success("New paper added successfully!")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Unexpected error adding paper: {str(e)}")
            
with tab4:
    st.header("Operation Logs")

    logs_df = pd.DataFrame(log_data)
    st.write(log_data, use_container_width=True)