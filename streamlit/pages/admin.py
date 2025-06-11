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

# Get configuration
api_key, credentials, spreadsheet_ids , email, password = get_configuration()

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

# Initialize ETL Pipeline
@st.cache_resource
def initialize_pipeline():
    """Initialize and cache the ETL pipeline"""
    return main.ETLPipeline(
        api_key=api_key,
        credentials_json=credentials,
        spreadsheet_id_json=spreadsheet_ids,
        limit=10
    )

data_pipeline = initialize_pipeline()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from Google Sheets with caching"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        papers_url = 'https://docs.google.com/spreadsheets/d/1nrZC6zJ50DouMCHIWZOl-tQ4BAZfEojJymtzUh26nP0/edit?usp=sharing'
        topics_url = 'https://docs.google.com/spreadsheets/d/1cspghq8R0Xlf2jk0TacGv8bC6C4EGIUnGuUQJWYASpk/edit?usp=sharing'

        papers_df = conn.read(spreadsheet=papers_url)
        topics_df = conn.read(spreadsheet=topics_url)

        return papers_df, topics_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# Main content
tab1, tab2, tab3 = st.tabs(["Update Database", "View Data", "Logs"])

with tab1:
    st.header("Update Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.today())
        end_date = st.date_input("End Date", datetime.today())

        if 'data_fetched' not in st.session_state:
            st.session_state.data_fetched = False

        if st.button("Fetch Data From API", type="primary"):
            # Show warning only during active fetching
            warning_placeholder = st.empty()
            
            if not st.session_state.data_fetched:
                warning_placeholder.warning("While fetching data, please do not close the tab or refresh the page.")
                
                try:
                    papers = data_pipeline.run(
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'))
                    st.session_state.papers = papers
                    st.session_state.data_fetched = True
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
                    warning_placeholder.empty()

            if st.session_state.data_fetched:
                # Clear the warning and show success
                warning_placeholder.empty()
                st.success("Data fetched successfully! You can validate the data to confirm accuracy of ETL processing results.")

    with col2:
        if st.session_state.data_fetched:
            def display_paper_details(papers_df):
                # Drop down to select paper
                with st.expander("Human Review", expanded=True):
                    st.markdown("##### Validate Extracted Papers")
                    
                    # Select paper dropdown with improved styling
                    selected_paper = st.selectbox("Validate extracted data and confirm accuracy of ETL processing results", papers_df['title'].unique())

                    # Display the selected paper's details
                    selected_paper_details = papers_df[papers_df['title'] == selected_paper]
                    # format all list values in the dataframe to string
                    selected_paper_details = selected_paper_details.applymap(lambda x: ', '.join(x) if isinstance(x, list) else x)

                    st.markdown(f"**Context:** {selected_paper_details['poverty_context'].values[0]}")
                    st.markdown(f"**Mechanism:** {selected_paper_details['mechanism'].values[0]}")
                    st.markdown(f"**Study Type:** {selected_paper_details['study_type'].values[0]}")
                    st.markdown(f"**Authors:** {selected_paper_details['authors'].values[0]}")
                    st.markdown(f"**Abstract:** {selected_paper_details['abstract'].values[0]}")

            display_paper_details(st.session_state.papers)

with tab2:
    papers_df, topics_df = load_data()
    if not papers_df.empty:
        st.dataframe(papers_df[['doi', 'title', 'authors', 'abstract', 'country', 'institution', 'study_type', 'poverty_context', 'mechanism']], use_container_width=True)
    else:
        st.info("No data available")

    # button to reload with st.rerun
    if st.button("Reload Data"):
        with st.spinner("Reloading data..."):
            # Clear cache and reload
            load_data.clear()
            st.rerun()

with tab3:
    st.header("Operation Logs")
    
    if 'logs' in st.session_state and st.session_state['logs']:
        logs_df = pd.DataFrame(st.session_state['logs'])
        st.dataframe(logs_df, use_container_width=True)
    else:
        st.info("No logs available yet")
    
    if st.button("Clear Logs"):
        if 'logs' in st.session_state:
            st.session_state['logs'] = []
            st.success("Logs cleared")

    try:
        from streamlit_tree_select import tree_select

        data = [
            {
                "label": "Fruit",
                "value": "fruit",
                "children": [
                    {"label": "Apple", "value": "apple"},
                    {"label": "Banana", "value": "banana"}
                ]
            },
            {
                "label": "Vegetable",
                "value": "vegetable",
                "children": [
                    {"label": "Carrot", "value": "carrot"},
                    {"label": "Spinach", "value": "spinach"}
                ]
            }
        ]

        selection = tree_select(data)
        st.write("Selection:", selection)
    except ImportError:
        st.info("streamlit_tree_select not available")