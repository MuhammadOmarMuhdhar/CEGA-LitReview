import streamlit as st
import pandas as pd
import os 
import sys
import plotly.graph_objs as go
from streamlit_tree_select import tree_select
from dotenv import load_dotenv
import ast
import json
import psutil 
import time
import re
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from visuals import bar, sankey, heatMap
from data.bigQuery import Client

st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

def monitor_and_clear_cache():
    """Monitor memory usage and clear cache if needed"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Clear cache if memory usage exceeds 1.5GB
        if memory_mb > 1500:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()  # Rerun to refresh the page after clearing cache
            return True
        return False
    except Exception:
        return False

def load_filters_json():
    """Load filters JSON once and cache it"""
    with open('data/trainingData/labels.json', 'r') as f:
        return json.load(f)

@st.cache_data(ttl=3600, max_entries=2, show_spinner="Processing data...")
def preprocess_papers_data(papers_df):
    """One-time expensive preprocessing with categorical optimization"""
    df = papers_df.copy()
    
    # Pre-parse institutions (convert lists to strings for caching)
    df['institutions_list'] = df['institution'].apply(
        lambda x: str(list(dict.fromkeys([i for i in ast.literal_eval(str(x)) if i is not None])))
    )
    
    # Pre-parse countries (convert lists to strings for caching)
    df['countries_list'] = df['country_of_study'].apply(
        lambda x: str([i.strip() for i in str(x).split(',') if i.strip() and i.strip().lower() != 'nan'])
    )
    
    df['institutions_list'] = df['institutions_list'].astype('category')
    df['countries_list'] = df['countries_list'].astype('category')
    
    return df

# Cache expensive data processing operations - NOW MUCH FASTER
@st.cache_data(ttl=1800, max_entries=3, show_spinner="Loading Filters...")
def process_countries_and_institutions(preprocessed_df):
    """Process and extract unique countries and institutions from preprocessed dataset"""
    # Extract unique countries from string representations
    countries = []
    for country_str in preprocessed_df['countries_list']:
        country_list = ast.literal_eval(country_str)  # Convert string back to list
        countries.extend(country_list)
    countries = list(set(countries))
    countries = [str(country) for country in countries if country and str(country).lower() != 'nan']
    
    # Extract unique institutions from string representations
    institutions = []
    for inst_str in preprocessed_df['institutions_list']:
        inst_list = ast.literal_eval(inst_str)  # Convert string back to list
        institutions.extend(inst_list)
    institutions = list(set(institutions))
    institutions = [str(inst) for inst in institutions if inst]
    
    return sorted(countries), sorted(institutions)

@st.cache_data(ttl=1800, max_entries=5, show_spinner="Loading Metadata Filters...")
def lightning_fast_filter(preprocessed_df, selected_country, selected_institution):
    """Super fast filtering using preprocessed data - no more ast.literal_eval!"""
    result = preprocessed_df.copy()
    
    if selected_country != 'All':
        mask = result['countries_list'].apply(lambda x: selected_country in ast.literal_eval(x))
        result = result[mask]
    
    if selected_institution != 'All':
        mask = result['institutions_list'].apply(lambda x: selected_institution in ast.literal_eval(x))
        result = result[mask]
    
    return result

@st.cache_data(ttl=1800, max_entries=3, show_spinner="Loading Metadata...")
def calculate_statistics(filtered_df, papers_df, selected_country, selected_institution):
    """Calculate statistics for the filtered dataset"""
    # Filter publications based on the DOIs in the working dataframe
    filtered_publications = papers_df[papers_df['doi'].isin(filtered_df['doi'].tolist())]
    
    # Basic stats
    total_papers = len(filtered_publications)
    
    # Date range
    if not filtered_publications.empty:
        min_date = filtered_publications['date'].min()
        max_date = filtered_publications['date'].max()
        date_range = f"{min_date} – {max_date}"
    else:
        date_range = "No data available"
    
    # Countries count - use preprocessed data
    if selected_country != 'All':
        countries_count = 1
    else:
        all_countries = []
        for country_str in filtered_df['countries_list']:
            country_list = ast.literal_eval(country_str)  # Convert string back to list
            all_countries.extend(country_list)
        countries_count = len(set(all_countries))
    
    # Institutions count - use preprocessed data
    if selected_institution != 'All':
        institutions_count = 1
    else:
        all_institutions = []
        for inst_str in filtered_df['institutions_list']:
            inst_list = ast.literal_eval(inst_str)  # Convert string back to list
            all_institutions.extend(inst_list)
        institutions_count = len(set(all_institutions))
    
    return {
        'total_papers': total_papers,
        'date_range': date_range,
        'countries_count': countries_count,
        'institutions_count': institutions_count
    }

def get_filtered_data(selected_country, selected_institution):
    """Cache filtered results to avoid recomputation"""
    preprocessed_papers = get_preprocessed_papers()
    return lightning_fast_filter(preprocessed_papers, selected_country, selected_institution)

def get_exploded_sankey_data(working_df_dois, all_selected_context, all_selected_study_types, all_selected_mechanisms, all_selected_behaviors):
    """Filter FIRST, then explode - with categorical preservation"""
    import re
    
    labels_data = get_labels_data()
    
    # Step 1: Filter by DOIs first (smallest operation)
    sankey_working_df = labels_data[labels_data['doi'].isin(working_df_dois)]
    
    # Step 2: Apply content filters BEFORE exploding
    # Convert categorical to string for filtering, then back to categorical
    
    if all_selected_context:
        context_pattern = '|'.join([re.escape(ctx) for ctx in all_selected_context])
        # Convert categorical to string for str.contains, then filter
        context_mask = sankey_working_df['poverty_context'].astype(str).str.contains(context_pattern, na=False, case=False)
        sankey_working_df = sankey_working_df[context_mask]
    
    if all_selected_study_types:
        study_pattern = '|'.join([re.escape(st) for st in all_selected_study_types])
        study_mask = sankey_working_df['study_type'].astype(str).str.contains(study_pattern, na=False, case=False)
        sankey_working_df = sankey_working_df[study_mask]
    
    if all_selected_mechanisms:
        mechanism_pattern = '|'.join([re.escape(mech) for mech in all_selected_mechanisms])
        mechanism_mask = sankey_working_df['mechanism'].astype(str).str.contains(mechanism_pattern, na=False, case=False)
        sankey_working_df = sankey_working_df[mechanism_mask]
    
    if all_selected_behaviors:
        behavior_pattern = '|'.join([re.escape(beh) for beh in all_selected_behaviors])
        behavior_mask = sankey_working_df['behavior'].astype(str).str.contains(behavior_pattern, na=False, case=False)
        sankey_working_df = sankey_working_df[behavior_mask]
    
    # Step 3: Convert categoricals to strings for exploding
    working_df_exploded = sankey_working_df.copy()
    
    # Convert categorical columns to string before string operations
    categorical_columns = ['poverty_context', 'mechanism', 'study_type', 'behavior']
    for col in categorical_columns:
        if col in working_df_exploded.columns:
            working_df_exploded[col] = working_df_exploded[col].astype(str)
    
    # Explode operations (now on string columns)
    working_df_exploded['poverty_context'] = working_df_exploded['poverty_context'].str.split(',')
    working_df_exploded = working_df_exploded.explode('poverty_context')
    working_df_exploded['poverty_context'] = working_df_exploded['poverty_context'].str.strip()
    
    working_df_exploded['mechanism'] = working_df_exploded['mechanism'].str.split(',')
    working_df_exploded = working_df_exploded.explode('mechanism')
    working_df_exploded['mechanism'] = working_df_exploded['mechanism'].str.strip()
    
    working_df_exploded['study_type'] = working_df_exploded['study_type'].str.split(',')
    working_df_exploded = working_df_exploded.explode('study_type')
    working_df_exploded['study_type'] = working_df_exploded['study_type'].str.strip()
    
    working_df_exploded['behavior'] = working_df_exploded['behavior'].str.split(',')
    working_df_exploded = working_df_exploded.explode('behavior')
    working_df_exploded['behavior'] = working_df_exploded['behavior'].str.strip()

    # Step 4: Final exact filtering
    if all_selected_context:
        working_df_exploded = working_df_exploded[working_df_exploded['poverty_context'].isin(all_selected_context)]
    if all_selected_study_types:
        working_df_exploded = working_df_exploded[working_df_exploded['study_type'].isin(all_selected_study_types)]
    if all_selected_mechanisms:
        working_df_exploded = working_df_exploded[working_df_exploded['mechanism'].isin(all_selected_mechanisms)]
    if all_selected_behaviors:
        working_df_exploded = working_df_exploded[working_df_exploded['behavior'].isin(all_selected_behaviors)]
    
    # Step 5: Convert back to categorical for final DataFrame (optional but saves memory)
    for col in categorical_columns:
        if col in working_df_exploded.columns:
            working_df_exploded[col] = working_df_exploded[col].astype('category')
    
    return working_df_exploded

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

@st.cache_data(ttl=7200, max_entries=1, show_spinner="Loading Configuration...")
def get_configuration():
    """Load and cache configuration from environment variables"""
    get_env_var = load_environment_variables()
    
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
    
    email = get_env_var("USERNAME")
    password = get_env_var("PASSWORD")
    
    return api_key, credentials, email, password

# Assign variables from cached configuration
api_key, credentials, email, password = get_configuration()

st.sidebar.markdown("""
         <style>
            .sidebar-icons {
                    position: absolute;
                    bottom: -40px;
                    right: 80;
                    display: flex;
                    flex-direction: column;
                    align-items: flex-left;
                            }
            .sidebar-icons img {
                    width: 30px; /* Adjust size as needed */
                    margin-bottom: 10px;
                            }
        </style>
                        """,
                        unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Connecting to Database...")
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

def get_healthy_bigquery_client():
    """Get a healthy BigQuery client, handling refresh automatically"""
    client = get_bigquery_client()
    
    # This will automatically refresh the connection if unhealthy
    client.get_healthy_client()
    return client

@st.cache_data(ttl=3600, max_entries=1, show_spinner="Connecting to Database...")
def load_country_institution_data():
    try:
        client = get_healthy_bigquery_client()
        df = client.execute_query(
            "SELECT doi, country, date, institution, country_of_study "
            "FROM `literature-452020.psychology_of_poverty_literature.papers`"
        )
        
        categorical_columns = ['country', 'institution', 'country_of_study']
        
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=1, show_spinner="Loading Sankey Diagram Data...")
def load_label_data():
    try:
        client = get_healthy_bigquery_client()
        df = client.execute_query(
            "SELECT doi, authors, study_type, poverty_context, "
            "mechanism, behavior "
            "FROM `literature-452020.psychology_of_poverty_literature.papers`"
        )
        
        categorical_columns = ['study_type', 'poverty_context', 'mechanism', 'behavior', 'authors']
        
        for col in categorical_columns:
            if col in df.columns:
                # Convert to categorical - massive memory savings
                df[col] = df[col].astype('category')
                
        return df
    except Exception as e:
        st.error(f"Failed to load label data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=1, show_spinner="Loading Heatmap Data...")
def load_topics():
    try:
        client = get_healthy_bigquery_client()
        return client.execute_query(
            "SELECT * "
            "FROM `literature-452020.psychology_of_poverty_literature.topics`"
        )
    except Exception as e:
        st.error(f"Failed to load topics: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=1800, max_entries=10, show_spinner="Loading Abstract Data...")
def load_abstract_data(title):
    try:
        client = get_healthy_bigquery_client()
        # Handle all problematic characters
        safe_title = title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        
        return client.execute_query(
            f"""
            SELECT abstract, title, authors, study_type, poverty_context, mechanism, behavior
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            WHERE title = '{safe_title}'
            LIMIT 1
            """
        )
    except Exception as e:
        st.error(f"Failed to load abstract data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, max_entries=1, show_spinner="Loading Heat Map Data...")
def load_umap():
    try:
        client = get_healthy_bigquery_client()
        return client.execute_query(
            f"""
            SELECT  title, doi, UMAP1, UMAP2, date
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            """
        )
    except Exception as e:
        st.error(f"Failed to load UMAP data: {str(e)}")
        return pd.DataFrame()
    
def get_working_df_exploded_cached(country, institution, contexts, study_types, mechanisms, behaviors):
    """Process on demand - now much faster due to pre-filtering"""
    working_df = get_filtered_data(country, institution)
    
    return get_exploded_sankey_data(
        working_df['doi'].tolist(),
        list(contexts),  # Convert tuples to lists
        list(study_types), 
        list(mechanisms),
        list(behaviors)
    )

def get_papers_data():
    return load_country_institution_data()

def get_preprocessed_papers():
    papers_df = get_papers_data()
    return preprocess_papers_data(papers_df)

def get_labels_data():
    return load_label_data()

def get_umap_data():
    return load_umap()

@st.fragment
def paper_details_fragment():
    """
    Isolated fragment for paper selection and details display.
    This will only rerun when interactions happen within this fragment,
    preventing the entire app from rerunning when selecting papers.
    """
    
    # Paper count display
    with st.expander("Number of Papers Visualized", expanded=True):
        paper_count = len(st.session_state.ui_state.get('current_papers_list', []))
        st.markdown(f"**{paper_count:,}**")

    # Paper selection and details
    with st.expander("Select Paper", expanded=True):
        available_papers = st.session_state.ui_state.get('current_papers_list', [])
        
        if available_papers:
            # Use a completely independent selectbox
            selected_paper = st.selectbox(
                "Select Paper", 
                available_papers,
                key="isolated_paper_selector",  # Unique key to avoid conflicts
                help="Select a paper to view detailed information"
            )

            # Only load abstract if paper changed (completely independent operation)
            if selected_paper != st.session_state.ui_state.get('current_selected_paper'):
                st.session_state.ui_state['current_selected_paper'] = selected_paper
                with st.spinner("Loading paper details..."):
                    st.session_state.ui_state['current_paper_details'] = load_abstract_data(selected_paper)
            
            # Display cached paper details
            filtered_data = st.session_state.ui_state.get('current_paper_details', pd.DataFrame())

            if not filtered_data.empty:
                st.markdown("###### Paper Details")
                # Display the selected paper's details

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

                mechanisms =  filtered_data['mechanism'].to_list()
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


from functools import lru_cache

@st.cache_data
def load_and_process_filters():
    """Cache the filter loading and tree building to avoid repeated computation"""
    filters = load_filters_json()
    
    # Pre-build trees once and cache them
    study_types_tree = build_tree_optimized(filters['study_types'])
    mechanisms_tree = build_tree_optimized(filters['mechanisms']) 
    behaviors_tree = build_tree_optimized(filters['Behaviors'])
    
    return filters, study_types_tree, mechanisms_tree, behaviors_tree

def build_tree_optimized(data, path=""):
    """Optimized tree builder - reduces string operations and memory allocation"""
    if not data:
        return []
    
    tree = []
    for key, value in data.items():
        node_path = f"{path} > {key}" if path else key
        
        if isinstance(value, dict):
            children = build_tree_optimized(value, node_path)
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
        elif isinstance(value, list):
            # Pre-allocate list and use list comprehension
            children = [
                {"label": item, "value": f"{node_path} > {item}"} 
                for item in value
            ]
            tree.append({
                "label": key,
                "value": node_path,
                "children": children
            })
    return tree

def process_tree_selections(selections, min_depth=2):
    """Optimized selection processing - reduces string splitting operations"""
    if not selections or not selections.get('checked'):
        return []
    
    result = []
    for value in selections['checked']:
        # Split once and reuse
        parts = value.split(' > ')
        if len(parts) >= min_depth:
            result.append(parts[min_depth - 1])
    return result

@st.fragment
def filter_selection_fragment():
    """
    Optimized filter selection fragment with caching and reduced memory usage
    """
    # Load cached data
    filters, study_types_tree, mechanisms_tree, behaviors_tree = load_and_process_filters()
    
    # UI Components
    st.markdown("###### Poverty Contexts")
    selected_contexts = st.multiselect(
        "Select", 
        list(filters['poverty_contexts'].keys()), 
        key="sankey_contexts"
    )
    
    st.markdown("###### Study Types") 
    selected_study_types = tree_select(study_types_tree, key="sankey_study_types")
    
    st.markdown("###### Psychological Mechanisms")
    selected_mechanisms = tree_select(mechanisms_tree, key="sankey_mechanisms")
    
    st.markdown("###### Behavioral Outcomes")
    selected_behaviors = tree_select(behaviors_tree, key="sankey_behaviors")
    
    # Process selections efficiently
    all_selected_context = []
    if selected_contexts:
        # Use extend with generator for memory efficiency
        for key in selected_contexts:
            all_selected_context.extend(filters['poverty_contexts'][key])
    
    # Process tree selections with optimized function
    all_selected_study_types = process_tree_selections(selected_study_types, min_depth=3)
    all_selected_mechanisms = process_tree_selections(selected_mechanisms, min_depth=2)
    all_selected_behaviors = process_tree_selections(selected_behaviors, min_depth=2)
    
    # Create signature more efficiently using frozenset for hashable collections
    current_signature = hash((
        frozenset(all_selected_context),
        frozenset(all_selected_study_types), 
        frozenset(all_selected_mechanisms),
        frozenset(all_selected_behaviors)
    ))
    
    previous_signature = st.session_state.get('filter_signature', 0)
    
    # Only update session state if values changed (reduces memory writes)
    if current_signature != previous_signature:
        st.session_state.update({
            'filters': filters,
            'selected_contexts': selected_contexts,
            'all_selected_context': all_selected_context,
            'all_selected_study_types': all_selected_study_types,
            'all_selected_mechanisms': all_selected_mechanisms,
            'all_selected_behaviors': all_selected_behaviors,
            'filter_signature': current_signature
        })
        
        # Only rerun if this isn't the initial load
        if previous_signature != 0:
            st.rerun()


            

def main():  

    with st.sidebar:
        st.markdown("### System Status")
        
        # Memory Usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 1500:
                st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="Healthy")
            else:
                st.metric("Memory Usage", f"{memory_mb:.0f} MB", delta="High", delta_color="inverse")
                
        except ImportError:
            st.metric("Memory Usage", "N/A", delta="Install psutil")

    monitor_and_clear_cache()

    try: 
        if 'connection_tested' not in st.session_state:
            client = get_healthy_bigquery_client()
            st.session_state.connection_tested = True
    except Exception as e:
        st.error("Database connection failed. Please refresh the page.")
        st.stop()


    # if st.sidebar.button("Test DB Connection"):
    #     client = get_healthy_bigquery_client()
    #     if client._is_client_healthy():
    #         st.sidebar.success("Database connection healthy")
    #     else:
    #         st.sidebar.error("Database connection failed")

  

    # Initialize persistent UI state in session state (small objects only!)
    if 'ui_state' not in st.session_state:
        st.session_state.ui_state = {
            'selected_country': 'All',
            'selected_institution': 'All',
            'sankey_filters': {
                'contexts': [],
                'study_types': [],
                'mechanisms': [], 
                'behaviors': []
            },
            'selected_paper': None,
            'stats_computed': False,
            'paper_details': pd.DataFrame()
        }

    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.title(" Psychology and Economics of Poverty Literature Review Dashboard")
    with col2:
        st.image("streamlit/logo.png", use_container_width=False)

    # Create tabs for workspace functionalities
    tab1, tab2= st.tabs(["Dashboard", "About" ])

    # Introduction Tab
    with tab2:
       
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
       
    # Understanding the Field Tab
    with tab1:

        # Get base data (cached - loads once)
        papers_df = get_papers_data()
        preprocessed_papers = get_preprocessed_papers()

        # Introduction
        st.markdown("""
            In the research landscape below, we offer a guided visual exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
            """)

        st.markdown("##### Data at a Glance")

        # Get countries/institutions (cached)
        countries, all_institutions = process_countries_and_institutions(preprocessed_papers)

        # Create layout with appropriate column widths
        col1, col2 = st.columns([1, 1.4])

        with col1:
            # Filtering expander
            with st.expander("Filter by Country and Institution", expanded=True):
                col3, col4 = st.columns(2)
                
                # Country filter dropdown
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
                
                # Only update if changed
                if selected_country != st.session_state.ui_state['selected_country']:
                    st.session_state.ui_state['selected_country'] = selected_country
                    st.session_state.ui_state['stats_computed'] = False  # Flag for recomputation
                
                # Get institutions for current country (this part can be cached too)
                temp_filtered_df = get_filtered_data(selected_country, 'All')
                
                # Institution filter dropdown (dependent on country selection)
                with col4:
                    # Get institutions for the selected country filter
                    if selected_country != 'All':
                        filtered_institutions = []
                        for inst_str in temp_filtered_df['institutions_list']:
                            inst_list = ast.literal_eval(inst_str)  # Convert string back to list
                            filtered_institutions.extend(inst_list)
                        institutions_list = sorted(list(set(filtered_institutions)))
                    else:
                        institutions_list = all_institutions
                    
                    # Ensure current selection is valid
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
                
                # Only update if changed
                if selected_institution != st.session_state.ui_state['selected_institution']:
                    st.session_state.ui_state['selected_institution'] = selected_institution
                    st.session_state.ui_state['stats_computed'] = False
            
            # Get working data (cached based on filters)
            working_df = get_filtered_data(
                st.session_state.ui_state['selected_country'], 
                st.session_state.ui_state['selected_institution']
            )
            
            # Calculate stats (only if needed)
            if not st.session_state.ui_state['stats_computed']:
                st.session_state.ui_state['current_stats'] = calculate_statistics(
                    working_df, papers_df, 
                    st.session_state.ui_state['selected_country'], 
                    st.session_state.ui_state['selected_institution']
                )
                st.session_state.ui_state['stats_computed'] = True
            
            stats = st.session_state.ui_state['current_stats']
            
            # Display statistics (these won't reload unless filters change)
            with st.expander("Total Number of Papers", expanded=True):
                st.markdown(f"**{stats['total_papers']:,}**")
            
            with st.expander("Research Time Span", expanded=True):
                st.markdown(f"**{stats['date_range']}**")
            
            # Display countries and institutions statistics in two columns
            col3, col4 = st.columns(2)
            
            with col3:
                with st.expander("Countries ", expanded=True):
                    st.markdown(f"**{stats['countries_count']}**")
            
            with col4:
                with st.expander("Institutions", expanded=True):
                    st.markdown(f"**{stats['institutions_count']}**")

                    
        with col2:
            # Bar chart (only recomputes if working_df changes)
            with st.spinner("Generating Bar Chart..."):
                with st.expander(" ", expanded=True):
                    # Use preprocessed institutions list for counting
                    all_institutions_in_filtered = []
                    for inst_str in working_df['institutions_list']:
                        inst_list = ast.literal_eval(inst_str)  # Convert string back to list
                        all_institutions_in_filtered.extend(inst_list)
                    
                    # Count institutions
                    from collections import Counter
                    institution_counts = Counter(all_institutions_in_filtered)
                    
                    # Convert to DataFrame for plotting
                    top_institutions = pd.DataFrame([
                        {'institution': inst, 'count': count} 
                        for inst, count in institution_counts.most_common(10)
                    ])

                    st.markdown("###### Research Institutions -  Number of Publications")
                    if not top_institutions.empty:
                        institution_figure = bar.create(top_institutions, x_column='institution', y_column='count', title=None, coord_flip=True, height= 345)
                        st.plotly_chart(institution_figure, use_container_width=True)
                    else:
                        st.write("No data available for selected filters.")

        st.markdown("#### Connecting Poverty Context, Psychological Mechanisms and Behavior")
        st.markdown("""
        Use the filters below to customize the Sankey diagram. 
        Performance decreases when visualizing a large number of papers.
        """)
            
        col1, col2 = st.columns([1, 6])

        with col1:


            with st.spinner("Applying Filters..."):

                filter_selection_fragment()

                filters = st.session_state.get('filters', {})
                selected_contexts = st.session_state.get('selected_contexts', [])
                all_selected_context = st.session_state.get('all_selected_context', [])
                all_selected_study_types = st.session_state.get('all_selected_study_types', [])
                all_selected_mechanisms = st.session_state.get('all_selected_mechanisms', [])
                all_selected_behaviors = st.session_state.get('all_selected_behaviors', [])
                
                # Get exploded data (cached based on sankey filters)
                working_df_exploded = get_working_df_exploded_cached(
                    st.session_state.ui_state['selected_country'],
                    st.session_state.ui_state['selected_institution'],
                    tuple(all_selected_context),
                    tuple(all_selected_study_types),
                    tuple(all_selected_mechanisms),
                    tuple(all_selected_behaviors)
                )

        # In col2, use working_df_viz for the Sankey
        with col2:
            # Access the filter values from session state (set by the fragment)
            filters = st.session_state.get('filters', {})
            selected_contexts = st.session_state.get('selected_contexts', [])
            all_selected_context = st.session_state.get('all_selected_context', [])
            all_selected_study_types = st.session_state.get('all_selected_study_types', [])
            all_selected_mechanisms = st.session_state.get('all_selected_mechanisms', [])
            all_selected_behaviors = st.session_state.get('all_selected_behaviors', [])
            
            # Get exploded data (cached based on sankey filters)
            working_df_exploded = get_working_df_exploded_cached(
                st.session_state.ui_state['selected_country'],
                st.session_state.ui_state['selected_institution'],
                tuple(all_selected_context),
                tuple(all_selected_study_types),
                tuple(all_selected_mechanisms), 
                tuple(all_selected_behaviors)
            )
            
            with st.spinner("Generating Sankey Diagram..."):   

                # Create columns for the toggle chips
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    show_context = st.checkbox("Poverty Context", value=True, key="node_context")

                with col2:
                    show_study = st.checkbox("Study Type", value=True, key="node_study")

                with col3:
                    show_mechanism = st.checkbox("Psychological Mechanism", value=True, key="node_mechanism")

                with col4:
                    show_behavior = st.checkbox("Behavioral Outcomes", value=True, key="node_behavior")

                # Build selected nodes list
                selected_nodes = []
                if show_context:
                    selected_nodes.append('poverty_context')
                if show_study:
                    selected_nodes.append('study_type')
                if show_mechanism:
                    selected_nodes.append('mechanism')
                if show_behavior:
                    selected_nodes.append('behavior')
                else: 
                    selected_nodes = ['poverty_context', 'study_type', 'mechanism', 'behavior']

                # Prepare active_filters dictionary for the sankey function
                active_filters = {
                    'contexts': selected_contexts if selected_contexts else [],
                    'study_types': all_selected_study_types if all_selected_study_types else [],
                    'mechanisms': all_selected_mechanisms if all_selected_mechanisms else [],
                    'behaviors': all_selected_behaviors if all_selected_behaviors else []
                }
                
                # Create and display the Sankey diagram with adaptive detail
                sankey_diagram = sankey.Sankey(filters_json = filters)

                if not working_df_exploded.empty:
                    sankey_fig = sankey_diagram.draw(working_df_exploded, 
                                                    active_filters=active_filters,
                                                    columns_to_show = selected_nodes)
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.write("No data available for selected filters.")

            # IMPORTANT: Calculate filter signature BEFORE the Research Landscape section
            # This prevents the visualization from recalculating when only paper selection changes
            current_filter_signature = f"{st.session_state.ui_state['selected_country']}_{st.session_state.ui_state['selected_institution']}_{tuple(all_selected_context)}_{tuple(all_selected_study_types)}_{tuple(all_selected_mechanisms)}_{tuple(all_selected_behaviors)}"
            
            # Only recompute visualization data when filters actually change
            if ('current_filter_signature' not in st.session_state.ui_state or 
                st.session_state.ui_state['current_filter_signature'] != current_filter_signature):
                
                # Store new signature
                st.session_state.ui_state['current_filter_signature'] = current_filter_signature
                
                # Get UMAP data and filter it (only when filters change)
                umap_data = get_umap_data()
                plot_df = umap_data[umap_data['doi'].isin(working_df_exploded['doi'].tolist())]
                topics_df = load_topics()
                
                # Cache the visualization data
                st.session_state.ui_state['cached_plot_df'] = plot_df
                st.session_state.ui_state['cached_topics_df'] = topics_df
                st.session_state.ui_state['current_papers_list'] = plot_df['title'].tolist() if not plot_df.empty else []
          

        st.markdown("#### Research Landscape")

        # REORDERED: Heatmap first (col1), then paper details (col2)
        col1, col2 = st.columns([2, 1])

        # Use cached data for visualizations (completely separate from paper selection)
        plot_df = st.session_state.ui_state.get('cached_plot_df', pd.DataFrame())
        topics_df = st.session_state.ui_state.get('cached_topics_df', pd.DataFrame())

        with col1:
            # UMAP visualization (only regenerates when filter signature changes)
            if not plot_df.empty:
                with st.spinner("Generating research landscape visualization..."):
                    plot_df['UMAP1'] = pd.to_numeric(plot_df['UMAP1'], errors='coerce')
                    plot_df['UMAP2'] = pd.to_numeric(plot_df['UMAP2'], errors='coerce')
                    topics_df['umap_1_mean'] = pd.to_numeric(topics_df['umap_1_mean'], errors='coerce')
                    topics_df['umap_2_mean'] = pd.to_numeric(topics_df['umap_2_mean'], errors='coerce')
                    
                    from visuals import scatterplot
                    heatmap_fig = heatMap.heatmap(plot_df, topics_df)
                    st.plotly_chart(heatmap_fig.draw(), use_container_width=True)
            else:
                st.write("No data available for heatmap with current filters.")
            
            # Main explanation with better terminology
            st.markdown("""            
            This visualization creates a **living map** of academic research, where similar studies naturally cluster together like neighborhoods in a city. 
            Watch how knowledge evolves, new ideas emerge, and research communities form over time.
            """)
            
            # Enhanced instruction columns
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
            
            # Advanced insights section
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
            
            # Technical note
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

        with col2:
            paper_details_fragment()
            

if __name__ == "__main__":
    main()