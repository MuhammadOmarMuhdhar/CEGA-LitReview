import streamlit as st
import pandas as pd
import os 
import sys
import plotly.graph_objs as go
from streamlit_tree_select import tree_select
from dotenv import load_dotenv
import ast
import json
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from visuals import bar, sankey, heatMap
from data.bigQuery import Client

@st.cache_data
def load_filters_json():
    """Load filters JSON once and cache it"""
    with open('data/trainingData/labels.json', 'r') as f:
        return json.load(f)

# NEW: Pre-process expensive operations once
@st.cache_data(show_spinner="Processing data...")
def preprocess_papers_data(papers_df):
    """One-time expensive preprocessing to avoid repeated ast.literal_eval calls"""
    df = papers_df.copy()
    
    # Pre-parse institutions (convert lists to strings for caching)
    df['institutions_list'] = df['institution'].apply(
        lambda x: str(list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None])))
    )
    
    # Pre-parse countries (convert lists to strings for caching)
    df['countries_list'] = df['country_of_study'].apply(
        lambda x: str([i.strip() for i in str(x).split(',') if i.strip() and i.strip().lower() != 'nan'])
    )
    
    return df

# Cache expensive data processing operations - NOW MUCH FASTER
@st.cache_data(show_spinner="Loading Filters...")
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

@st.cache_data(show_spinner="Loading Metadata Filters...")
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

@st.cache_data(show_spinner="Loading Metadata...")
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

# Initialize environment variables handler
if 'environment_variables' not in st.session_state:
    st.session_state.environment_variables = load_environment_variables()

get_env_var = st.session_state.environment_variables

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
    
    email = get_env_var("USERNAME")
    password = get_env_var("PASSWORD")
    
    return api_key, credentials, email, password

# Initialize and load configuration only once
if 'configuration' not in st.session_state:
    st.session_state.configuration = get_configuration()

# Assign variables from cached configuration
api_key, credentials, email, password = st.session_state.configuration

st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

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

@st.cache_resource
def get_bigquery_client():
    try:
        return Client(credentials, 'literature-452020')
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

@st.cache_data(show_spinner="Connecting to Database...")
def load_country_institution_data():
    try:
        client = get_bigquery_client()
        return client.execute_query(
            "SELECT doi, country, date, institution, country_of_study "
            "FROM `literature-452020.psychology_of_poverty_literature.papers`"
        )
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()

@ st.cache_data(show_spinner="Loading Sankey Diagram Data...")
def load_label_data():
    try:
        client = get_bigquery_client()
        return client.execute_query(
            "SELECT doi, authors, study_type, poverty_context, "
            "mechanism, behavior "
            "FROM `literature-452020.psychology_of_poverty_literature.papers`"
        )
    except Exception as e:
        st.error(f"Failed to load label data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading Topics...")
def load_topics():
    try:
        client = get_bigquery_client()
        return client.execute_query(
            "SELECT * "
            "FROM `literature-452020.psychology_of_poverty_literature.topics`"
        )
    except Exception as e:
        st.error(f"Failed to load topics: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading Abstract Data...")
def load_abstract_data(title):
    try:
        client = get_bigquery_client()
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

@st.cache_data(show_spinner="Loading Heat Map Data...")
def load_umap():
    try:
        client = get_bigquery_client()
        return client.execute_query(
            f"""
            SELECT  title, doi, UMAP1, UMAP2, date
            FROM `literature-452020.psychology_of_poverty_literature.papers`
            """
        )
    except Exception as e:
        st.error(f"Failed to load UMAP data: {str(e)}")
        return pd.DataFrame()

def main():  

    try:
        if 'connection_tested' not in st.session_state:
            get_bigquery_client()
            st.session_state.connection_tested = True
    except Exception as e:
        st.error("Database connection failed. Please refresh the page.")
        st.stop()  

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

        # Load and preprocess data
        if 'papers_df' not in st.session_state:
            st.session_state.papers_df = load_country_institution_data()

        # NEW: Preprocess the data once for lightning-fast filtering
        if 'preprocessed_papers' not in st.session_state:
            st.session_state.preprocessed_papers = preprocess_papers_data(st.session_state.papers_df)

        papers_df = st.session_state.papers_df
        preprocessed_papers = st.session_state.preprocessed_papers

        # Introduction
        st.markdown("""
            In the research landscape below, we offer a guided visual exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
            """)

        st.markdown("##### Data at a Glance")

        # Initialize session state variables if they don't exist
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = 'All'
        if 'selected_institution' not in st.session_state:
            st.session_state.selected_institution = 'All'

        # Get processed countries and institutions (cached) - now using preprocessed data
        countries, all_institutions = process_countries_and_institutions(preprocessed_papers)

        # Create layout with appropriate column widths
        col1, col2 = st.columns([1, 1.4])

        with col1:
            # Filtering expander
            with st.expander("Filter by Country and Institution", expanded=True):
                col3, col4 = st.columns(2)
                
                # Country filter dropdown
                with col3:
                    selected_country = st.selectbox(
                        "Filter by Country", 
                        ['All'] + countries,
                        key="country_selector"
                    )
                    
                    # Store the selected country in session state
                    st.session_state.selected_country = selected_country
                
                # Get filtered data based on country selection (for institution dropdown)
                temp_filtered_df = lightning_fast_filter(preprocessed_papers, selected_country, 'All')
                
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
                    
                    selected_institution = st.selectbox(
                        "Filter by Institution", 
                        ['All'] + institutions_list,
                        key="institution_selector"
                    )
                    
                    # Store the selected institution in session state
                    st.session_state.selected_institution = selected_institution
            
            # Get final filtered data using lightning-fast filtering
            working_df = lightning_fast_filter(preprocessed_papers, selected_country, selected_institution)
            
            # Calculate statistics (cached)
            stats = calculate_statistics(working_df, papers_df, selected_country, selected_institution)
            
            # Display statistics
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
                with st.expander("Institutions Represented", expanded=True):
                    st.markdown(f"**{stats['institutions_count']}**")

                    
        with col2:
            # Preprocess and visualize top institutions - now much faster!
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
            
        col1, col2 = st.columns([1  , 6])

        if "labels_data" not in st.session_state:
            st.session_state.labels_data = load_label_data()
        with col1:

            filters = load_filters_json()

            def build_tree(data, path=""):
                tree = []
                for key, value in data.items():
                    node_path = f"{path} > {key}" if path else key
                    if isinstance(value, dict):
                        children = build_tree(value, node_path)
                        tree.append({
                            "label": key,
                            "value": node_path,
                            "children": children
                        })
                    elif isinstance(value, list):
                        children = [{"label": item, "value": f"{node_path} > {item}"} for item in value]
                        tree.append({
                            "label": key,
                            "value": node_path,
                            "children": children
                        })
                return tree
            
            st.markdown("###### Poverty Contexts")
            selected_contexts = st.multiselect("Select", list(filters['poverty_contexts'].keys()))
            st.markdown("###### Study Types")
            selected_study_types = tree_select(build_tree(filters['study_types']))
            st.markdown("###### Psychological Mechanisms")
            selected_mechanisms = tree_select(build_tree(filters['mechanisms']))
            st.markdown("###### Behavioral Outcomes")
            selected_behaviors = tree_select(build_tree(filters['Behaviors']))

            all_selected_context = []
            for key in selected_contexts:
                values = filters['poverty_contexts'][key]
                all_selected_context.append(values)

            all_selected_study_types = []
            for value in selected_study_types['checked']:
                value_list = (list(value.split(' > ')))
                if len(value_list) < 3:
                    continue
                else: 
                    all_selected_study_types.append(value_list[2])

            all_selected_mechanisms = []
            for value in selected_mechanisms['checked']:
                value_list = (list(value.split(' > ')))
                if len(value_list) < 2:
                    continue
                else: 
                    all_selected_mechanisms.append(value_list[1])

            all_selected_behaviors = []
            for value in selected_behaviors['checked']:
                value_list = (list(value.split(' > ')))
                if len(value_list) < 2:
                    continue
                else: 
                    all_selected_behaviors.append(value_list[1])

            labels_df = st.session_state.labels_data
            sankey_working_df = labels_df[labels_df['doi'].isin(working_df['doi'].tolist())]           
            
            working_df_exploded = sankey_working_df.copy()
            working_df_exploded['poverty_context'] = working_df_exploded['poverty_context'].str.split(',')
            working_df_exploded = working_df_exploded.explode('poverty_context')
            # remove leading and trailing spaces
            working_df_exploded['poverty_context'] = working_df_exploded['poverty_context'].str.strip()
            working_df_exploded['mechanism'] = working_df_exploded['mechanism'].str.split(',')
            working_df_exploded = working_df_exploded.explode('mechanism')
            # remove leading and trailing spaces
            working_df_exploded['mechanism'] = working_df_exploded['mechanism'].str.strip()
            working_df_exploded['study_type'] = working_df_exploded['study_type'].str.split(',')
            working_df_exploded = working_df_exploded.explode('study_type')
            # remove leading and trailing spaces
            working_df_exploded['study_type'] = working_df_exploded['study_type'].str.strip()
            working_df_exploded['behavior'] = working_df_exploded['behavior'].str.split(',')
            working_df_exploded = working_df_exploded.explode('behavior')
            # remove leading and trailing spaces
            working_df_exploded['behavior'] = working_df_exploded['behavior'].str.strip()

            if all_selected_context:
                working_df_exploded = working_df_exploded[working_df_exploded['poverty_context'].isin(all_selected_context)]
            if all_selected_study_types:
                working_df_exploded = working_df_exploded[working_df_exploded['study_type'].isin(all_selected_study_types)]
            if all_selected_mechanisms:
                working_df_exploded = working_df_exploded[working_df_exploded['mechanism'].isin(all_selected_mechanisms)]
            if all_selected_behaviors:
                working_df_exploded = working_df_exploded[working_df_exploded['behavior'].isin(all_selected_behaviors)]

            sankey_working_df = sankey_working_df[sankey_working_df['doi'].isin(working_df_exploded['doi'].tolist())]       

        # In col2, use working_df_viz for the Sankey
        with col2:
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

        st.markdown("#### Research Landscape")

        col1, col2 = st.columns([2, 1])

        if 'umap_data' not in st.session_state:
                st.session_state.umap_data = load_umap()

        umap_data = st.session_state.umap_data
        # Filter UMAP data to only include papers in the working_df
        plot_df = umap_data[umap_data['doi'].isin(working_df_exploded['doi'].tolist())]
        topics_df = load_topics()

        with col2:
            with st.spinner("Loading Research Paper Details"):  
                working_df_copy = st.session_state.labels_data
                working_df_copy = working_df_copy[working_df_copy['doi'].isin(working_df_exploded['doi'].tolist())]

                with st.expander("Number of Papers Visualized", expanded=True):
                    st.markdown(f"**{len(working_df_copy):,}**")

                # drop down of to select paper 
                with st.expander("Select Paper", expanded=True):
                    if not plot_df.empty:
                        selected_paper = st.selectbox(
                            "Select Paper", 
                            plot_df['title'].tolist(),
                            key="paper_selector"
                        )

                        filtered_data = load_abstract_data(selected_paper)

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

        with col1:
            with st.spinner("Generating research landscape visualization..."):
                # if not plot_df.empty and not topics_df.empty:
                plot_df['UMAP1'] = pd.to_numeric(plot_df['UMAP1'], errors='coerce')
                plot_df['UMAP2'] = pd.to_numeric(plot_df['UMAP2'], errors='coerce')
                topics_df['umap_1_mean'] = pd.to_numeric(topics_df['umap_1_mean'], errors='coerce')
                topics_df['umap_2_mean'] = pd.to_numeric(topics_df['umap_2_mean'], errors='coerce')
                
                from visuals import scatterplot
                heatmap_fig = heatMap.heatmap(plot_df, topics_df)
                st.plotly_chart(heatmap_fig.draw(), use_container_width=True)
            
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



if __name__ == "__main__":
    main()