import streamlit as st
import pandas as pd
import os 
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objs as go
from streamlit_tree_select import tree_select
from PIL import Image
from dotenv import load_dotenv
import ast
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python module search path
sys.path.append(parent_dir)
from visuals import heat_map, bar, sankey
from featureEngineering import encoder, labels, topicClusters
from streamlit_gsheets import GSheetsConnection
from data.googleSheets import API


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

get_env_var = load_environment_variables()

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

st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

google_sheets_connection = API(credentials_json=credentials)

# Cache spreadsheets in session state to avoid reloading every time
if 'papers_df' not in st.session_state or st.session_state.papers_df is None:
    with st.spinner("Loading data from Google Sheets..."):
        st.session_state.papers_df = google_sheets_connection.read(
            spreadsheet_id=spreadsheet_ids['papers']
        )

if 'topics_df' not in st.session_state or st.session_state.topics_df is None:
    with st.spinner("Loading topics data..."):
        st.session_state.topics_df = google_sheets_connection.read(
            spreadsheet_id=spreadsheet_ids['topics']
        )


def main():    

    papers_df = st.session_state.papers_df
    topics_df = st.session_state.topics_df

    col1, col2 = st.columns([4, 1.5])
    with col1:
        st.title(" Psychology of Poverty Literature Review Dashboard")
    with col2:
        st.image("streamlit/logo.png", use_container_width=False)

    # Create tabs for workspace functionalities
    tab1, tab2= st.tabs(["About", "Dashboard"])

    # Introduction Tab
    with tab1:
       
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

                    Special thanks to **Kellie Hogue** at UC Berkeley’s D-Lab and **Van Tran**.

                    ---
                    If you are a researcher or project manager interested in adapting this tool for your field, visit the **Documentation** tab to learn about the technology behind it and how to implement it in your research domain.

                    Comments, questions, ideas: Kristina Hallez, Senior Research Manager, khallez@berkeley.edu
                    
                   

         """)
       
       # add logo
    # st.image("logo.png", width=200)


    # Understanding the Field Tab
    with tab2:

        # Introduction
        st.markdown("""
            In this research landscape analysis, we offer a guided exploration of research on the psychology poverty research through an interactive data dashboard. Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.
            """)

        
        st.markdown("##### Data at a Glance")

        # Initialize session state variables if they don't exist
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = 'All'
        if 'selected_institution' not in st.session_state:
            st.session_state.selected_institution = 'All'

        # Extract unique countries from data (removing None values)e
        
        country_filter = papers_df.copy()
        country_filter['country'] = country_filter['country'].apply(
            lambda x: list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None]))
        )
        countries = country_filter['country'].explode().unique()
        countries = [str(country) for country in countries if country is not None]
        all_countries_count = len(countries)

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
                        ['All'] + sorted(countries),
                        key="country_selector"
                    )
                    
                    # Store the selected country in session state
                    st.session_state.selected_country = selected_country
                
                # Start with the original data each time
                working_df = papers_df.copy()
                
                # Apply country filter if not "All"
                if selected_country != 'All':
                    working_df = working_df[
                        working_df['country'].apply(lambda x: selected_country in ast.literal_eval(x))
                    ]
                
                # Institution filter dropdown (dependent on country selection)
                with col4:
                    # Get institutions for the selected country filter
                    institution_filter = working_df.copy()
                    institution_filter['institution'] = institution_filter['institution'].apply(
                        lambda x: list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None]))
                    )
                    filtered_institutions = institution_filter['institution'].explode().unique()
                    institutions_list = [str(inst) for inst in filtered_institutions if inst is not None]
                    
                    selected_institution = st.selectbox(
                        "Filter by Institution", 
                        ['All'] + sorted(institutions_list),
                        key="institution_selector"
                    )
                    
                    # Store the selected institution in session state
                    st.session_state.selected_institution = selected_institution
                
                # Apply institution filter if not "All"
                if selected_institution != 'All':
                    working_df = working_df[
                        working_df['institution'].apply(lambda x: selected_institution in ast.literal_eval(x))
                    ]
            
            # Filter publications based on the DOIs in the working dataframe
            filtered_publications = papers_df[papers_df['doi'].isin(working_df['doi'].tolist())]
            
            # Display statistics based on filtered data
            with st.expander("Total Number of Papers", expanded=True):
                st.markdown(f"**{len(filtered_publications):,}**")
            
            with st.expander("Research Time Span", expanded=True):
                if not filtered_publications.empty:
                    min_date = filtered_publications['date'].min()
                    max_date = filtered_publications['date'].max()
                    st.markdown(f"**{min_date} – {max_date}**")
                else:
                    st.markdown("**No data available**")
            
            # Display countries and institutions statistics in two columns
            col3, col4 = st.columns(2)
            
            with col3:
                with st.expander("Countries of Institutions ", expanded=True):
                    # Calculate countries count based on current filter
                    if selected_country != 'All':
                        countries_count = 1  # If filtering by a specific country
                    else:
                        # Count unique countries in the filtered dataset
                        filtered_country_data = working_df.copy()
                        filtered_country_data['country'] = filtered_country_data['country'].apply(
                            lambda x: list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None]))
                        )
                        filtered_countries = filtered_country_data['country'].explode().unique()
                        filtered_countries = [str(c) for c in filtered_countries if c is not None]
                        countries_count = len(filtered_countries)
                    st.markdown(f"**{countries_count}**")
            
            with col4:
                with st.expander("Institutions Represented", expanded=True):
                    # Calculate institutions count based on current filter
                    if selected_institution != 'All':
                        institutions_count = 1  # If filtering by a specific institution
                    else:
                        # Count unique institutions in the filtered dataset
                        filtered_inst_data = working_df.copy()
                        filtered_inst_data['institution'] = filtered_inst_data['institution'].apply(
                            lambda x: list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None]))
                        )
                        filtered_institutions = filtered_inst_data['institution'].explode().unique()
                        filtered_institutions = [str(inst) for inst in filtered_institutions if inst is not None]
                        institutions_count = len(filtered_institutions)
                    st.markdown(f"**{institutions_count}**")

                
        with col2:
            # Preprocess and visualize top institutions
            with st.expander(" ", expanded=True):
                exploded_institutions = working_df.copy()
                exploded_institutions['institution'] = exploded_institutions['institution'].apply(lambda x: list(dict.fromkeys([i for i in ast.literal_eval(x) if i is not None])))
                exploded_institutions = exploded_institutions.explode('institution')
               
                institution_counts = exploded_institutions.groupby('institution').size().reset_index(name='count')
                top_institutions = institution_counts.sort_values('count', ascending=False).head(10)

                st.markdown("###### Research Institutions -  Number of Publications")
                institution_figure = bar.create(top_institutions, x_column='institution', y_column='count', title=None, coord_flip=True, height= 345)
                st.plotly_chart(institution_figure, use_container_width=True)

                

        st.markdown("#### Connecting Poverty Context, Psychological Mechanisms and Behavior")
        st.markdown("""
                    Use the filters to select the different poverty context and psychological mechanisms you are interested in exploring in the literature.
                    """)
        
            

        col1, col2 = st.columns([1  , 6])

        import json
        with col1:
            with open('data/trainingData/labels.json', 'r') as f:
                filters = json.load(f)
                # st.write(testing)

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

            

            working_df_exploded = working_df.copy()
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

            working_df = working_df[working_df['doi'].isin(working_df_exploded['doi'].tolist())]       

        # In col2, use working_df_viz for the Sankey
        with col2:

            # Simple toggle chips for node selection
            # st.write("**Flow Configuration:**")

            # st.markdown(" ")

            # Create columns for the toggle chips
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                show_context = st.checkbox("Context", value=True, key="node_context")

            with col2:
                show_study = st.checkbox("Study Type", value=True, key="node_study")

            with col3:
                show_mechanism = st.checkbox("Mechanism", value=True, key="node_mechanism")

            with col4:
                show_behavior = st.checkbox("Behavior", value=True, key="node_behavior")

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

            sankey_fig = sankey_diagram.draw(working_df_exploded, 
                                             active_filters=active_filters,
                                             columns_to_show = selected_nodes)

            # sankey_fig = sankey.draw(
            #     working_df_exploded, 
            #     filters_json=filters, 
            #     active_filters=active_filters,
            #     manual_detail_level=None,

            # )
            st.plotly_chart(sankey_fig, use_container_width=True)

        st.markdown("#### Research Landscape Visualized")

        
        col1, col2 = st.columns([2, 1])

        # plot_df = working_df.copy()

        with col1:
            with st.spinner("Generating research landscape visualization..."):

                st.markdown("## Placeholder until we have all the data loaded")
                plot_df = working_df.copy()
                # plot_df['UMAP1'] = pd.to_numeric(plot_df['UMAP1'], errors='coerce')
                # plot_df['UMAP2'] = pd.to_numeric(plot_df['UMAP1'], errors='coerce')
                # topics_df = topics_df.copy()
                # visual = heat_map.create_cumulative_visualization(plot_df, topics_df)
                # st.plotly_chart(visual, use_container_width=True)

            st.markdown("""
            This visualization maps the research landscape over time, revealing how academic topics cluster, evolve, and gain momentum. Use it to discover emerging research areas, track the changing focus of scholarly work, and identify influential research communities.
            """)

            col4, col5 = st.columns(2)
            with col4:
                st.markdown("""
                **What it shows:**
                - Each white dot = one research paper positioned by content (You can hover over it to see title)
                - Color intensity = density of related research (Purple = sparse, bright yellow = concentrated)
                - Spatial proximity = thematic similarity between papers
                - White text labels = major research themes and topic clusters
                - Overall layout = the intellectual landscape of the field
            """)
            with col5:
                st.markdown("""
                **How to use:**
                - Drag the year slider to animate research evolution over time
                - Hover over individual dots to see paper titles and details
                - Observe clusters forming, growing, merging, or dissolving
                - Identify research hotspots (bright yellow concentrations)
                - Track how topics migrate and transform across the map
                - Note emerging areas at the periphery moving toward the center
                """)



        with col2:

            # st.markdown("###### Research Landscape Visualized")

            with st.expander("Number of Papers Visualized", expanded=True):

                st.markdown(f"**{len(plot_df):,}**")

            # with st.expander("Number of Research Clusters", expanded=True):
            #     st.markdown(f"**{len(clusters):,}**")

            # drop down of to select paper 
            with st.expander("Select Paper", expanded=True):
                selected_paper = st.selectbox(
                    "Select Paper", 
                    plot_df['title'].tolist(),
                    key="paper_selector"
                )

                # seelct doi 
                selected_doi = working_df[working_df['title'] == selected_paper]['doi'].values[0]

                # create vertical table of paper, abstract, mechanism and context 
                st.markdown("###### Paper Details")
                # Display the selected paper's details

                authors = working_df[working_df['doi'] == selected_doi]['authors'].to_list()


                authors = ast.literal_eval(authors[0])
                authors = [author for author in authors if author != 'Insufficient info']
                authors = ', '.join(authors)

                context = working_df[working_df['doi'] == selected_doi]['poverty_context'].values
                context = list(set(context))
                context = [c for c in context if c != 'Insufficient info']
                if len(context) > 1:
                    context = ', '.join(context)
                elif len(context) == 1:
                    context = context[0]
                else:
                    context = "None"
                study_types = working_df[working_df['doi'] == selected_doi]['study_type'].values
                study_types = [study for study in study_types if study != 'Insufficient info']
                study_types = list(set(study_types))
                if len(study_types) > 1:
                    study_types = ', '.join(study_types)
                elif len(study_types) == 1:
                    study_types = study_types[0]
                else:
                    study_types = "None"

                mechanisms = working_df[working_df['doi'] == selected_doi]['mechanism'].values
                mechanisms = [m for m in mechanisms if m != 'Insufficient info']
                mechanisms = list(set(mechanisms))

                behavior = working_df[working_df['doi'] == selected_doi]['behavior'].values
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
                

                selected_paper_details = working_df[working_df['title'] == selected_paper].iloc[0]
                st.markdown(f"**Title:** {selected_paper_details['title']}")
                st.markdown(f"**Authors:** {authors}")
                st.markdown(f"**Context:** {context}")
                st.markdown(f"**Study Type:** {study_types}")
                st.markdown(f"**Mechanism:** {mechanisms}")
                st.markdown(f"**Behavior:** {behavior}")
                st.markdown(f"**Abstract:** {selected_paper_details['abstract']}")


if __name__ == "__main__":
    main()