# app.py
import streamlit as st
import pandas as pd
import os 
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objs as go
from streamlit_gsheets import GSheetsConnection


import ast
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python module search path
sys.path.append(parent_dir)
from visuals import heat_map, bar, sankey
from featureEngineering import encoder, labels, topicClusters




st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

def main():    

    # Load data function
    @st.cache_data
    def load_data():

        conn = st.connection("gsheets", type=GSheetsConnection)
        papers_url = 'https://docs.google.com/spreadsheets/d/1nrZC6zJ50DouMCHIWZOl-tQ4BAZfEojJymtzUh26nP0/edit?usp=sharing'
        topics_url = 'https://docs.google.com/spreadsheets/d/1cspghq8R0Xlf2jk0TacGv8bC6C4EGIUnGuUQJWYASpk/edit?usp=sharing'

        papers_df = conn.read(spreadsheet = papers_url)
        topics_df = conn.read(spreadsheet= topics_url)

        return papers_df, topics_df

    papers_df, topics_df = load_data()
    # Set page title
    st.title(" Psychology of Poverty Literature Review Dashboard")

    # Create tabs for workspace functionalities
    tab1, tab2= st.tabs(["Introduction", "Dashboard" ])

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

                    This project is being developed by the **Psychology and Economics of Poverty Initiative (PEP)** at the **Center for Effective Global Action (CEGA)** at Berkeley.

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
            Our dataset encompasses academic scholarship across diverse disciplines and institutional sources spanning multiple geographic regions, providing a multi-faceted lens into contemporary poverty studies.

            Our goal is to map the intellectual terrain of poverty research in a manner that can be explored. Through this exploratory dashboard, we hope to illuminate significant scholarly contributions, trace emerging research trends, and encourage more interdisciplinary collaboration around poverty alleviation and social development.

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
                with st.expander("Countries of Institutions Represented ", expanded=True):
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
        
 
        with st.expander("Filter by Context and Mechanism", expanded=True):
            # Initial filtering based on working_df
            filter_dois = working_df['doi'].tolist()
            filtered_papers = papers_df[papers_df['doi'].isin(filter_dois)]
            # remove rows with 'Insufficient info' in poverty_context and drop None values and drop if values are less than 5
            # filtered_papers = filtered_papers[filtered_papers['poverty_context'].map(filtered_papers['poverty_context'].value_counts()) ]
            filtered_papers = filtered_papers[filtered_papers['poverty_context'] != 'Insufficient info']
            filtered_papers = filtered_papers[filtered_papers['poverty_context'].notnull()]

            # remove rows with 'Insufficient info' in mechanism
            # filtered_papers = filtered_papers[filtered_papers['mechanism'].map(filtered_papers['mechanism'].value_counts()) > 5]
            filtered_papers = filtered_papers[filtered_papers['mechanism'] != 'Insufficient info']
            filtered_papers = filtered_papers[filtered_papers['mechanism'].notnull()]
            # remove rows with 'Insufficient info' in study_type
            # filtered_papers = filtered_papers[filtered_papers['study_type'].map(filtered_papers['study_type'].value_counts()) > 5]
            filtered_papers = filtered_papers[filtered_papers['study_type'] != 'Insufficient info']
            filtered_papers = filtered_papers[filtered_papers['study_type'].notnull()]
            
            # Get unique values for dropdowns (excluding 'Insufficient info')
            # explode dictionary by poverty_context

            context_df = filtered_papers.copy()
            context_df['poverty_context'] = context_df['poverty_context'].str.split(',')
            context_df = context_df.explode('poverty_context')
            # remove leading and trailing spaces
            context_df['poverty_context'] = context_df['poverty_context'].str.strip()
            context_list = sorted([str(x) for x in context_df[context_df['poverty_context'] != 'Insufficient info']['poverty_context'].unique()])
            context_list = context_df['poverty_context'].values
            context_list = list(set(context_list))
            # extract value from inner list

            mechanisms_df = filtered_papers.copy()
            mechanisms_df['mechanism'] = mechanisms_df['mechanism'].str.split(',')
            mechanisms_df = mechanisms_df.explode('mechanism')
            # remove leading and trailing spaces
            mechanisms_df['mechanism'] = mechanisms_df['mechanism'].str.strip() 
            mechanisms_list = sorted([str(x) for x in mechanisms_df[mechanisms_df['mechanism'] != 'Insufficient info']['mechanism'].unique()]) 
            mechanisms_list = mechanisms_df['mechanism'].values     
            mechanisms_list = list(set(mechanisms_list))       

            study_types_df = filtered_papers.copy()
            study_types_df['study_type'] = study_types_df['study_type'].str.split(',')
            study_types_df = study_types_df.explode('study_type')
            # remove leading and trailing spaces
            study_types_df['study_type'] = study_types_df['study_type'].str.strip()
            study_types_list = sorted([str(x) for x in study_types_df[study_types_df['study_type'] != 'Insufficient info']['study_type'].unique()])
            study_types_list = study_types_df['study_type'].values
            study_types_list = list(set(study_types_list))
            
            
            # Create multiselect filters
            col3, col4, col5 = st.columns(3)
            
            with col3:
                selected_contexts = st.multiselect(
                    "Filter by Context", 
                    context_list,
                    default=[],  # Start with no selections (shows all)
                    placeholder="Select one or more contexts"
                )
            
            with col4:
                selected_study_types = st.multiselect(
                    "Filter by Study Type", 
                    study_types_list,
                    default=[],  # Start with no selections (shows all)
                    placeholder="Select one or more study types"
                )
            
            with col5:
                selected_mechanisms = st.multiselect(
                    "Filter by Mechanism", 
                    mechanisms_list,
                    default=[],  # Start with no selections (shows all)
                    placeholder="Select one or more mechanisms"
                )

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
            
            # Apply filters - if nothing selected, show all
            if selected_contexts:
                working_df_exploded = working_df_exploded[working_df_exploded['poverty_context'].isin(selected_contexts)]
                working_df = working_df[working_df['doi'].isin(working_df_exploded['doi'].tolist())]

            
            if selected_mechanisms:
                working_df_exploded = working_df_exploded[working_df_exploded['mechanism'].isin(selected_mechanisms)]
                working_df = working_df[working_df['doi'].isin(working_df_exploded['doi'].tolist())]
            
            if selected_study_types:
                working_df_exploded = working_df_exploded[working_df_exploded['study_type'].isin(selected_study_types)]
                working_df = working_df[working_df['doi'].isin(working_df_exploded['doi'].tolist())]
            
            # Create summaries
            context_sum = working_df.groupby('poverty_context').size().reset_index(name='count')
            mechanisms_sum = working_df.groupby('mechanism').size().reset_index(name='count')
            

        col1, col2 = st.columns([1  , 6])

        # with col1:  
 
        #     st.markdown("""

        #             Poverty Context includes factors like low resource levels (absolute, perceived, or relative), resource volatility (unpredictable shocks), the physical environment (violence, noise, neighborhood quality), and the social environment (stigma, discrimination, cultural norms).

        #             Psychological Mechanisms are mental processes shaped by poverty. Affective factors (anxiety, depression, stress) influence emotions. Beliefs about self-worth and aspirations affect motivation, while cognitive functions (memory, executive control) are hindered by poverty. Preferences, such as time and risk preferences, are shaped by immediate needs.

        #             """)
            
        # with col2:

      
        

        # Create and display the Sankey diagram
        sankey_fig = sankey.draw(working_df_exploded)
        st.plotly_chart(sankey_fig, use_container_width=True)

        st.markdown("#### Research Landscape Visualized")

        
        col1, col2 = st.columns([2, 1])

        with col1:
            with st.spinner("Generating research landscape visualization..."):
                plot_df = working_df.copy()
                topics_df = topics_df.copy()
                visual = heat_map.create_cumulative_visualization(plot_df, topics_df)
                st.plotly_chart(visual, use_container_width=True)

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
                if len(mechanisms) > 1:
                    mechanisms = ', '.join(mechanisms)
                elif len(mechanisms) == 1:
                    mechanisms = mechanisms[0]
                else:
                    mechanisms = "None"
                

                selected_paper_details = working_df[working_df['title'] == selected_paper].iloc[0]
                st.markdown(f"**Title:** {selected_paper_details['title']}")
                st.markdown(f"**Context:** {context}")
                st.markdown(f"**Mechanism:** {mechanisms}")
                st.markdown(f"**Study Type:** {study_types}")
                st.markdown(f"**Authors:** {authors}")
                st.markdown(f"**Abstract:** {selected_paper_details['abstract']}")
              





        

            
            

            


if __name__ == "__main__":
    main()