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
from visuals import heat_map, bar
from featureEngineering import encoder, labels, topicClusters




st.set_page_config(page_title="Workspace", layout="wide", initial_sidebar_state='collapsed')

def main():    

    # Load data function
    @st.cache_data
    def load_data():

        conn = st.connection("gsheets", type=GSheetsConnection)
        publications_url = 'https://docs.google.com/spreadsheets/d/1hCEGlef7wNGi1mk7jBLYfGz8mgw0kBJS78UUEd11sGU/edit?usp=sharing'
        context_url = 'https://docs.google.com/spreadsheets/d/12JzU4gqCwwNIhD646m8daMHXmsYQ7YMKHMwKbn1o75A/edit?usp=sharing'
        mechanisms_url = 'https://docs.google.com/spreadsheets/d/1b0bNul7tbueTFx2JU9t5lyfAnYzxQYef59UarfpIVD0/edit?usp=sharing'
        institutions_url = 'https://docs.google.com/spreadsheets/d/16hm7heBfiCLKI-PNXUOaGZLVtNoDWP4UrRbCA4_K_Qs/edit?usp=sharing'
        authors_url = 'https://docs.google.com/spreadsheets/d/1bC4R3-qoG2jd1QYU1UOIucA-fDQM71_fhjo72p24WjU/edit?usp=sharing'
        citations_url = 'https://docs.google.com/spreadsheets/d/1iqbtpc7O4DsUd4my9YAAlAEzGhHIfASVyEXiB4VrEpk/edit?usp=sharing'
        umap_df_url = 'https://docs.google.com/spreadsheets/d/13qrr95xnBanScDrW4qWYxumJW1s6pJbxZ8wz-0IIuyo/edit?usp=sharing'
        clusters_url = 'https://docs.google.com/spreadsheets/d/1tIDFq2aF_3Cc4nlRuXQ7hkDjrLYqbZxyQSGKxqtWezU/edit?gid=308545453#gid=308545453'

        publications = conn.read(spreadsheet = publications_url)
        context = conn.read( spreadsheet = context_url)
        mechanisms = conn.read(spreadsheet=mechanisms_url)
        institutions = conn.read(spreadsheet=institutions_url)
        authors = conn.read(spreadsheet=authors_url)
        citations = conn.read(spreadsheet=citations_url)
        umap_df = conn.read(spreadsheet=umap_df_url)
        clusters = conn.read(spreadsheet=clusters_url)

        return publications, context, mechanisms, institutions, authors, citations, umap_df, clusters

    # Load data
    publications, context, mechanisms, institutions, authors, citations, umap_df, clusters = load_data()

    # Set page title
    st.title("Center for Effective Global Action Literature Review")

    # Create tabs for workspace functionalities
    tab1, tab2= st.tabs(["Introduction", "Dashboard" ])

    # Introduction Tab
    with tab1:
       st.markdown("""

            Research on the psychology and economics of poverty has emerged from diverse academic disciplines, each offering distinct perspectives and methodologies. However, these fields have often developed in isolation, leading to fragmented insights and limited interdisciplinary dialogue. This lack of integration makes it challenging to develop a comprehensive understanding of poverty and its multifaceted effects.  

            To address this issue, we have developed an **interactive literature review tool** that enables researchers, policymakers, and practitioners to explore, analyze, and synthesize knowledge from a growing body of research. 

            ###### Key Features of Our Tool:  

            - **Research Exploration:** An interactive platform that allows users to navigate research findings in an intuitive manner.  
            - **Data Visualization:**  Visualizations that help users identify trends, patterns, and gaps in the literature.
            - **Continuous Updates:** Designed to evolve, incorporating new studies and findings as they emerge.  

                    
            ### Our Mission  

            We aim to enhance the understanding of how poverty-related research is structured, supporting evidence-based decision-making and providing a robust foundation for researchers and practitioners working to design effective interventions.  

            This project is being developed by the **Center for Effective Global Action (CEGA) at Berkeley**.  

            - To download the data, please click [here](https://docs.google.com/spreadsheets/d/1npkoU3RmhnKTSKsk_BbXerrSrHtbmyaPXcjO8YSJlUI/edit?gid=1950861456#gid=1950861456).
            - If you are a researcher or project manager interested in adapting this tool for your field, visit the documentation tab to learn about the technology behind it and how to implement it in your research domain.
                   
            """)

    # Understanding the Field Tab
    with tab2:

        # Introduction
        st.markdown("""
            In this research landscape analysis, we offer a guided exploration of poverty research through an interactive dashboard. 
            Our  dataset encompasses academic scholarship from diverse institutional sources across multiple geographic regions, providing a multifaceted lens into contemporary poverty studies.
                        
            Our goal is to map the intellectual terrain of poverty research in a manner that can be explored. 
            Throuhg this exploration tool, we hop... illuminates significant scholarly contributions, trace emerging research trends, and reveals the structure of academic dialogue surrounding poverty alleviation and social development.
            
            This dashboard is as an exploration tool, enabling researchers, policymakers, and stakeholders to navigate complex scholarly insights. 
            Through intuitive interfaces and sophisticated data visualization, users can drill down into research patterns, comparative analyses, and thematic connections that define the current understanding of poverty as a multidimensional global challenge.
            """)

        
        st.markdown("##### Data at a Glance")

        # Initialize session state variables if they don't exist
        if 'selected_country' not in st.session_state:
            st.session_state.selected_country = 'All'
        if 'selected_institution' not in st.session_state:
            st.session_state.selected_institution = 'All'

        # Extract unique countries from data (removing None values)
        country_filter = institutions.copy()
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
                working_df = institutions.copy()
                
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
            filtered_publications = publications[publications['doi'].isin(working_df['doi'].tolist())]
            
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
                with st.expander("Countries Represented", expanded=True):
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

                st.markdown("###### Top Research Institutions")
                institution_figure = bar.create(top_institutions, x_column='institution', y_column='count', title=None, coord_flip=True, height= 345)
                st.plotly_chart(institution_figure, use_container_width=True)

                

        st.markdown("#### Poverty Context and Psychological Mechanisms")
        col1, col2 = st.columns([1  , 1.4])

        with col1:  
 
            with st.expander("Filter by Context and Mechanism", expanded=True):
                filter_dois = working_df['doi'].tolist()


                # Filtering Section
                filter_dois = working_df['doi'].tolist()
                context = context[context['doi'].isin(filter_dois)]
                context_filter = context.copy()
                context_filter = context_filter['context'].explode().unique()
                context_list = [str(context) for context in context_filter if context_filter is not None]
                # st.write(context_list)


                mechanisms = mechanisms[mechanisms['doi'].isin(filter_dois)]
                mechanisms_filter = mechanisms.copy()
                mechanisms_filter = mechanisms_filter['mechanisms'].explode().unique()
                mechanisms_list = [str(mechanisms) for mechanisms in mechanisms_filter if mechanisms_filter is not None]
                # st.write(mechanisms_list)

                col3, col4 = st.columns(2)
                with col3:
                    selected_context = st.selectbox("Filter by Context", ['All'] + sorted(context_list))
                    if selected_context != 'All':
                        context = context[context['context'] == selected_context]
                    elif selected_context == 'All':
                        context = context.copy()

                with col4:
                    selected_mechanism = st.selectbox("Filter by Mechanism", ['All'] + sorted(mechanisms_list))
                    if selected_mechanism != 'All':
                        mechanisms = mechanisms[mechanisms['mechanisms'] == selected_mechanism]
                    elif selected_mechanism == 'All':
                        mechanisms = mechanisms.copy()

                working_context = working_df.merge(context, on='doi', how='left')
                # Then join mechanisms
                context_mechanism = working_context.merge(mechanisms, on='doi', how='left')

                if selected_context != 'All':
                    context_mechanism = context_mechanism[context_mechanism['context'] == selected_context]

                if selected_mechanism != 'All':
                    context_mechanism = context_mechanism[context_mechanism['mechanisms'] == selected_mechanism]

                context_sum = context_mechanism.groupby('context').size().reset_index(name='count')
                mechanisms_sum = context_mechanism.groupby('mechanisms').size().reset_index(name='count')
 
            st.markdown("""

                    Poverty Context includes factors like low resource levels (absolute, perceived, or relative), resource volatility (unpredictable shocks), the physical environment (violence, noise, neighborhood quality), and the social environment (stigma, discrimination, cultural norms).

                    Psychological Mechanisms are mental processes shaped by poverty. Affective factors (anxiety, depression, stress) influence emotions. Beliefs about self-worth and aspirations affect motivation, while cognitive functions (memory, executive control) are hindered by poverty. Preferences, such as time and risk preferences, are shaped by immediate needs.

                    """)
            
        with col2:
            

            link_counts = context_mechanism.groupby(['context', 'mechanisms']).size().reset_index(name='Count')

            # Create node labels
            unique_poverty_contexts = link_counts['context'].unique().tolist()
            unique_mechanisms = link_counts['mechanisms'].unique().tolist()
            all_labels = unique_poverty_contexts + unique_mechanisms

            # Assign indices for Sankey nodes
            label_to_index = {label: i for i, label in enumerate(all_labels)}

            # Prepare data for Sankey diagram
            source_indices = link_counts['context'].map(label_to_index).tolist()
            target_indices = link_counts['mechanisms'].map(label_to_index).tolist()
            values = link_counts['Count'].tolist()

            # Create Sankey Diagram
            fig = go.Figure(go.Sankey(
                node=dict(
                    pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=all_labels
                    # color="blue"  # Sets node fill color
                ),
                link=dict(
                    source=source_indices,
                    target=target_indices,
                    value=values,
                ),
            ))

            # Update text size and color for labels
            fig.update_layout(
                font=dict(size=14, color="black"),  # Changes font size and color for all text
                width=1000,
                height=450
            )

            # Show the figure
            st.plotly_chart(fig, use_container_width=False) 

        # st.write("Research Landscape")
        st.markdown("#### Research Landscape")
        
        filter_dois = context_mechanism['doi'].tolist()
        plot_df = publications[publications['doi'].isin(filter_dois)]


        # st.markdown("###### Influential Papers")
        
        # col1, col2 = st.columns([1, 1])
        # influential_papers = plot_df.sort_values('cited_by_count', ascending=False).head(5)  

        # with col1:

        # with st.expander("", expanded=True):
        #     st.write(influential_papers[['doi','title']].reset_index(drop=True))

        col1, col2 = st.columns([2, 1])

        with col1:
            # with st.expander(" ", expanded=True):


            # Normalize size_mean to range [0, 1]
            clusters['size_mean'] = (clusters['cluster'] - clusters['cluster'].min()) / (clusters['cluster'].max() - clusters['cluster'].min())

            # Scale to desired range, e.g., [10, 50]
            min_size = 100
            max_size = 200
            clusters['size_mean'] = clusters['size_mean'] * (max_size - min_size) + min_size

            clusters.rename(columns={'UMAP1': 'umap_1_mean', 'UMAP2': 'umap_2_mean'}, inplace=True)
            # Create animated figure
            umap_titles = umap_df.merge(plot_df, on='doi', how='inner')
           
            # fig = heat_map.create_cumulative_visualization(plot_df = umap_titles, topic_clusters=clusters)

            # Display the figure
            # st.plotly_chart(fig, use_container_width=True)

            # write palceholder text, until code is ready
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")

            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("  ")
            st.markdown("### Placeholder, visual is currently being refactored")

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
                selected_doi = publications[publications['title'] == selected_paper]['doi'].values[0]

                # create vertical table of paper, abstract, mechanism and context 
                st.markdown("###### Paper Details")
                # Display the selected paper's details

                selected_paper_details = publications[publications['title'] == selected_paper].iloc[0]
                st.markdown(f"**Title:** {selected_paper_details['title']}")
                st.markdown(f"**Context:** {context[context['doi'] == selected_doi]['context'].values}")
                st.markdown(f"**Mechanism:** {mechanisms[mechanisms['doi'] == selected_doi]['mechanisms'].values}")
                selected_authors = authors[authors['doi'] == selected_doi]['authors'].to_list()
                st.markdown(f"**Authors:** {', '.join(author for author in selected_authors)}")
                st.markdown(f"**Abstract:** {selected_paper_details['abstract']}")
              





        

            
            

            


if __name__ == "__main__":
    main()