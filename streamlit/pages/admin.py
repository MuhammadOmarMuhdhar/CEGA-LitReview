import streamlit as st



# enter username password otherwise page isnt authorized
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:

    col1, col2, col3 = st.columns([1, 4, 1])

    with col2:
        with st.expander("Login", expanded=True):

            st.markdown("  ")
            st.markdown("  ")
            

            st.write("This page is protected. Please enter credentials to access.")

            st.markdown("  ")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            st.markdown("  ")
            
            if st.button("Login"):
                if username == "PLACEHODLER" and password =="PLACEHODLER":
                    st.session_state['authenticated'] = True
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            st.markdown("  ")
            st.markdown("  ")
    
    st.stop()


import os
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import streamlit as st
import streamlit as st
import pandas as pd
from data.ETL import main
from datetime import datetime
from dotenv import load_dotenv
from streamlit_gsheets import GSheetsConnection





def load_data():

        conn = st.connection("gsheets", type=GSheetsConnection)
        papers_url = 'https://docs.google.com/spreadsheets/d/1nrZC6zJ50DouMCHIWZOl-tQ4BAZfEojJymtzUh26nP0/edit?usp=sharing'
        topics_url = 'https://docs.google.com/spreadsheets/d/1cspghq8R0Xlf2jk0TacGv8bC6C4EGIUnGuUQJWYASpk/edit?usp=sharing'

        papers_df = conn.read(spreadsheet = papers_url)
        topics_df = conn.read(spreadsheet= topics_url)

        return papers_df, topics_df
# Set page config
st.set_page_config(page_title="Database Updater", layout="wide")
# Streamlit UI

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

# Load Google Sheets credentials from environment variables
type= os.getenv("type")
project_id = os.getenv("project_id")
private_key_id = os.getenv("private_key_id") 
private_key = os.getenv("private_key").replace("\\n", "\n")  
client_email = os.getenv("client_email")
client_id = os.getenv("client_id")
auth_uri = os.getenv("auth_uri")
token_uri = os.getenv("token_uri")
auth_provider_x509_cert_url = os.getenv("auth_provider_x509_cert_url")
client_x509_cert_url= os.getenv("client_x509_cert_url")
universe_domain= os.getenv("universe_domain")
credenetials = {
    "type": type,
    "project_id": project_id,
    "private_key_id": private_key_id,
    "private_key": private_key,
    "client_email": client_email,
    "client_id": client_id,
    "auth_uri": auth_uri,
    "token_uri": token_uri,
    "auth_provider_x509_cert_url": auth_provider_x509_cert_url,
    "client_x509_cert_url": client_x509_cert_url,
    "universe_domain": universe_domain
}
# Load Google Sheets IDs from environment variables
papers_spreadsheet_id = os.getenv("papers_spreadsheet_id")
density = os.getenv("density")
density_X = os.getenv("X")
density_Y = os.getenv("Y")
topics_spreadsheet_id = os.getenv("topics_spreadsheet_id")
spreadsheet_id =  {
    "papers": papers_spreadsheet_id,
    "density": density,
    "density_X": density_X,
    "density_Y": density_Y,
    "topics": topics_spreadsheet_id
}

# Initialize WTL
data_pipeline = main.ETLPipeline(
    api_key=api_key,
    credentials_json =credenetials,
    spreadsheet_id_json =spreadsheet_id,
    limit=10
)

def load_data():

        conn = st.connection("gsheets", type=GSheetsConnection)
        papers_url = 'https://docs.google.com/spreadsheets/d/1nrZC6zJ50DouMCHIWZOl-tQ4BAZfEojJymtzUh26nP0/edit?usp=sharing'
        topics_url = 'https://docs.google.com/spreadsheets/d/1cspghq8R0Xlf2jk0TacGv8bC6C4EGIUnGuUQJWYASpk/edit?usp=sharing'

        papers_df = conn.read(spreadsheet = papers_url)
        topics_df = conn.read(spreadsheet= topics_url)

        return papers_df, topics_df
        
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
                
                papers = data_pipeline.run(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'))
                st.session_state.papers = papers
                st.session_state.data_fetched = True

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
                        
                        # st.markdown("---")


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
    st.dataframe(papers_df[['doi', 'title', 'authors', 'abstract', 'country', 'institution', 'study_type', 'poverty_context', 'mechanism']], use_container_width=True)

    # button to reload with st.rerun
    if st.button("Reload Data"):
        with st.spinner("Reloading data..."):
            papers_df, topics_df = load_data()
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