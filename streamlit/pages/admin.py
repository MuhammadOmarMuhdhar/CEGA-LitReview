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
from data import googleSheets, main
from datetime import datetime
from dotenv import load_dotenv
from streamlit_gsheets import GSheetsConnection
import json




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

# Initialize the data extractor
data_extractor = main.Extract(
    api_key = api_key,
    limit = 1
)

# Connect to Google Sheets
google_sheets = googleSheets.API(
    credentials_json=credenetials)

def load_data():

        conn = st.connection("gsheets", type=GSheetsConnection)
        papers_url = 'https://docs.google.com/spreadsheets/d/1nrZC6zJ50DouMCHIWZOl-tQ4BAZfEojJymtzUh26nP0/edit?usp=sharing'
        topics_url = 'https://docs.google.com/spreadsheets/d/1cspghq8R0Xlf2jk0TacGv8bC6C4EGIUnGuUQJWYASpk/edit?usp=sharing'

        papers_df = conn.read(spreadsheet = papers_url)
        topics_df = conn.read(spreadsheet= topics_url)

        return papers_df, topics_df

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
                if username == os.getenv("USERNAME") and password == os.getenv("PASSWORD"):
                    st.session_state['authenticated'] = True
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

            st.markdown("  ")
            st.markdown("  ")
    
    st.stop()
           

# Main content
tab1, tab2, tab3 = st.tabs(["Update Database", "View Data", "Logs"])

with tab1:
    st.header("Update Database")
    
    col1, col2 = st.columns(2)
    
    with col1:

        start_date = st.date_input("Start Date", datetime.today())
        end_date = st.date_input("End Date", datetime.today())

        if st.button("Fetch Data From API", type="primary"):
            with st.spinner("Fetching data..."):
                formatted_start_date = start_date.strftime("%Y-%m-%d")
                formatted_end_date = end_date.strftime("%Y-%m-%d")
                
                density, clusters, papers = data_extractor.run(
                    start_date=formatted_start_date, end_date=formatted_end_date
                )
            
            papers_df = pd.DataFrame(papers)
            df_x = pd.DataFrame(density['x'])
            df_y = pd.DataFrame(density['y'])
            df_density = pd.DataFrame({
                'x_flat': density['x_flat'],
                'y_flat': density['y_flat'],
                'density': density['density_flat']
            })
            clusters_df = pd.DataFrame(clusters)

            # Save data to Google Sheets
            google_sheets.append(
                df=papers_df,
                spreadsheet_id=papers_spreadsheet_id,
                sheet_name='Sheet1',
                include_headers=True
            )

            google_sheets.replace(
                df=df_x,
                spreadsheet_id=density,
                sheet_name='Sheet1',
                include_headers=True
            )

            google_sheets.replace(
                df=df_y,
                spreadsheet_id=density,
                sheet_name='Sheet1',
                include_headers=True
            )

            google_sheets.replace(
                df=df_density,
                spreadsheet_id=density,
                sheet_name='Sheet1',
                include_headers=True
            )

            google_sheets.replace(
                df=clusters_df,
                spreadsheet_id=topics_spreadsheet_id,
                sheet_name='Sheet1',
                include_headers=True
            )
            st.success("Data fetched successfully!")
           
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