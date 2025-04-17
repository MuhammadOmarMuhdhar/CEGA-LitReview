import streamlit as st
import streamlit as st
import pandas as pd
import requests
import sqlite3
from datetime import datetime
import json

# Set page config
st.set_page_config(page_title="Database Updater", layout="wide")

# Database connection function
def connect_to_db():
    conn = sqlite3.connect('your_database.db')
    return conn

# API connection function
def fetch_api_data(api_url, params=None, headers=None):
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None

# Function to update database with API data
def update_database(data, table_name):
    if not data:
        return False, "No data to update"
    
    try:
        conn = connect_to_db()
        df = pd.DataFrame(data)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True, f"Successfully updated {len(df)} records in {table_name}"
    except Exception as e:
        return False, f"Database error: {str(e)}"

# Streamlit UI
st.title("API to Database Updater")



# Main content
tab1, tab2, tab3 = st.tabs(["Update Database", "View Data", "Logs"])

with tab1:
    st.header("Update Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Fetch Data From API", type="primary"):
            st.info("Curretny not implemented")
            # with st.spinner("Fetching data from API..."):
            #     api_data = fetch_api_data(api_url, params, headers, )
                
            #     if api_data:
            #         st.session_state['api_data'] = api_data
            #         st.success(f"Successfully fetched data from API. Found {len(api_data if isinstance(api_data, list) else api_data.get('results', []))} records.")
                    
            #         # Display preview
            #         st.subheader("Data Preview")
            #         if isinstance(api_data, list):
            #             st.dataframe(pd.DataFrame(api_data).head())
            #         else:
            #             # Handle nested API responses
            #             if 'results' in api_data:
            #                 st.dataframe(pd.DataFrame(api_data['results']).head())
            #             else:
            #                 st.json(api_data)
    
    with col2:
        if st.button("Update Database", type="primary", disabled='api_data' not in st.session_state):
            if 'api_data' in st.session_state:
                with st.spinner("Updating database..."):
                    data_to_update = st.session_state['api_data']
                    if not isinstance(data_to_update, list) and 'results' in data_to_update:
                        data_to_update = data_to_update['results']
                        
                    success, message = update_database(data_to_update, table_name)
                    
                    if success:
                        st.success(message)
                        
                        # Add to logs
                        if 'logs' not in st.session_state:
                            st.session_state['logs'] = []
                        
                        st.session_state['logs'].append({
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'action': "Database Update",
                            'status': "Success",
                            'message': message
                        })
                    else:
                        st.error(message)
            else:
                st.warning("Please fetch data from API first")

with tab2:
    st.header("View Current Database Data")
    
    if st.button("Load Database Data"):
        st.info("Currently not implemented")
        # try:
        #     conn = connect_to_db()
        #     query = f"SELECT * FROM {table_name}"
        #     df = pd.read_sql_query(query, conn)
        #     conn.close()
            
        #     if not df.empty:
        #         st.dataframe(df)
                
        #         # Download option
        #         csv = df.to_csv(index=False)
        #         st.download_button(
        #             label="Download as CSV",
        #             data=csv,
        #             file_name=f"{table_name}_export.csv",
        #             mime="text/csv"
        #         )
        #     else:
        #         st.info(f"No data found in table '{table_name}'")
        # except Exception as e:
        #     st.error(f"Error loading data: {str(e)}")

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