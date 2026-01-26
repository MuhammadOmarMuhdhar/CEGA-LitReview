import streamlit as st
import os
import sys
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from data.bigQuery import Client


def load_environment_variables():
    """Load environment variables with fallback to Streamlit secrets"""
    load_dotenv()

    def get_env_var(key, default=None):
        value = os.getenv(key)
        if value is not None:
            return value

        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

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

    api_key = get_env_var("GEMINI_API_KEY")

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


@st.cache_resource(show_spinner="Connecting to Database...")
def get_bigquery_client():
    try:
        _, credentials, _, _ = get_configuration()
        client = Client(credentials, 'literature-452020')
        if not client._is_client_healthy():
            logger.error("Failed to establish healthy database connection")
            st.error("Failed to establish healthy database connection")
            st.stop()
        return client
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

def get_healthy_bigquery_client():
    """Get a healthy BigQuery client, handling refresh automatically"""
    client = get_bigquery_client()
    client.get_healthy_client()
    return client
