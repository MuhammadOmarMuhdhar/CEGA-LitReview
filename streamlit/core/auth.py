import streamlit as st
import logging

logger = logging.getLogger(__name__)


def check_authentication(get_env_var) -> bool:
    """
    Check if user is authenticated. Returns True if authenticated, False otherwise.
    If not authenticated, renders the login form and stops execution.
    """
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        _render_login_form(get_env_var)
        return False

    logger.info("USER_ACTION [authenticated_access]: session_authenticated=True")
    return True


def _render_login_form(get_env_var):
    """Render the login form UI"""
    logger.info("USER_ACTION [login_attempt]: showing_login_form")

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
                _handle_login(username, password, get_env_var)

            st.markdown("  ")
            st.markdown("  ")

    st.stop()


def _handle_login(username: str, password: str, get_env_var):
    """Handle login form submission"""
    logger.info(f"USER_ACTION [login_submit]: username={username[:3] if username else ''}***")

    expected_username = get_env_var("USERNAME")
    expected_password = get_env_var("PASSWORD")

    if username == expected_username and password == expected_password:
        st.session_state['authenticated'] = True
        logger.info(f"USER_ACTION [login_success]: user={username}")
        st.success("Logged in successfully")
        st.rerun()
    else:
        logger.info(f"USER_ACTION [login_failed]: user={username}")
        st.error("Invalid username or password")
