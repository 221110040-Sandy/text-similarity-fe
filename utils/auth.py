import streamlit as st
import hashlib
from pathlib import Path

ADMIN_CREDENTIALS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest()
}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username: str, password: str) -> bool:
    if username in ADMIN_CREDENTIALS:
        return ADMIN_CREDENTIALS[username] == hash_password(password)
    return False

def initialize_auth_state():
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None

def login(username: str, password: str) -> bool:
    if check_credentials(username, password):
        st.session_state.is_logged_in = True
        st.session_state.username = username
        return True
    else:
        return False

def logout():
    st.session_state.is_logged_in = False
    st.session_state.username = None
    # Clear query params
    st.query_params.clear()

def is_logged_in() -> bool:
    return st.session_state.get('is_logged_in', False)

def get_current_user() -> str:
    return st.session_state.get('username', None)

def require_auth():
    # Check query params for auto-login token
    query_params = st.query_params
    if "auth_token" in query_params and "user" in query_params:
        token = query_params.get("auth_token")
        user = query_params.get("user")
        expected_token = hashlib.sha256(f"{user}_admin_secret".encode()).hexdigest()
        if token == expected_token and user in ADMIN_CREDENTIALS:
            st.session_state.is_logged_in = True
            st.session_state.username = user
    
    if not is_logged_in():
        st.warning("Please login first to access this page.")
        show_login_form()
        st.stop()

def show_login_form():
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin: 2rem 0;'>
        <h2 style='color: white; text-align: center; margin-bottom: 1rem;'>Login Admin Panel</h2>
        <p style='color: rgba(255,255,255,0.9); text-align: center;'>
            Please login to access the admin dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Back to Home"):
        st.switch_page("app_frontend.py")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            submit = st.form_submit_button("Login", width='stretch')
            
            if submit:
                if username and password:
                    if login(username, password):
                        # Set query params untuk persist session
                        auth_token = hashlib.sha256(f"{username}_admin_secret".encode()).hexdigest()
                        st.query_params["auth_token"] = auth_token
                        st.query_params["user"] = username
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
                else:
                    st.warning("Please enter username and password!")

