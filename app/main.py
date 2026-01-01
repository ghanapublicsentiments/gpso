"""Main Streamlit application entry point for Ghana Public Sentiments Observatory."""

import os
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from config import MODEL_PROVIDER_MAP


# Load available secrets from st.secrets into os.environ
def load_secrets_to_env():
    """Load Streamlit secrets into environment variables."""
    if hasattr(st, 'secrets') and st.secrets:
        for provider in ["OPENAI", "ANTHROPIC", "GOOGLE", "GROK", "NVIDIA"]:
            key_name = f"{provider}_API_KEY"
            if key_name in st.secrets:
                os.environ[key_name] = st.secrets[key_name]
        
        if "BIGQUERY_DATASET" in st.secrets:
            os.environ["BIGQUERY_DATASET"] = st.secrets["BIGQUERY_DATASET"]
        if "BIGQUERY_IS_PROD" in st.secrets:
            os.environ["BIGQUERY_IS_PROD"] = st.secrets["BIGQUERY_IS_PROD"]
        
        # Store GCP credentials in session state
        if "gcp_service_account" in st.secrets:
            if "gcp_credentials" not in st.session_state:
                st.session_state["gcp_credentials"] = dict(st.secrets["gcp_service_account"])
        
        for key, value in st.secrets.items():
            if key not in os.environ and isinstance(value, (str, int, float, bool)):
                os.environ[key] = str(value)


# Initialize secrets before importing other modules that might use config
try:
    load_secrets_to_env()
except StreamlitSecretNotFoundError: # No secrets file available
    pass

if "cache_ttl" not in st.session_state:
    st.session_state.cache_ttl = {
        "sentiment_data": 300,
        "playground_data": 600,
        "default": 300
    }


def create_model_selector(key: str, help_text: str | None = None) -> str:
    """Create a model selector dropdown for cloud models.
    
    Args:
        key: Unique key for the selectbox widget.
        help_text: Custom help text for the selector.
    
    Returns:
        str: The selected model name.
    """
    available_models = list(MODEL_PROVIDER_MAP.keys())
    
    if help_text is None:
        help_text = "Select a model for inference"
    
    selected_model = st.selectbox(
        "Select Model:",
        options=available_models,
        index=0,
        help=help_text,
        key=key,
        label_visibility="collapsed"
    )
    
    return selected_model

# Define pages
home_page = st.Page("pages/home.py", title="Home", icon=":material/home:")
chat_page = st.Page("pages/chat.py", title="Chat", icon=":material/chat:")
playground_page = st.Page("pages/playground.py", title="Playground", icon=":material/experiment:")
methodology_page = st.Page("pages/methodology.py", title="Methodology", icon=":material/description:")
faqs_page = st.Page("pages/faqs.py", title="FAQs", icon=":material/help_outline:")
tos_page = st.Page("pages/terms_of_service.py", title="Terms of Service", icon=":material/gavel:")

# Create navigation
page = st.navigation([home_page, chat_page, playground_page, methodology_page, faqs_page, tos_page])

# Set page config
st.set_page_config(page_title="Ghana Public Sentiments Observatory", page_icon="💬", layout="wide")

# Navigation bar with custom header
st.html(
    """
    <style>
    /* Remove all rounded corners */
    .stApp, .main, .block-container, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    .stMarkdown, .element-container, div[data-testid="column"] > div,
    section[data-testid="stSidebar"], iframe {
        border-radius: 0 !important;
    }
    
    .stAppHeader {
        background-color: #fdfdf8;
        border-bottom: 1px solid #e0e0d8;
        border-radius: 0 !important;
    }
    .stAppHeader::before {
        content: "Ghana Public Sentiments Observatory";
        color: #3d3a2a;
        font-size: 20px;
        font-weight: 800;
        font-family: 'SpaceGrotesk', sans-serif;
        position: absolute;
        left: 5rem;
        top: 50%;
        transform: translateY(-50%);
        white-space: nowrap;
    }
    
    /* Responsive header for tablets and smaller screens */
    @media (max-width: 768px) {
        .stAppHeader::before {
            font-size: 16px;
            left: 3rem;
        }
    }
    
    /* Responsive header for mobile devices */
    @media (max-width: 480px) {
        .stAppHeader::before {
            content: "GPSO";
            font-size: 18px;
            left: 2rem;
        }
    }
    footer {
        visibility: hidden;
    }
    footer::after {
        content: '';
        visibility: hidden;
    }
    </style>
    """
)

# Run the selected page
page.run()

# Sidebar content
with st.sidebar.container(height=310):
    if page.title == "Home":
        st.page_link("pages/home.py", label="Home", icon=":material/home:")
        st.write("Welcome to the Ghana Public Sentiments Observatory!")
        st.write(
            """
            We monitor discussions around news items across the country 
            and bring you insights into how the public feels about them.
            """.strip()
        )

    elif page.title == "Chat":
        st.page_link("pages/chat.py", label="Chat", icon=":material/chat:")
        st.write("Chat with an AI assistant for custom analysis of public sentiments.")

        st.session_state.chat_model = create_model_selector(key="chat_model_selector")
        
        selected_model = st.session_state.chat_model
        provider = MODEL_PROVIDER_MAP.get(selected_model)
        
        if provider:
            st.text_input(
                f"{provider} API Key:",
                type="password",
                key="chat_api_key",
                placeholder=f"Enter your {provider} API key",
                help=f"Required for {provider} models. Your key is stored on your device and only for this session."
            )

    elif page.title == "Playground":
        st.page_link("pages/playground.py", label="Playground", icon=":material/experiment:")
        st.write("Simulate public sentiments to news")

        st.session_state.playground_model = create_model_selector(key="playground_model_selector")
        
        selected_model = st.session_state.playground_model
        provider = MODEL_PROVIDER_MAP.get(selected_model)
        
        if provider:
            st.text_input(
                f"{provider} API Key:",
                type="password",
                key="playground_api_key",
                placeholder=f"Enter your {provider} API key",
                help=f"Required for {provider} models. Your key is stored on your device and only for this session."
            )

    elif page.title == "Methodology":
        st.page_link("pages/methodology.py", label="Methodology", icon=":material/description:")
        st.write("Understand how we collect data, analyze sentiments, and generate insights on public opinion in Ghana.")

    elif page.title == "FAQs":
        st.page_link("pages/faqs.py", label="FAQs", icon=":material/help_outline:")
        st.write("Find answers to some frequently asked questions about our platform.")

    elif page.title == "Terms of Service":
        st.page_link("pages/terms_of_service.py", label="Terms of Service", icon=":material/gavel:")
        st.write("Review the terms and conditions for using the Ghana Public Sentiments Observatory platform.")

# Refresh button
if st.sidebar.button("Refresh data", icon=":material/autorenew:", type="secondary", use_container_width=True):
    st.rerun()

st.sidebar.divider()

st.sidebar.caption(
    """
    An open-source initiative by Drumline Strategies. You can contribute [here](https://github.com/ghanapublicsentiments/gpso).
    """.strip()
)

