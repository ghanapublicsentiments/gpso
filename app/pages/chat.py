"""Chat interface page for interactive sentiment data exploration."""

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from chat_orchestrator import execute_tool_calls
from database.bigquery_manager import get_bigquery_manager
from input_sanitizer import sanitize_user_prompt


# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

selected_model = st.session_state.get("chat_model")
chat_api_key = st.session_state.get("chat_api_key")

avatars = {
    "user": "👩🏾‍🦱",
    "assistant": "💻",
    "tool": "🔧",
}

# ============================================================================
# DATA INITIALIZATION
# ============================================================================

@st.cache_data(ttl=st.session_state.get("cache_ttl", {}).get("sentiment_data", 300), show_spinner=False)
def load_sentiment_data() -> pd.DataFrame:
    """
    Load sentiment data from BigQuery.
    
    Returns:
        DataFrame containing entity summaries.
    """
    manager = get_bigquery_manager()
    summary_rows = manager.get_all_entity_summaries()
    df_summaries = pd.DataFrame(summary_rows)
    return df_summaries


def init_sentiment_data() -> None:
    """Initialize sentiment data and related structures in session state."""
    if "sentiment_data_loaded" not in st.session_state:
        df_summaries = load_sentiment_data()
        print(f"Loaded {len(df_summaries)} entity summaries")
        
        st.session_state["df_entity_summaries"] = df_summaries
        st.session_state["df_entity_summaries_id"] = "df_entity_summaries"
        
        # Store df summary for system instructions
        st.session_state["df_entity_summaries_describe"] = df_summaries.describe(include='all').to_json()
        
        # Dictionary to store dynamically created dataframes
        st.session_state["created_dataframes"] = {}
        
        # Dictionary to store created plotly figures
        st.session_state["created_figures"] = {}
        st.session_state["sentiment_data_loaded"] = True

# Initialize data
init_sentiment_data()


# ============================================================================
# SYSTEM INSTRUCTIONS
# ============================================================================

def get_system_message() -> str:
    """Generate system instruction with data descriptions."""
    entity_describe = st.session_state.get("df_entity_summaries_describe", "{}")
    
    df_summaries = st.session_state.get("df_entity_summaries", pd.DataFrame())
    
    prompt = f"""You are an AI assistant specialized in analyzing public sentiment data and providing custom insights.

        You have access to one main dataset:

        **Entity Summaries** (df_entity_summaries):
        - Contains aggregated sentiment summaries for each entity (key players, key issues, major topics)
        - Total records: {len(df_summaries)}
        - Columns: {', '.join(df_summaries.columns.tolist()) if not df_summaries.empty else 'N/A'}
        - Statistical summary: {entity_describe}

        Your role is to:
        - Answer questions about sentiment trends and patterns
        - Filter and aggregate data efficiently to provide insights
        - Create visualizations to help users understand the data
        - Provide data-driven analysis and recommendations
        - Analyze custom entities (not in the standard dataset) using the analyze_custom_entity_sentiment tool

        When working with data:
        - Use get_dataframe to filter/aggregate existing entity summary data efficiently
        - Use create_dataframe to create custom datasets for analysis
        - Use create_plotly_figure to visualize data with appropriate chart types
        - Use analyze_custom_entity_sentiment when users ask about entities not in df_entity_summaries
        - Always minimize data returned - only include what's necessary to answer the query
        - When showing data to users, provide context and interpretation
        
        Important: Individual comment-level sentiments are not directly accessible to protect user privacy.
        Only aggregated entity-level summaries are available. For custom entity analysis, use the 
        analyze_custom_entity_sentiment tool which applies the full pipeline and returns aggregates.
        
        For methodology questions, direct users to the Methodology page in the app navigation or
        the GitHub repository for detailed technical documentation.
    """.strip()
    return prompt


# ============================================================================
# CHAT UI AND MESSAGE HANDLING
# ============================================================================


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    
    # Skip tool and system messages in history display
    if role in ["tool", "system"]:
        continue
    
    # Skip assistant messages with tool calls but no content (intermediate processing steps)
    if role == "assistant" and "tool_calls" in message and not message.get("content"):
        continue
    
    # Get appropriate avatar
    avatar = avatars.get(role, "💬")

    with st.chat_message(role, avatar=avatar), st.empty():
        # Display message content
        if message.get("content"):
            st.markdown(message["content"])
        
        # Check if message has metadata for figures or dataframes to display
        if "figure_ids" in message:
            for fig_id in message["figure_ids"]:
                fig = st.session_state.get("created_figures", {}).get(fig_id)
                if fig:
                    st.plotly_chart(fig, width='stretch', theme=None)
        
        if "dataframe_ids" in message:
            for df_id in message["dataframe_ids"]:
                if df_id in ["df_entity_summaries"]:
                    df = st.session_state.get(df_id, pd.DataFrame())
                else:
                    df = st.session_state.get("created_dataframes", {}).get(df_id, pd.DataFrame())
                if not df.empty:
                    st.dataframe(df, width='stretch')

if prompt := st.chat_input("What would you like to know?"):
    # Sanitize user input before processing
    try:
        sanitized_prompt = sanitize_user_prompt(prompt, max_length=4000)
    except ValueError as e:
        st.error(f"Invalid input: {str(e)}")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
    with st.chat_message("user", avatar=avatars["user"]), st.empty():
        st.markdown(sanitized_prompt)
    
    # Call the orchestrator with current messages
    with st.chat_message("assistant", avatar=avatars["assistant"]), st.empty():
        try:
            with st.spinner("Thinking..."):
                response, updated_messages = execute_tool_calls(
                    st.session_state.messages,
                    selected_model,
                    get_system_message,
                    api_key=chat_api_key
                )
            st.markdown(response)
            
            # Display any figures or dataframes that were created
            last_message = updated_messages[-1] if updated_messages else {}
            if "figure_ids" in last_message:
                for fig_id in last_message["figure_ids"]:
                    fig = st.session_state.get("created_figures", {}).get(fig_id)
                    if fig:
                        st.plotly_chart(fig, width='stretch', theme=None)
            
            if "dataframe_ids" in last_message:
                for df_id in last_message["dataframe_ids"]:
                    if df_id in ["df_entity_summaries"]:
                        df = st.session_state.get(df_id, pd.DataFrame())
                    else:
                        df = st.session_state.get("created_dataframes", {}).get(df_id, pd.DataFrame())
                    if not df.empty:
                        st.dataframe(df, width='stretch')
            
            # Update session state with all new messages (including tool interactions)
            st.session_state.messages = updated_messages
            
        except ValueError as e:
            # Handle configuration errors (API keys, invalid models, etc.)
            error_str = str(e)
            
            if "API key" in error_str or "not found" in error_str:
                # API key configuration error
                error_message = """
                    I'm sorry, but I can't connect right now because the API key for the selected model isn't configured. 

                    Please paste your API key in the textbox in the sidebar's Playground section.
                """.strip()
            elif "Model" in error_str and "not found" in error_str:
                # Invalid model error
                error_message = """
                    Hmm, the selected model doesn't seem to be available. 
                    Try selecting a different model from the playground, or double-check that the model name is correct.
                """.strip()
            else:
                # Other ValueError
                error_message = """
                    I ran into a configuration issue. Try refreshing the page, selecting a different model, or checking your settings.
                """.strip()
            
            st.error(error_message)
            
            # Add error to message history so user can see it
            error_msg = {
                "role": "assistant",
                "content": error_message
            }
            st.session_state.messages.append(error_msg)
            
        except ConnectionError as e:
            # Handle network/connection errors
            error_message = """
                I'm having trouble connecting to the AI service. Please check your internet connection and try again in a few moments. If you're behind a firewall, that might be blocking the connection.
            """.strip()
            st.error(error_message)
            
            error_msg = {
                "role": "assistant",
                "content": error_message
            }
            st.session_state.messages.append(error_msg)
            
        except Exception as e:
            # Catch-all for any unexpected errors
            error_str = str(e).lower()
            
            # Try to provide context-specific guidance
            if "timeout" in error_str:
                error_message = """
                    The AI service took too long to respond. Wait a moment and try again, or try asking a simpler question.
                """.strip()
            elif "rate limit" in error_str:
                error_message = """
                    Looks like we've hit the rate limit from too many requests. Wait a few moments before trying again. 
                    If this happens often, you might want to consider upgrading your API plan.
                """.strip()
            elif "authentication" in error_str or "unauthorized" in error_str:
                error_message = """
                    Your API key seems to be invalid or expired. Please check that it's correct and hasn't expired.
                """.strip()
            else:
                error_message = """
                    Something unexpected went wrong while processing your request. Try refreshing the page and asking again. 
                    If the problem keeps happening, please contact support.
                """.strip()
            
            st.error(error_message)
            
            # Add error to message history so user can see it
            error_msg = {
                "role": "assistant",
                "content": error_message
            }
            st.session_state.messages.append(error_msg)
