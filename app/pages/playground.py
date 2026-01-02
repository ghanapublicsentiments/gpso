"""Playground page for simulating public sentiment on policy announcements."""

import random
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_client
from database.bigquery_manager import BigQueryManager

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


class SimulatedCommentsModel(BaseModel):
    """Model for 10 simulated comments with structured output."""
    
    comment_1: str = Field(description="First simulated comment")
    comment_2: str = Field(description="Second simulated comment")
    comment_3: str = Field(description="Third simulated comment")
    comment_4: str = Field(description="Fourth simulated comment")
    comment_5: str = Field(description="Fifth simulated comment")
    comment_6: str = Field(description="Sixth simulated comment")
    comment_7: str = Field(description="Seventh simulated comment")
    comment_8: str = Field(description="Eighth simulated comment")
    comment_9: str = Field(description="Ninth simulated comment")
    comment_10: str = Field(description="Tenth simulated comment")


def generate_random_avatars(count: int = 10) -> list[str]:
    """
    Generate random human avatars for simulated users.
    
    Args:
        count: Number of avatars to generate.
    
    Returns:
        List of avatar emoji strings.
    """
    avatar_pool = [
        "👨🏿", "👩🏿", "👨🏾", "👩🏾", "🧔🏿", "👱🏿‍♀️", 
        "👨🏿‍🦱", "👩🏿‍🦱", "👨🏾‍🦲", "👩🏾‍🦲", "🧔🏾‍♂️", "👨🏿‍🦰",
        "👩🏾‍🦱", "👨🏾‍🦱", "🧑🏿", "🧑🏾", "👴🏿", "👵🏿",
        "🙍🏿‍♂️", "🙍🏿‍♀️", "🙎🏿‍♂️", "🙎🏿‍♀️", "🙅🏿‍♂️", "🙅🏿‍♀️",
        "🙆🏿‍♂️", "🙆🏿‍♀️", "💁🏿‍♂️", "💁🏿‍♀️", "🙋🏿‍♂️", "🙋🏿‍♀️"
    ]
    
    # Randomly select unique avatars
    selected = random.sample(avatar_pool, min(count, len(avatar_pool)))
    return selected


def embed_text(text: str, model: str = "all-MiniLM-L6-v2") -> list[float] | None:
    """
    Embed text using SentenceTransformer model.
    
    Args:
        text: Text to embed.
        model: SentenceTransformer model name.
    
    Returns:
        List of embedding values, or None if embedding fails.
    """
    try:
        embedder = SentenceTransformer(model)
        embedding = embedder.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        st.error(f"Error embedding text: {str(e)}")
        return None


@st.cache_data(ttl=st.session_state.get("cache_ttl", {}).get("playground_data", 600), show_spinner=False)
def retrieve_similar_content_and_comments(_creds_dict: Optional[dict], query_embedding: list[float], top_k: int = 5, source_filter: str | None = None) -> list[dict]:
    """
    Retrieve most similar content and their comments from database based on cosine similarity.
    
    Args:
        _creds_dict: Optional credentials dictionary (prefixed with _ to exclude from caching).
        query_embedding: Embedding vector of the query text.
        top_k: Number of top similar content items to return.
        source_filter: Optional filter for content type ('youtube' or 'facebook'). None = all sources.
    
    Returns:
        List of dictionaries containing:
            - title: Content title/preview
            - source_name: Channel/page name
            - content_type: 'youtube' or 'facebook'
            - comments: List of comment texts
            - similarity_score: Cosine similarity score
    """
    manager = BigQueryManager(creds_dict=_creds_dict)
    
    # Get all content with comments using unified query
    rows = manager.get_content_with_comments(limit=1000, source_filter=source_filter)
    
    if not rows:
        return []
    
    # Group by content
    content_items = {}
    for row in rows:
        content_id = row['content_id']
        if content_id not in content_items:
            content_items[content_id] = {
                'title': row['title'],
                'source_name': row['source_name'] or 'Unknown Source',
                'content_type': row['content_type'],
                'comments': []
            }
        content_items[content_id]['comments'].append(row['comment_text'])
    
    # Batch embed all content titles
    content_ids = list(content_items.keys())
    titles = [content_items[cid]['title'] for cid in content_ids]
    
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        title_embeddings = embedder.encode(titles, convert_to_numpy=True)
    except Exception as e:
        st.error(f"Error embedding titles: {str(e)}")
        return []
    
    # Calculate similarities for all content items
    similarities = cosine_similarity([query_embedding], title_embeddings)[0]
    
    # Build results with similarity scores
    results = []
    for idx, content_id in enumerate(content_ids):
        item = content_items[content_id]
        results.append({
            'title': item['title'],
            'source_name': item['source_name'],
            'content_type': item['content_type'],
            'comments': item['comments'][:10],  # Limit to 10 comments per content
            'similarity_score': float(similarities[idx])
        })
    
    # Sort by similarity and return top_k
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]


def _display_comment_if_new(
    field_name: str,
    comment_text: str,
    comment_index: int,
    avatars: list[str],
    displayed_comments: set[str]
) -> None:
    """
    Display a comment if it hasn't been shown yet and meets the criteria.
    
    Args:
        field_name: The field name (e.g., 'comment_1').
        comment_text: The comment text to display.
        comment_index: The 1-based index of the comment (1-10).
        avatars: List of avatar emojis.
        displayed_comments: Set of already displayed comment field names (modified in place).
    """
    if comment_text and field_name not in displayed_comments:
        if len(comment_text) > 10:
            avatar = avatars[comment_index - 1]
            with st.chat_message("user", avatar=avatar):
                st.write(comment_text)
            displayed_comments.add(field_name)
            if len(st.session_state.playground_generated_comments) < comment_index:
                st.session_state.playground_generated_comments.append(comment_text)
            else:
                st.session_state.playground_generated_comments[comment_index - 1] = comment_text


def _handle_streaming_response(stream, avatars: list[str]) -> None:
    """
    Handle streaming response from the LLM and display comments as they arrive.
    
    Args:
        stream: The streaming response object from the API.
        avatars: List of avatar emojis to use for each comment.
    """
    # Track which comments have been displayed
    displayed_comments = set()
    
    for event in stream:
        if event.type == "content.delta":
            if event.parsed is not None:
                # Get parsed data as dict
                if hasattr(event.parsed, 'model_dump'):
                    parsed_dict = event.parsed.model_dump()
                elif isinstance(event.parsed, dict):
                    parsed_dict = event.parsed
                else:
                    continue
                
                # Check each comment field and display new ones
                for i in range(1, 11):
                    field_name = f"comment_{i}"
                    if field_name in parsed_dict and parsed_dict[field_name]:
                        _display_comment_if_new(
                            field_name,
                            parsed_dict[field_name],
                            i,
                            avatars,
                            displayed_comments
                        )
        
        elif event.type == "content.done":
            # Final update - ensure all comments are displayed
            final_completion = stream.get_final_completion()
            if final_completion and final_completion.choices:
                final_parsed = final_completion.choices[0].message.parsed
                if final_parsed:
                    if hasattr(final_parsed, 'model_dump'):
                        final_dict = final_parsed.model_dump()
                    elif isinstance(final_parsed, dict):
                        final_dict = final_parsed
                    else:
                        final_dict = {}
                    
                    # Display any remaining comments that weren't shown yet
                    for i in range(1, 11):
                        field_name = f"comment_{i}"
                        if field_name in final_dict:
                            _display_comment_if_new(
                                field_name,
                                final_dict[field_name],
                                i,
                                avatars,
                                displayed_comments
                            )
        
        elif event.type == "error":
            st.error(f"❌ Error in stream: {event.error}")


def generate_simulated_comments(user_input: str, similar_examples: list[dict], model: str = "gpt-5-nano") -> None:
    """
    Generate simulated comments based on user's policy/announcement input and similar examples.
    
    Streams the structured output as it's generated, displaying each comment in a chat message.
    
    Args:
        user_input: The policy announcement or topic text.
        similar_examples: List of similar content items with comments.
        model: LLM model to use for generation.
    """
    # Build examples section
    examples_text = "Here are some examples of real content and their associated comments:\n\n"
    for i, example in enumerate(similar_examples[:10], 1):  # Use top 10 examples
        examples_text += f"EXAMPLE {i}:\n"
        examples_text += f"Content: {example['title']}\n"
        examples_text += f"Source: {example['source_name']}\n"
        examples_text += "Comments:\n"
        for comment in example['comments']:  # Show all comments per example
            examples_text += f"- {comment}\n"
        examples_text += "\n"

    user_prompt = f"""
    You are provided with examples of social media content and their associated comments. Based on the similarity to these examples, generate a set of 10 realistic and culturally relevant comments for the following announcement or topic:
    {user_input}

    Examples:
    {examples_text}

    Important Guidelines:
    - From the examples, reason about the elements that influenced commenter reactions, and use this to influence your comment generation.
    - The comments must pertain to the specific announcement or topic provided.
    - Do not mention or reference the example content or comments in your output.
    - Do not mention any names that are not in the announcement.
    - Do not generate comments based on your own assumptions; rely solely on the provided announcement and examples.
    - Reflect typical Ghanaian commenter reactions, expressions, and slang.
    - Do not include any offensive or inappropriate content.
    - Do not deliberately ensure diversity in perspectives (positive, negative, neutral). Simply reflect how people would naturally respond to this particular announcement, based on the examples provided.

    Generate exactly 10 comments, one for each field in the response format.
    """.strip()
    
    # Get the OpenAI-compatible client with user-provided API key if available
    api_key = st.session_state.get("playground_api_key")
    
    try:
        openai_client = get_client(model, api_key=api_key)
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.info("💡 Please provide an API key in the sidebar or select a local model (🖥️).")
        return None
    
    try:
        # Use avatars from session state
        avatars = st.session_state.playground_avatars
        
        # Create streaming completion with structured output
        with openai_client.beta.chat.completions.stream(
            model=model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            response_format=SimulatedCommentsModel,
        ) as stream:
            _handle_streaming_response(stream, avatars)
        
    except Exception as e:
        st.error(f"❌ Error generating comments: {str(e)}")
        # Check if it's a model support issue
        if "does not support" in str(e).lower() or "structured_outputs" in str(e).lower():
            st.warning("⚠️ The selected model does not support structured outputs. Please use a compatible model like gpt-5-mini, or gemini-2.5-flash.")
        return None


# Initialize session state for persistence
if "playground_news_input" not in st.session_state:
    st.session_state.playground_news_input = ""
if "playground_similar_examples" not in st.session_state:
    st.session_state.playground_similar_examples = []
if "playground_generated_comments" not in st.session_state:
    st.session_state.playground_generated_comments = []
if "playground_avatars" not in st.session_state:
    st.session_state.playground_avatars = []
if "playground_data_source" not in st.session_state:
    st.session_state.playground_data_source = "All Sources"

# Get credentials from session state if available (for Streamlit Cloud)
creds_dict = st.session_state.get("gcp_credentials")

# Main UI
selected_model = st.session_state.get("playground_model")

# Add data source filter dropdown
top_col1, top_col2 = st.columns([3, 1])

with top_col2:
    source_options = ["All Sources", "YouTube Only", "Facebook Only"]
    selected_source = st.selectbox(
        "Data source:",
        options=source_options,
        index=source_options.index(st.session_state.playground_data_source),
        key="playground_source_filter",
        help="Filter training data by source",
        label_visibility="collapsed"
    )
    st.session_state.playground_data_source = selected_source

# Map UI selection to filter parameter
source_param = None
if selected_source == "YouTube Only":
    source_param = "youtube"
elif selected_source == "Facebook Only":
    source_param = "facebook"

news = st.text_area(
    "Enter news item or policy announcement:",
    height=200,
    placeholder="Example: Government announces new education policy to provide free laptops to all senior high school students...",
    value=st.session_state.playground_news_input,
    key="news_input_widget"
)

col1, col2 = st.columns([1, 4])
with col1:
    button = st.button("Simulate Sentiments", type="primary")
with col2:
    if st.button("Clear", type="secondary"):
        st.session_state.playground_news_input = ""
        st.session_state.playground_similar_examples = []
        st.session_state.playground_generated_comments = []
        st.session_state.playground_avatars = []
        st.rerun()

if news and button:
    # Validate input is not empty or just whitespace
    if not news.strip():
        st.warning("⚠️ Please enter some text before running the simulation.")
        st.stop()
    
    # Update session state with current input
    st.session_state.playground_news_input = news
    
    with st.spinner("Analyzing input and finding similar content..."):
        # Step 1: Embed user input
        query_embedding = embed_text(news)
        
        if query_embedding is None:
            st.error("Failed to process input. Please try again.")
            st.stop()
        
        # Step 2: Retrieve similar content and comments
        similar_examples = retrieve_similar_content_and_comments(
            _creds_dict=creds_dict, 
            query_embedding=query_embedding, 
            top_k=5, 
            source_filter=source_param
        )
        
        if not similar_examples:
            st.warning(f"⚠️ No similar content found in database for {selected_source}. Unable to generate simulation.")
            st.info("💡 Try selecting a different data source.")
            st.stop()
        
        # Store similar examples in session state
        st.session_state.playground_similar_examples = similar_examples
    
    # Show similar examples found
    with st.expander("📚 Similar Content Found (used as examples)", expanded=False):
        for i, example in enumerate(similar_examples[:3], 1):
            st.markdown(f"**{i}. {example['title']}**")
            st.caption(f"Source: {example['source_name']}")
            st.caption(f"Similarity: {example['similarity_score']:.2%}")
            st.caption(f"Sample comments: {len(example['comments'])}")
            st.divider()
    
    # Step 3: Generate simulated comments

    # Generate new avatars and store them
    st.session_state.playground_avatars = generate_random_avatars(10)
    
    # Clear previous comments before generating new ones
    st.session_state.playground_generated_comments = []
    
    with st.container(height=400):
        with st.spinner("Generating simulated comments..."):
            generate_simulated_comments(news, similar_examples, model=selected_model)

# Display previously generated comments if they exist and no new generation is happening
elif st.session_state.playground_generated_comments:
    # Show similar examples found
    if st.session_state.playground_similar_examples:
        with st.expander("📚 Similar Content Found (used as prompt examples)", expanded=False):
            for i, example in enumerate(st.session_state.playground_similar_examples[:10], 1):
                st.markdown(f"**{i}. {example['title']}**")
                st.caption(f"Source: {example['source_name']}")
                st.caption(f"Similarity: {example['similarity_score']:.2%}")
                st.caption(f"Sample comments: {len(example['comments'])}")
                st.divider()
    
    # Display cached comments
    with st.container(height=400):
        for i, comment_text in enumerate(st.session_state.playground_generated_comments):
            avatar = st.session_state.playground_avatars[i] if i < len(st.session_state.playground_avatars) else "👤"
            with st.chat_message("user", avatar=avatar):
                st.write(comment_text)

