"""Home page for the Ghana Public Sentiments Observatory platform."""

from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.bigquery_manager import get_bigquery_manager
from pipeline.entities import KEY_ISSUES, KEY_PLAYERS
from ui_components import render_entity_card


@st.cache_data(ttl=st.session_state.get("cache_ttl", {}).get("sentiment_data", 300), show_spinner=False)
def get_latest_data_date() -> str | None:
    """Get the latest run date from the pipeline_entity_summaries table.
    
    Returns:
        str | None: Formatted string with the date and day of week, or None if no data.
    """
    manager = get_bigquery_manager()
    run_info = manager.get_latest_run_info()
    
    if run_info and 'completed_at' in run_info:
        completed_at = run_info['completed_at']
        
        day_of_week = completed_at.strftime('%A')
        formatted_date = completed_at.strftime('%B %d, %Y')
        formatted_time = completed_at.strftime('%H:%M:%S')
        
        return f"*Data as of: {day_of_week}, {formatted_date}, {formatted_time} GMT*"
    
    return None


@st.cache_data(ttl=st.session_state.get("cache_ttl", {}).get("sentiment_data", 300), show_spinner=False)
def load_current_sentiment_data(source_filter: Optional[str] = None) -> pd.DataFrame:
    """Load current sentiment data from pipeline_comment_sentiments table.
    
    Returns DataFrame with both tracked entities (KEY_PLAYERS + KEY_ISSUES) and discussion topics.
    Filters by maximum run_id with sentiment_count >= 2.
    
    Tracked entities: Those defined in KEY_PLAYERS or KEY_ISSUES.
    Discussion topics: All other entities found in the data.
    
    Args:
        source_filter: Optional filter for content type ('youtube' or 'facebook'). None = all sources.
    
    Returns:
        pd.DataFrame: DataFrame with entity sentiment data.
    """
    tracked_entity_names = set(KEY_PLAYERS.keys()) | set(KEY_ISSUES.keys())
    
    manager = get_bigquery_manager()
    rows = manager.get_current_sentiment_data(min_sentiment_count=2, source_filter=source_filter)
    
    if not rows:
        return pd.DataFrame()
    
    data = []
    for row in rows:
        entity_name = row['entity_name']
        
        is_tracked = entity_name in tracked_entity_names
        
        if entity_name in KEY_PLAYERS:
            entity_type = 'player'
        elif entity_name in KEY_ISSUES:
            entity_type = 'issue'
        else:
            entity_type = 'discussion'
        
        data.append({
            'Entity': entity_name,
            'Type': entity_type,
            'Is New': not is_tracked,  # True if discussion topic, False if tracked
            'Avg Sentiment': round(row['avg_sentiment'], 3) if row['avg_sentiment'] is not None else 0.0,
            'Mentions': row['mention_count'],
            'Content Items': row['content_count'],
            'Std Dev': round(row['std_dev'], 3) if row['std_dev'] is not None else 0.0,
            'Sentiment Summary': row['sentiment_summary'] if row['sentiment_summary'] else None
        })
    
    return pd.DataFrame(data)


@st.cache_data(ttl=st.session_state.get("cache_ttl", {}).get("sentiment_data", 300), show_spinner=False)
def load_sentiment_trends(source_filter: Optional[str] = None) -> pd.DataFrame:
    """Load sentiment trends over time for tracked entities only.
    
    Returns DataFrame with date, entity, avg_sentiment, sentiment_summary, and cumulative unique authors.
    Ensures only the latest run_id per date/entity combination.
    Includes only KEY_PLAYERS + KEY_ISSUES entities.
    
    Args:
        source_filter: Optional filter for content type ('youtube' or 'facebook'). None = all sources.
    
    Returns:
        pd.DataFrame: DataFrame with sentiment trend data.
    """
    tracked_entity_names = set(KEY_PLAYERS.keys()) | set(KEY_ISSUES.keys())
    
    manager = get_bigquery_manager()
    rows = manager.get_sentiment_trends_with_authors(min_sentiment_count=2, source_filter=source_filter)

    print(rows)
    
    if not rows:
        return pd.DataFrame()
    
    data = []
    for row in rows:
        entity_name = row['entity_name']
        
        if entity_name not in tracked_entity_names:
            continue
        
        if entity_name in KEY_PLAYERS:
            entity_type = 'player'
        elif entity_name in KEY_ISSUES:
            entity_type = 'issue'
        else:
            entity_type = 'unknown'
        
        date_obj = row['date']
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d').date()
        elif hasattr(date_obj, 'date'):
            date_obj = date_obj.date()
        
        data.append({
            'date': date_obj,
            'entity': entity_name,
            'entity_type': entity_type,
            'avg_sentiment': row['avg_sentiment'],
            'mentions': row['mention_count'],
            'sentiment_summary': row['sentiment_summary'] if row['sentiment_summary'] else 'No summary available',
            'unique_authors': row['cumulative_unique_authors'] if row['cumulative_unique_authors'] is not None else 0
        })
    
    return pd.DataFrame(data)


def plot_sentiment_trends(df: pd.DataFrame, selected_entities: list) -> None:
    """Plot sentiment trends over time for selected entities.
    
    Shows basic info in hover tooltip, full summaries in table below.
    
    Args:
        df: DataFrame with sentiment trend data.
        selected_entities: List of entity names to plot.
    """
    if df.empty or not selected_entities:
        st.warning("No data available to plot")
        return

    df_filtered = df[df['entity'].isin(selected_entities)].copy()

    if df_filtered.empty:
        st.warning("No data found for selected entities")
        return
    
    fig = go.Figure()
    
    for entity in selected_entities:
        entity_data = df_filtered[df_filtered['entity'] == entity].copy()
        
        if not entity_data.empty:
            # Simplified hover template without full summary
            fig.add_trace(
                go.Scatter(
                    x=entity_data['date'],
                    y=entity_data['avg_sentiment'],
                    mode='lines+markers',
                    name=entity,
                    line=dict(width=2),
                    marker=dict(size=8),
                    customdata=entity_data[['mentions', 'unique_authors']].values,
                    hovertemplate=(
                        f'<b>{entity}</b><br>' +
                        'Date: %{x}<br>' +
                        'Sentiment: %{y:.3f}<br>' +
                        'Mentions: %{customdata[0]}<br>' +
                        'Unique Authors: %{customdata[1]}<br>' +
                        '<extra></extra>'
                    )
            )
            )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Sentiment trends over time",
        xaxis_title="Date",
        yaxis_title="Average sentiment score",
        yaxis=dict(range=[-1, 1]),
        hovermode='closest',
        height=500,
        # template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True, theme=None)


def color_sentiment_rows(row: pd.Series) -> list[str]:
    """
    Apply color coding to sentiment rows based on average sentiment score.
    
    Args:
        row: A pandas Series containing sentiment data with 'Avg Sentiment' column.
    
    Returns:
        List of CSS background-color styles for the row.
    """
    score = row['Avg Sentiment']
    if score > 0.3:
        return ['background-color: #90EE90'] * len(row)  # Light green
    elif score < -0.3:
        return ['background-color: #FFB6C1'] * len(row)  # Light red
    else:
        return ['background-color: #FFFFE0'] * len(row)  # Light yellow


def render_newspaper_layout(
    discussion_topics_df: pd.DataFrame,
    tracked_entities_df: pd.DataFrame,
    show_popular_badge: bool = False
) -> None:
    """
    Render a 3-column newspaper layout:
    - Left: Discussion topics as expanders (each shows entity card)
    - Middle: Featured topic card + carousel to switch topics
    - Right: Key players/issues as expanders (each shows entity card)
    
    Args:
        discussion_topics_df: DataFrame of discussion topics.
        tracked_entities_df: DataFrame of tracked entities (players/issues).
        show_popular_badge: Whether to show popular badge.
    """
    # Sort dataframes
    discussion_sorted = discussion_topics_df.sort_values('Mentions', ascending=False) if not discussion_topics_df.empty else pd.DataFrame()
    tracked_sorted = tracked_entities_df.sort_values('Mentions', ascending=False) if not tracked_entities_df.empty else pd.DataFrame()
    
    # Get all entities (no multiselect filtering)
    all_discussions = discussion_sorted['Entity'].tolist() if not discussion_sorted.empty else []
    all_tracked = tracked_sorted['Entity'].tolist() if not tracked_sorted.empty else []
    
    # Get badge entities
    if not discussion_sorted.empty:
        discussion_most_negative = discussion_sorted.loc[discussion_sorted['Avg Sentiment'].idxmin(), 'Entity']
        discussion_most_positive = discussion_sorted.loc[discussion_sorted['Avg Sentiment'].idxmax(), 'Entity']
    
    if not tracked_sorted.empty:
        tracked_most_negative = tracked_sorted.loc[tracked_sorted['Avg Sentiment'].idxmin(), 'Entity']
        tracked_most_positive = tracked_sorted.loc[tracked_sorted['Avg Sentiment'].idxmax(), 'Entity']
        tracked_top_3 = tracked_sorted.head(3)['Entity'].tolist() if show_popular_badge else []
    
    # Create 3-column layout
    left_col, middle_col, right_col = st.columns([1, 2, 1], gap="medium")
    
    # LEFT COLUMN: Discussion topics
    with left_col:
        with st.container(border=False):
            st.markdown("📰 Discussion Topics")
            
            if all_discussions and not discussion_sorted.empty:
                for entity_name in all_discussions:
                    entity = discussion_sorted[discussion_sorted['Entity'] == entity_name].iloc[0]
                    is_best = (entity_name == discussion_most_positive)
                    is_worst = (entity_name == discussion_most_negative)
                    
                    with st.expander(entity_name, expanded=False):
                        st.markdown(render_entity_card(entity_name, entity, is_best, is_worst, False), unsafe_allow_html=True)
                        
                        sentiment_summary = entity.get('Sentiment Summary')
                        if sentiment_summary and pd.notna(sentiment_summary):
                            st.markdown("---")
                            st.markdown("**Summary:**")
                            st.markdown(sentiment_summary)
            else:
                st.info("No discussion topics available")
    
    # MIDDLE COLUMN: Featured topic + carousel
    with middle_col:
        with st.container(border=True):
            if all_discussions and not discussion_sorted.empty:
                # Initialize carousel index in session state
                carousel_key = "newspaper_carousel_index"
                if carousel_key not in st.session_state:
                    st.session_state[carousel_key] = 0
                
                # Current featured topic
                current_idx = st.session_state[carousel_key] % len(all_discussions)
                featured_entity_name = all_discussions[current_idx]
                featured_entity = discussion_sorted[discussion_sorted['Entity'] == featured_entity_name].iloc[0]
                
                is_best = (featured_entity_name == discussion_most_positive)
                is_worst = (featured_entity_name == discussion_most_negative)
                
                # Display featured entity card
                st.markdown(render_entity_card(featured_entity_name, featured_entity, is_best, is_worst, False), unsafe_allow_html=True)
                
                # Sentiment summary
                sentiment_summary = featured_entity.get('Sentiment Summary')
                if sentiment_summary and pd.notna(sentiment_summary):
                    with st.container(border=True):
                        st.markdown(sentiment_summary)
                
                # Carousel navigation
                if len(all_discussions) > 1:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if st.button("◀ Previous", key=f"{carousel_key}_prev", use_container_width=True):
                            st.session_state[carousel_key] = (current_idx - 1) % len(all_discussions)
                            st.rerun()
                    
                    with col2:
                        st.markdown(f"<div style='text-align: center; padding: 8px;'>{current_idx + 1} of {len(all_discussions)}</div>", unsafe_allow_html=True)
                    
                    with col3:
                        if st.button("Next ▶", key=f"{carousel_key}_next", use_container_width=True):
                            st.session_state[carousel_key] = (current_idx + 1) % len(all_discussions)
                            st.rerun()
            else:
                st.info("No discussion topics available")
    
    # RIGHT COLUMN: Key players/issues
    with right_col:
        with st.container(border=False):
            st.markdown("🎯 Key Players & Issues")
            
            if all_tracked and not tracked_sorted.empty:
                for entity_name in all_tracked:
                    entity = tracked_sorted[tracked_sorted['Entity'] == entity_name].iloc[0]
                    is_best = (entity_name == tracked_most_positive)
                    is_worst = (entity_name == tracked_most_negative)
                    is_popular = (entity_name in tracked_top_3) if show_popular_badge else False
                    
                    with st.expander(entity_name, expanded=False):
                        st.markdown(render_entity_card(entity_name, entity, is_best, is_worst, is_popular), unsafe_allow_html=True)
                        
                        sentiment_summary = entity.get('Sentiment Summary')
                        if sentiment_summary and pd.notna(sentiment_summary):
                            st.markdown("---")
                            st.markdown("**Summary:**")
                            st.markdown(sentiment_summary)
            else:
                st.info("No key players/issues available")


def render_entity_tab(
    entities_df: pd.DataFrame,
    session_state_key: str,
    multiselect_key: str,
    label: str,
    help_text: str,
    show_popular_badge: bool = False
) -> None:
    """
    Render a tab displaying entity cards with sentiment data.
    
    Args:
        entities_df: DataFrame of entities to display.
        session_state_key: Key for storing selection in session state.
        multiselect_key: Unique key for the multiselect widget.
        label: Label for the multiselect widget.
        help_text: Help text for the multiselect widget.
        show_popular_badge: Whether to show "Popular" badge for top 3 entities by mentions.
    """
    if entities_df.empty:
        st.info(f"No {label.lower()} with 2 or more comments found.")
        return
    
    # Sort by mentions
    entities_sorted = entities_df.sort_values('Mentions', ascending=False)
    
    # Get top 6 for default selection
    top_6_entities = entities_sorted.head(6)['Entity'].tolist()
    
    # Get entity with most negative and positive sentiment for badges
    most_negative = entities_sorted.loc[entities_sorted['Avg Sentiment'].idxmin(), 'Entity']
    most_positive = entities_sorted.loc[entities_sorted['Avg Sentiment'].idxmax(), 'Entity']

    # Display latest data date
    latest_date = get_latest_data_date()
    if latest_date:
        st.caption(latest_date)

    # Determine default selection: use session state if available, otherwise top 6
    if st.session_state[session_state_key] is None:
        default_selection = top_6_entities
    else:
        # Use cached selection, but filter to only valid entities still in data
        valid_cached = [
            e for e in st.session_state[session_state_key]
            if e in entities_sorted['Entity'].tolist()
        ]
        default_selection = valid_cached if valid_cached else top_6_entities
    
    # Multiselect for entities
    selected_entities = st.multiselect(
        label,
        options=entities_sorted['Entity'].tolist(),
        default=default_selection,
        key=multiselect_key,
        help=help_text
    )
    
    # Update session state with current selection
    st.session_state[session_state_key] = selected_entities
    st.markdown("")
    
    # Get top 3 entities by mentions for "Popular" badge (if enabled)
    top_3_by_mentions = entities_sorted.head(3)['Entity'].tolist() if show_popular_badge else []
    
    # Display entities in 2-column layout
    if selected_entities:
        for idx, entity_name in enumerate(selected_entities):
            entity = entities_sorted[entities_sorted['Entity'] == entity_name].iloc[0]
            
            # Check if this entity is best, worst, or popular
            is_best = (entity_name == most_positive)
            is_worst = (entity_name == most_negative)
            is_popular = (entity_name in top_3_by_mentions) if show_popular_badge else False
            
            # Create 2 columns for each row
            if idx % 2 == 0:
                col1, col2 = st.columns(2, gap="medium")
            
            # Determine which column to use
            column = col1 if idx % 2 == 0 else col2
            
            with column:
                with st.container(border=True):
                    st.markdown(render_entity_card(entity_name, entity, is_best, is_worst, is_popular), unsafe_allow_html=True)
                    
                    # Add Details expander with sentiment summary
                    sentiment_summary = entity.get('Sentiment Summary')
                    if sentiment_summary and pd.notna(sentiment_summary):
                        with st.expander("📋 Details"):
                            st.markdown(sentiment_summary)
                    else:
                        with st.expander("📋 Details"):
                            st.info("No sentiment summary available for this entity.")
                st.markdown("")  # Spacer
    else:
        st.info(f"Please select at least one {label.lower()} to display.")


def main() -> None:
    """Main function to render the home page with sentiment analysis tabs."""
    # Initialize session state for multiselect persistence
    if "home_discussion_entities" not in st.session_state:
        st.session_state.home_discussion_entities = None
    if "home_tracked_entities" not in st.session_state:
        st.session_state.home_tracked_entities = None
    if "home_trends_entities" not in st.session_state:
        st.session_state.home_trends_entities = None
    if "data_source_filter" not in st.session_state:
        st.session_state.data_source_filter = "All Sources"

    top_col1, top_col2 = st.columns([3, 1])
    
    with top_col1:
        st.markdown("##### 💬 What People are Saying")
        latest_date = get_latest_data_date()
        if latest_date:
            st.caption(latest_date)
    
    with top_col2:
        # Add source filter dropdown
        source_options = ["All Sources", "YouTube Only", "Facebook Only"]
        selected_source = st.selectbox(
            "Data source:",
            options=source_options,
            index=source_options.index(st.session_state.data_source_filter),
            key="source_filter_select",
            help="Filter sentiment data by source",
            label_visibility="collapsed"
        )
        st.session_state.data_source_filter = selected_source
    
    # Map UI selection to filter parameter
    source_param = None
    if selected_source == "YouTube Only":
        source_param = "youtube"
    elif selected_source == "Facebook Only":
        source_param = "facebook"
    
    # Load data
    with st.spinner("Loading sentiment data..."):
        current_data = load_current_sentiment_data(source_filter=source_param)
    
    if current_data.empty:
        st.warning("⚠️ No sentiment data found in the database.")
        return

    # Separate tracked entities and discussion topics
    tracked_entities = current_data[current_data['Is New'] == False].copy()
    discussion_topics = current_data[current_data['Is New'] == True].copy()
    
    # Filter out entities with fewer than 2 comments (to compute std dev safely)
    tracked_entities = tracked_entities[tracked_entities['Mentions'] >= 2]
    discussion_topics = discussion_topics[discussion_topics['Mentions'] >= 2]
    
    # Get top 5 tracked entities by mentions
    tracked_entities_sorted = tracked_entities.sort_values('Mentions', ascending=False)
    top_5_tracked = tracked_entities_sorted.head(5)
    top_5_entity_names = top_5_tracked['Entity'].tolist()
    
    # TODAY'S SENTIMENTS
    st.markdown("")
    render_newspaper_layout(
        discussion_topics_df=discussion_topics,
        tracked_entities_df=top_5_tracked,
        show_popular_badge=True
    )
    
    # HISTORICAL VIEW
    st.markdown("")
    st.markdown("##### 📈 Historical Trends")
    
    with st.spinner("Loading trend data..."):
        trends_df = load_sentiment_trends(source_filter=source_param)
    
    if not trends_df.empty:
        # Get unique entities
        all_entities = sorted(trends_df['entity'].unique())
        
        # Determine default selection: use top 5 tracked entities from current data
        if top_5_entity_names:
            # Filter top 5 to only those present in trends data
            default_selection = [e for e in top_5_entity_names if e in all_entities]
            # If none of the top 5 are in trends, fall back to first 5 from trends
            if not default_selection:
                default_selection = all_entities[:5] if len(all_entities) >= 5 else all_entities
        else:
            default_selection = all_entities[:5] if len(all_entities) >= 5 else all_entities
        
        # Override with session state if available
        if st.session_state.home_trends_entities is not None:
            valid_cached = [
                e for e in st.session_state.home_trends_entities
                if e in all_entities
            ]
            if valid_cached:
                default_selection = valid_cached
        
        # Entity selector
        selected_entities = st.multiselect(
            "Select key players or issues to plot:",
            options=all_entities,
            default=default_selection,
            help="Choose which entities to display on the chart",
            key="trends_entities_select"
        )
        
        # Update session state with current selection
        st.session_state.home_trends_entities = selected_entities
        
        if selected_entities:
            plot_sentiment_trends(trends_df, selected_entities)
        else:
            st.info("Please select at least one entity to plot")
    else:
        st.warning("Not enough data for time-series analysis. Run sentiment analysis to collect more data.")

if __name__ == "__main__":
    main()
