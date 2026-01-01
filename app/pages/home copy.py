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
        
        return f"*Data as of: 📅 {day_of_week}, {formatted_date}, {formatted_time} GMT*"
    
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
    
    # Add expandable section with detailed summaries table
    with st.expander("📋 View Detailed Summaries", expanded=False):
        # Entity selector dropdown
        selected_summary_entity = st.selectbox(
            "Select entity to view summaries:",
            options=selected_entities,
            key="summary_entity_selector"
        )
        
        # Filter data for selected entity
        summary_df = df_filtered[df_filtered['entity'] == selected_summary_entity].copy()

        # Sort by date descending
        summary_df = summary_df.sort_values('date', ascending=False)
        
        if not summary_df.empty:
            # Prepare table data with formatted columns
            table_data = summary_df.copy()
            
            # Format date as string
            table_data['Date'] = table_data['date'].astype(str)
            
            # Select and rename columns for display
            display_df = table_data[[
                'Date', 'avg_sentiment', 'mentions', 'unique_authors', 'sentiment_summary'
            ]].rename(columns={
                'avg_sentiment': 'Sentiment Score',
                'mentions': 'Mentions',
                'unique_authors': 'Unique Authors',
                'sentiment_summary': 'Summary'
            })
            
            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Sentiment Score": st.column_config.NumberColumn(
                        "Sentiment Score",
                        format="%.3f",
                        width="small"
                    ),
                    "Mentions": st.column_config.NumberColumn("Mentions", width="small"),
                    "Unique Authors": st.column_config.NumberColumn(
                        "Unique Authors",
                        help="Cumulative count of unique authors mentioning this entity up to this date",
                        width="small"
                    ),
                    "Summary": st.column_config.TextColumn("Summary", width="large")
                }
            )
        else:
            st.info("No summary data available for the selected entity.")


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
    
    # Add source filter dropdown at the top
    source_options = ["All Sources", "YouTube Only", "Facebook Only"]
    selected_source = st.selectbox(
        "Choose data source:",
        options=source_options,
        index=source_options.index(st.session_state.data_source_filter),
        key="source_filter_select",
        help="Filter sentiment data by source (YouTube videos or Facebook posts)"
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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 What's Trending?",
        "🎯 Key Players & Issues",
       "📈 Historical View"
    ])
    
    # TAB 1: TODAY'S SENTIMENTS (DISCUSSION TOPICS ONLY)
    with tab1:
        render_entity_tab(
            entities_df=discussion_topics,
            session_state_key="home_discussion_entities",
            multiselect_key="discussion_entities_select",
            label="Select discussion topics:",
            help_text="Preselected: Top 6 discussion topics",
            show_popular_badge=False
        )
    
    # TAB 2: RECURRING ENTITIES (TRACKED ENTITIES)
    with tab2:
        render_entity_tab(
            entities_df=tracked_entities,
            session_state_key="home_tracked_entities",
            multiselect_key="tracked_entities_select",
            label="Select key players or issues:",
            help_text="Preselected: Top 6 tracked entities",
            show_popular_badge=True
        )
    
    # TAB 3: TRENDS OVER TIME
    with tab3:
        with st.spinner("Loading trend data..."):
            trends_df = load_sentiment_trends(source_filter=source_param)
        
        if not trends_df.empty:
            # Get unique entities
            all_entities = sorted(trends_df['entity'].unique())
            
            # Determine default selection: use session state if available, otherwise top 5
            top_5_entities = all_entities[:5] if len(all_entities) >= 5 else all_entities
            if st.session_state.home_trends_entities is None:
                default_selection = top_5_entities
            else:
                # Use cached selection, but filter to only valid entities still in data
                valid_cached = [
                    e for e in st.session_state.home_trends_entities
                    if e in all_entities
                ]
                default_selection = valid_cached if valid_cached else top_5_entities
            
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
