"""Chat tool implementations for the sentiment analysis chatbot.

This module contains function implementations that the LLM can call
to interact with sentiment data, create visualizations, and manipulate dataframes.

Tools are defined using Pydantic models, which automatically generate OpenAI-compatible
function schemas for the LLM.
"""

import json
import traceback
import uuid
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_client
from database.bigquery_manager import get_bigquery_manager
from input_sanitizer import sanitize_entity_name
from pipeline.stages.normalization import normalize_channels
from pipeline.stages.sentiments import build_prompt, create_sentiments_schema
from pipeline.stages.smoothing import smooth_sentiments


__all__ = [
    "ColumnFilter",
    "RowFilter",
    "Aggregation",
    "GetDataframeParams",
    "CreateDataframeParams",
    "CreatePlotlyFigureParams",
    "AnalyzeCustomEntityParams",
    "get_dataframe",
    "create_dataframe",
    "create_plotly_figure",
    "analyze_custom_entity_sentiment",
    "AVAILABLE_TOOLS",
    "TOOL_FUNCTIONS",
]


class ColumnFilter(BaseModel):
    """Select specific columns from a dataframe."""
    
    columns: list[str] = Field(..., description="List of column names to select")


class RowFilter(BaseModel):
    """Filter rows based on column values."""
    
    column: str = Field(..., description="Column name to filter on")
    operator: Literal["==", ">", "<", ">=", "<=", "!=", "in", "contains"] = Field(
        ..., description="Comparison operator"
    )
    value: Any = Field(..., description="Value to compare against")


class Aggregation(BaseModel):
    """Aggregate data by grouping and applying functions."""
    
    groupby: list[str] = Field(..., description="Columns to group by")
    agg: dict[str, str] = Field(
        ..., 
        description="Aggregation functions: {'column': 'mean|sum|count|min|max'}"
    )


class GetDataframeParams(BaseModel):
    """Parameters for filtering or aggregating dataframes."""
    
    dataframe_id: Literal["df_entity_summaries"] = Field(
        ...,
        description="ID of dataframe: 'df_entity_summaries' for entity aggregates"
    )
    column_filters: list[ColumnFilter] | None = Field(
        None,
        description="Select specific columns: [{'columns': ['col1', 'col2']}]"
    )
    row_filters: list[RowFilter] | None = Field(
        None,
        description="Filter rows: [{'column': 'sentiment_score', 'operator': '>', 'value': 0.5}]"
    )
    aggregations: list[Aggregation] | None = Field(
        None,
        description="Aggregate data: [{'groupby': ['entity_name'], 'agg': {'sentiment_score': 'mean'}}]"
    )
    limit: int | None = Field(
        None,
        description="Maximum rows to return (use to prevent overwhelming responses)"
    )


class CreateDataframeParams(BaseModel):
    """Parameters for creating a new dataframe."""
    
    records: list[dict[str, Any]] = Field(
        ...,
        description="List of dictionaries representing rows: [{'col1': val1, 'col2': val2}, ...]"
    )
    dataframe_name: str | None = Field(
        None,
        description="Optional name for the dataframe (auto-generated if omitted)"
    )


class CreatePlotlyFigureParams(BaseModel):
    """Parameters for creating a plotly visualization."""
    
    chart_type: Literal["line", "bar", "scatter", "pie", "histogram"] = Field(
        ...,
        description="Type of chart to create"
    )
    dataframe_id: str | None = Field(
        None,
        description="ID of dataframe to visualize"
    )
    x_column: str | None = Field(
        None,
        description="Column name for x-axis (if using dataframe)"
    )
    y_column: str | None = Field(
        None,
        description="Column name for y-axis (if using dataframe)"
    )
    x_values: list[Any] | None = Field(
        None,
        description="Direct x values (if not using dataframe)"
    )
    y_values: list[Any] | None = Field(
        None,
        description="Direct y values (if not using dataframe)"
    )
    title: str = Field(
        "",
        description="Chart title"
    )
    x_title: str = Field(
        "",
        description="X-axis title"
    )
    y_title: str = Field(
        "",
        description="Y-axis title"
    )
    color_column: str | None = Field(
        None,
        description="Column to use for coloring points/bars"
    )
    additional_params: dict[str, Any] | None = Field(
        None,
        description="Additional plotly parameters"
    )


class AnalyzeCustomEntityParams(BaseModel):
    """Parameters for analyzing sentiment for a custom entity."""
    
    entity_name: str = Field(
        ...,
        description="Name of the entity to analyze sentiment for (e.g., 'Alan Kyerematen', 'Free SHS policy')"
    )
    entity_type: Literal["person", "organization", "issue"] = Field(
        ...,
        description="Type of entity: 'person' or 'organization' (uses fuzzy matching), or 'issue' (uses embedding-based matching)"
    )
    aggregate_by: Literal["day", "week", "month", "overall"] | None = Field(
        "day",
        description="Time aggregation level: 'day', 'week', 'month', or 'overall' for single aggregate (default: 'day' for trends)"
    )
    limit_comments: int = Field(
        default=1000,
        ge=1,  # Greater than or equal to 1
        le=10000,  # Less than or equal to 10000
        description="Maximum number of comments to analyze (default 1000, min 1, max 10000)"
    )
    source_filter: Literal["youtube", "facebook", "all"] | None = Field(
        "all",
        description="Data source to analyze: 'youtube' for YouTube only, 'facebook' for Facebook only, or 'all' for both sources (default: 'all')"
    )
    
    @field_validator('entity_name')
    @classmethod
    def validate_entity_name(cls, v: str) -> str:
        """Validate and sanitize entity name."""
        return sanitize_entity_name(v)


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def get_dataframe(params: GetDataframeParams) -> str:
    """Filter or aggregate data from session state dataframes.
    
    Args:
        params: GetDataframeParams model with filtering/aggregation parameters.
    
    Returns:
        str: JSON string of filtered/aggregated dataframe.
    """
    if params.dataframe_id in ["df_entity_summaries"]:
        df = st.session_state.get(params.dataframe_id, pd.DataFrame()).copy()
    else:
        df = st.session_state.get("created_dataframes", {}).get(params.dataframe_id, pd.DataFrame()).copy()
    
    if df.empty:
        return json.dumps({"error": f"Dataframe {params.dataframe_id} not found or empty"})
    
    # Apply column filters (select specific columns)
    if params.column_filters:
        for cf in params.column_filters:
            if isinstance(cf, dict):
                cf = ColumnFilter(**cf)
            df = df[cf.columns]
    
    # Apply row filters
    if params.row_filters:
        for rf in params.row_filters:
            if isinstance(rf, dict):
                rf = RowFilter(**rf)
            
            if rf.column not in df.columns:
                continue
                
            if rf.operator == "==":
                df = df[df[rf.column] == rf.value]
            elif rf.operator == ">":
                df = df[df[rf.column] > rf.value]
            elif rf.operator == "<":
                df = df[df[rf.column] < rf.value]
            elif rf.operator == ">=":
                df = df[df[rf.column] >= rf.value]
            elif rf.operator == "<=":
                df = df[df[rf.column] <= rf.value]
            elif rf.operator == "!=":
                df = df[df[rf.column] != rf.value]
            elif rf.operator == "in":
                df = df[df[rf.column].isin(rf.value if isinstance(rf.value, list) else [rf.value])]
            elif rf.operator == "contains":
                df = df[df[rf.column].astype(str).str.contains(str(rf.value), case=False, na=False)]
    
    # Apply aggregations
    if params.aggregations:
        for agg in params.aggregations:
            if isinstance(agg, dict):
                agg = Aggregation(**agg)
            if agg.groupby and agg.agg:
                df = df.groupby(agg.groupby).agg(agg.agg).reset_index()
    
    # Apply limit
    if params.limit:
        df = df.head(params.limit)
    
    return df.to_json(orient="records")


def create_dataframe(params: CreateDataframeParams) -> str:
    """Create a new dataframe from column-value records.
    
    Args:
        params: CreateDataframeParams model with records and optional name.

    Returns:
        str: JSON string with dataframe_id, head, and summary.
    """
    try:
        df = pd.DataFrame(params.records)
        
        df_id = params.dataframe_name or f"custom_df_{uuid.uuid4().hex[:8]}"
        
        st.session_state["created_dataframes"][df_id] = df
        
        response = {
            "dataframe_id": df_id,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "head": json.loads(df.head(10).to_json(orient="records")),
            "summary": json.loads(df.describe(include='all').to_json()),
            "columns": df.columns.tolist()
        }
        
        return json.dumps(response)
    except Exception as e:
        return json.dumps({"error": f"Failed to create dataframe: {str(e)}"})


def create_plotly_figure(params: CreatePlotlyFigureParams) -> str:
    """Create a plotly figure for visualization.
    
    Args:
        params: CreatePlotlyFigureParams model with chart configuration.
    
    Returns:
        str: JSON string with figure_id and status.
    """
    try:
        df = None
        if params.dataframe_id:
            if params.dataframe_id in ["df_entity_summaries"]:
                df = st.session_state.get(params.dataframe_id, pd.DataFrame())
            else:
                df = st.session_state.get("created_dataframes", {}).get(params.dataframe_id, pd.DataFrame())
        
        if df is not None and not df.empty:
            x_data = df[params.x_column] if params.x_column else params.x_values
            y_data = df[params.y_column] if params.y_column else params.y_values
        else:
            x_data = params.x_values
            y_data = params.y_values
        
        fig = None
        if params.chart_type == "line":
            fig = go.Figure(data=go.Scatter(
                x=x_data, 
                y=y_data, 
                mode='lines+markers',
                marker=dict(size=8),
                line=dict(width=2)
            ))
        elif params.chart_type == "bar":
            fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
        elif params.chart_type == "scatter":
            fig = go.Figure(data=go.Scatter(
                x=x_data, 
                y=y_data, 
                mode='markers',
                marker=dict(size=10)
            ))
        elif params.chart_type == "pie":
            fig = go.Figure(data=go.Pie(labels=x_data, values=y_data))
        elif params.chart_type == "histogram":
            fig = go.Figure(data=go.Histogram(x=x_data))
        else:
            return json.dumps({"error": f"Unsupported chart type: {params.chart_type}"})
        
        fig.update_layout(
            title=dict(text=params.title, font=dict(size=18)),
            xaxis_title=params.x_title,
            yaxis_title=params.y_title,
            template="plotly_white",
            hovermode='closest',
            showlegend=True if params.color_column else False,
            height=500
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        fig_id = f"fig_{uuid.uuid4().hex[:8]}"
        st.session_state["created_figures"][fig_id] = fig
        
        response = {
            "figure_id": fig_id,
            "chart_type": params.chart_type,
            "title": params.title,
            "status": "success"
        }
        
        return json.dumps(response)
    except Exception as e:
        return json.dumps({"error": f"Failed to create figure: {str(e)}"})


def analyze_custom_entity_sentiment(params: AnalyzeCustomEntityParams) -> str:
    """Analyze sentiment for a custom entity by detecting it in comments and applying the full pipeline.
    
    This tool:
    1. Queries YouTube videos and comments from the database (with timestamps)
    2. Detects the entity in comments using fuzzy or embedding-based matching (based on entity type)
    3. Groups data by time period and processes each period independently through the pipeline
    4. Generates sentiment scores using LLM
    5. Applies KNN smoothing using embeddings
    6. Applies channel normalization
    7. Aggregates results by time period to show trends
    
    Args:
        params: AnalyzeCustomEntityParams model with entity details and aggregation settings.
    
    Returns:
        str: JSON string with sentiment analysis results including time series data.
    """
    try:
        # Entity name is already sanitized by Pydantic validator
        sanitized_entity = params.entity_name
        # limit_comments is already validated by Pydantic (ge=1, le=10000)
        limit_comments = params.limit_comments
        
        manager = get_bigquery_manager()
        
        # Build query based on source filter
        query_parts = []
        
        # Add YouTube query if not filtered to Facebook only
        if params.source_filter in ["youtube", "all"]:
            query_parts.append(f"""
                SELECT 
                    CAST(v.id AS STRING) as content_id,
                    'youtube' as content_type,
                    v.title,
                    v.channel_name as source_name,
                    c.comment_text,
                    c.author_id,
                    v.published_date,
                    DATE(v.published_date) as publish_date
                FROM `{manager.dataset_id}.youtube_videos` v
                LEFT JOIN `{manager.dataset_id}.youtube_comments` c ON v.id = c.video_id
                WHERE c.comment_text IS NOT NULL
                AND v.published_date IS NOT NULL
            """)
        
        # Add Facebook query if not filtered to YouTube only
        if params.source_filter in ["facebook", "all"]:
            query_parts.append(f"""
                SELECT 
                    CAST(p.id AS STRING) as content_id,
                    'facebook' as content_type,
                    SUBSTR(p.message, 1, 100) as title,
                    p.page_name as source_name,
                    c.comment_text,
                    c.author_id,
                    p.created_date as published_date,
                    DATE(p.created_date) as publish_date
                FROM `{manager.dataset_id}.facebook_posts` p
                LEFT JOIN `{manager.dataset_id}.facebook_comments` c ON p.id = c.post_id
                WHERE c.comment_text IS NOT NULL
                AND p.created_date IS NOT NULL
            """)
        
        if not query_parts:
            return json.dumps({"error": "No valid source filter specified"})
        
        # Combine query parts
        query = "\nUNION ALL\n".join(query_parts)
        query += f"\nORDER BY published_date DESC\nLIMIT {limit_comments}"
        
        try:
            content_data = manager._query(query)
        except Exception as e:
            # Handle table not found errors gracefully
            if "Not found: Table" in str(e):
                return json.dumps({
                    "error": f"Required tables not found for source filter '{params.source_filter}'. The data source may not be available.",
                    "source_filter": params.source_filter
                })
            raise
        
        if not content_data:
            return json.dumps({"error": "No content data available"})
        
        df_content = pd.DataFrame(content_data)
        
        # Detect entity mentions using appropriate method based on entity type
        if params.entity_type in ["person", "organization"]:
            # Use fuzzy matching for people and organizations
            entity_lower = sanitized_entity.lower()
            fuzzy_threshold = 80
            
            def entity_mentioned(comment_text: str) -> bool:
                """Check if entity is mentioned using fuzzy matching."""
                if not comment_text:
                    return False
                comment_lower = comment_text.lower()
                score = fuzz.partial_ratio(entity_lower, comment_lower)
                return score >= fuzzy_threshold
            
            df_content['mentions_entity'] = df_content['comment_text'].apply(entity_mentioned)
            
        else:  # issue
            # Use embedding-based matching for issues
            model = SentenceTransformer('all-MiniLM-L6-v2')
            entity_embedding = model.encode([sanitized_entity])[0].reshape(1, -1)
            embed_threshold = 0.5
            
            # Chunk comments into 5-word segments for better matching
            def create_chunks(text: str, chunk_size: int = 5) -> list[str]:
                if not text:
                    return []
                words = text.split()
                chunks = []
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i+chunk_size]
                    if chunk_words:
                        chunks.append(' '.join(chunk_words))
                return chunks
            
            def entity_mentioned_embedding(comment_text: str) -> bool:
                """Check if entity is mentioned using embedding similarity."""
                if not comment_text:
                    return False
                chunks = create_chunks(comment_text)
                if not chunks:
                    return False
                
                chunk_embeddings = model.encode(chunks, show_progress_bar=False)
                similarities = cosine_similarity(entity_embedding, chunk_embeddings)[0]
                max_similarity = similarities.max()
                return max_similarity >= embed_threshold
            
            df_content['mentions_entity'] = df_content['comment_text'].apply(entity_mentioned_embedding)
        
        df_relevant = df_content[df_content['mentions_entity']].copy()
        
        if df_relevant.empty:
            return json.dumps({
                "entity_name": params.entity_name,
                "entity_type": params.entity_type,
                "error": f"No mentions of '{params.entity_name}' found in the analyzed comments",
                "comments_analyzed": len(df_content)
            })
        
        # Group by time period for independent pipeline processing
        time_series_results = []
        
        if params.aggregate_by == "overall":
            # Single group for overall analysis
            time_groups = [("overall", df_relevant)]
        else:
            # Group by time period
            df_relevant['publish_date'] = pd.to_datetime(df_relevant['publish_date'])
            
            if params.aggregate_by == "day":
                df_relevant['time_group'] = df_relevant['publish_date']
            elif params.aggregate_by == "week":
                df_relevant['time_group'] = df_relevant['publish_date'].dt.to_period('W').dt.start_time
            elif params.aggregate_by == "month":
                df_relevant['time_group'] = df_relevant['publish_date'].dt.to_period('M').dt.start_time
            
            time_groups = list(df_relevant.groupby('time_group'))
        
        # Process each time group independently through the pipeline
        # Get model and API key from session state
        selected_model = st.session_state.get("chat_model")
        chat_api_key = st.session_state.get("chat_api_key")
        
        if not selected_model:
            return json.dumps({
                "error": "No model selected. Please select a model in the chat settings."
            })
        
        client = get_client(selected_model, api_key=chat_api_key)
        
        for time_label, group_df in time_groups:
            # Group by content_id within this time period
            grouped_data = []
            for content_id, content_group in group_df.groupby('content_id'):
                content_info = content_group.iloc[0]
                grouped_data.append({
                    'content_id': content_id,
                    'source_type': content_info['content_type'],
                    'source_name': content_info['source_name'],
                    'news_title': content_info['title'],
                    'comment_texts': content_group['comment_text'].tolist(),
                    'author_ids': content_group['author_id'].tolist()
                })
            
            sentiment_results = []
            
            for item in grouped_data[:20]:  # Limit to 20 content items per time period
                comments = item['comment_texts'][:30]  # Limit to 30 comments per content item
                author_ids = item['author_ids'][:30]
                
                comments_by_author = {}
                for i, (comment, author_id) in enumerate(zip(comments, author_ids)):
                    author_key = author_id if author_id else f"author_{i}"
                    if author_key not in comments_by_author:
                        comments_by_author[author_key] = []
                    comments_by_author[author_key].append({'text': comment})
                
                authors = list(comments_by_author.keys())
                Sentiments, field_to_author, field_to_entity = create_sentiments_schema(
                    authors, [params.entity_name]
                )
                
                prompt = build_prompt(comments_by_author, [params.entity_name], {'title': item['news_title']})
                
                try:
                    response = client.beta.chat.completions.parse(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format=Sentiments
                    )
                    
                    result = response.choices[0].message.parsed
                    
                    for field_name, author_id in field_to_author.items():
                        author_sentiments = getattr(result, field_name)
                        entity_field = list(field_to_entity.keys())[0]
                        sentiment_score = getattr(author_sentiments, entity_field)
                        
                        if sentiment_score is not None:
                            author_comments = comments_by_author[author_id]
                            comment_text = author_comments[0]['text'] if author_comments else ""
                            
                            sentiment_results.append({
                                'content_id': item['content_id'],
                                'author_id': author_id,
                                'entity_name': params.entity_name,
                                'sentiment': sentiment_score,
                                'comment_text': comment_text,
                                'source_type': item['source_type'],
                                'source_name': item['source_name']
                            })
                except Exception:
                    continue
            
            if not sentiment_results:
                continue
            
            df_sentiments = pd.DataFrame(sentiment_results)
            
            df_smoothed, smooth_stats = smooth_sentiments(
                df_sentiments,
                k_neighbors=5,
                min_neighbors=1
            )
            
            df_normalized, norm_stats = normalize_channels(
                df_smoothed,
                window_days=30,
                normalization_strength=0.8
            )
            
            time_result = {
                "date": str(time_label) if time_label != "overall" else "overall",
                "avg_sentiment": float(df_normalized['normalized_sentiment'].mean()),
                "sentiment_std": float(df_normalized['normalized_sentiment'].std()),
                "sentiment_count": len(df_normalized),
                "content_count": df_normalized['content_id'].nunique(),
                "mentions_count": len(group_df)
            }
            time_series_results.append(time_result)
        
        if not time_series_results:
            return json.dumps({
                "entity_name": params.entity_name,
                "entity_type": params.entity_type,
                "error": "Failed to generate sentiments for detected mentions",
                "mentions_found": len(df_relevant)
            })
        
        if params.aggregate_by != "overall":
            time_series_results.sort(key=lambda x: x['date'])
        
        all_sentiments = [r['avg_sentiment'] for r in time_series_results]
        result_data = {
            "entity_name": params.entity_name,
            "entity_type": params.entity_type,
            "matching_method": "fuzzy" if params.entity_type in ["person", "organization"] else "embedding",
            "aggregation": params.aggregate_by,
            "overall_avg_sentiment": float(np.mean(all_sentiments)),
            "overall_sentiment_std": float(np.std(all_sentiments)),
            "total_mentions": len(df_relevant),
            "total_periods": len(time_series_results),
            "comments_analyzed": len(df_content),
            "time_series": time_series_results
        }
        
        df_time_series = pd.DataFrame(time_series_results)
        df_id = f"custom_entity_{uuid.uuid4().hex[:8]}"
        st.session_state["created_dataframes"][df_id] = df_time_series
        result_data["dataframe_id"] = df_id
        
        return json.dumps(result_data)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to analyze custom entity: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ============================================================================
# SCHEMA GENERATION
# ============================================================================

def pydantic_to_openai_schema(
    model: type[BaseModel],
    name: str,
    description: str
) -> dict:
    """Convert a Pydantic model to an OpenAI function schema.
    
    Args:
        model: Pydantic model class.
        name: Function name.
        description: Function description.
    
    Returns:
        dict: OpenAI-compatible function schema.
    """
    schema = model.model_json_schema()
    
    def resolve_refs(obj: Any, defs: dict) -> Any:
        """Recursively resolve $ref references in the schema."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve_refs(defs[ref_name], defs)
                return obj
            else:
                return {k: resolve_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_refs(item, defs) for item in obj]
        else:
            return obj
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    if "properties" in schema:
        parameters["properties"] = schema["properties"]
    
    if "required" in schema:
        parameters["required"] = schema["required"]
    
    if "$defs" in schema:
        parameters["properties"] = resolve_refs(parameters["properties"], schema["$defs"])
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

# Define available tools with their Pydantic models and implementations
AVAILABLE_TOOLS = [
    pydantic_to_openai_schema(
        GetDataframeParams,
        "get_dataframe",
        """
        Filter or aggregate data from sentiment datasets. 
        Use this to retrieve specific data needed to answer queries. 
        Returns filtered data as JSON.
        Note: Only entity summaries are available, not individual comment sentiments.
        """
    ),
    pydantic_to_openai_schema(
        CreateDataframeParams,
        "create_dataframe",
        """
        Create a new custom dataframe from records. 
        Useful for manipulating data to create derived datasets.
        """
    ),
    pydantic_to_openai_schema(
        CreatePlotlyFigureParams,
        "create_plotly_figure",
        """
        Create a plotly visualization. 
        Use this to show charts to users.
        """
    ),
    pydantic_to_openai_schema(
        AnalyzeCustomEntityParams,
        "analyze_custom_entity_sentiment",
        """
        Analyze sentiment for a custom entity not in the predefined key issues/players.
        Queries comments with timestamps from selected data sources (YouTube, Facebook, or both), 
        detects entity mentions using fuzzy matching (for people/organizations) or embedding-based 
        matching (for issues), generates sentiments using LLM, and applies the full pipeline 
        (smoothing + normalization).
        
        IMPORTANT: User must specify entity_type:
        - 'person' or 'organization': Uses fuzzy string matching (best for names)
        - 'issue': Uses embedding-based semantic matching (best for policies/topics)
        
        Data source options:
        - 'youtube': Analyze only YouTube comments
        - 'facebook': Analyze only Facebook comments  
        - 'all': Analyze both sources (default)
        
        By default, aggregates by day to show sentiment trends over time. Each time period is 
        processed independently through the full pipeline. Returns time series data with 
        avg_sentiment, sentiment_std, and mention counts for each period.
        
        Use when user asks about entities not in the standard dataset.
        """
    ),
]

TOOL_FUNCTIONS: dict[str, Callable] = {
    "get_dataframe": get_dataframe,
    "create_dataframe": create_dataframe,
    "create_plotly_figure": create_plotly_figure,
    "analyze_custom_entity_sentiment": analyze_custom_entity_sentiment,
}
