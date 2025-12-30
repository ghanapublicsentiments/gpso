"""Database persistence helpers for pipeline run, sentiments and summaries."""

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from database.bigquery_manager import get_bigquery_manager
from pipeline.pipeline_config import PipelineConfig


def init_pipeline_run() -> int:
    """Generate a new pipeline run ID for tracking checkpoints.

    No database record is created yet - the run will only be saved
    to BigQuery when it completes successfully.

    Returns:
        int: The pipeline run ID.
    """
    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{manager.dataset_id}.pipeline_runs`"
    result = manager.client.query(query).result()
    run_id = list(result)[0].next_id

    return run_id


def complete_pipeline_run(run_id: int, token_stats: dict[str, int], status: str = 'completed') -> None:
    """Save a completed pipeline run to BigQuery with full configuration and statistics.

    This is the ONLY time we write to the pipeline_runs table - we only save
    complete runs, not incomplete/failed ones.

    Args:
        run_id: Pipeline run ID.
        token_stats: Dictionary with token usage statistics.
        status: 'completed' or 'failed'.
    """
    manager = get_bigquery_manager()
    config = PipelineConfig()

    now = datetime.now()

    caption_model = config.caption.model
    embedding_model = config.smoothing.embedding_model
    sentiments_model = config.sentiment.model
    summary_model = config.summary.model
    max_workers = config.sentiment.max_workers
    
    # Create config dict and remove fields that are stored separately to avoid duplication
    config_dict = config.to_dict()
    if 'caption' in config_dict and 'model' in config_dict['caption']:
        del config_dict['caption']['model']
    if 'smoothing' in config_dict and 'embedding_model' in config_dict['smoothing']:
        del config_dict['smoothing']['embedding_model']
    if 'sentiment' in config_dict:
        if 'model' in config_dict['sentiment']:
            del config_dict['sentiment']['model']
        if 'max_workers' in config_dict['sentiment']:
            del config_dict['sentiment']['max_workers']
    if 'summary' in config_dict and 'model' in config_dict['summary']:
        del config_dict['summary']['model']

    run_data = {
        'id': run_id,
        'completed_at': now.isoformat(),
        'status': status,
        
        # Model configuration (stored separately for easy querying)
        'caption_model': caption_model,
        'embedding_model': embedding_model,
        'sentiments_model': sentiments_model,
        'summary_model': summary_model,
        'max_workers': max_workers,
        
        # Token statistics
        'caption_input_tokens': token_stats.get('caption_input', 0),
        'caption_output_tokens': token_stats.get('caption_output', 0),
        'sentiments_input_tokens': token_stats.get('sentiments_input', 0),
        'sentiments_output_tokens': token_stats.get('sentiments_output', 0),
        'summary_input_tokens': token_stats.get('summary_input', 0),
        'summary_output_tokens': token_stats.get('summary_output', 0),
        'total_input_tokens': token_stats.get('total_input', 0),
        'total_output_tokens': token_stats.get('total_output', 0),
        'config_json': json.dumps(config_dict)
    }

    manager._insert_rows('pipeline_runs', [run_data])
    print(f"✓ Pipeline run {run_id} saved to BigQuery with status: {status}")


def persist_comment_sentiments(run_id: int, df: Any, model_used: str) -> int:
    """Persist comment sentiments to BigQuery.

    Args:
        run_id: Pipeline run ID.
        df: DataFrame with comment sentiments (must include: content_type, content_id,
            author_id, entity_name, sentiment_score, smoothed_sentiment_score,
            normalized_sentiment_score, source_name).
        model_used: Model name used for sentiment analysis.

    Returns:
        int: Number of rows inserted.
    """
    if df.empty:
        return 0

    manager = get_bigquery_manager()

    sentiments = [{
        'content_type': r.get('source_type') or r.get('content_type'),
        'content_id': r['content_id'],
        'author_id': r['author_id'],
        'entity_name': r['entity_name'],
        'sentiment_score': r.get('sentiment') or r.get('sentiment_score'),
        'smoothed_sentiment_score': r.get('smoothed_sentiment'),
        'normalized_sentiment_score': r.get('normalized_sentiment'),
        'source_name': r.get('source_name')
    } for r in df.to_dict('records')]

    return manager.insert_comment_sentiments(
        run_id=run_id,
        sentiments=sentiments,
        model_used=model_used
    )


def persist_entity_summaries(run_id: int, df: Any, model_used: str) -> int:
    """Persist entity summaries to BigQuery.

    Args:
        run_id: Pipeline run ID.
        df: DataFrame with entity summaries.
        model_used: Model name used for summary generation.

    Returns:
        int: Number of rows inserted.
    """
    if df.empty:
        return 0

    manager = get_bigquery_manager()

    summaries = [{
        'entity_name': r['entity_name'],
        'avg_sentiment': r.get('avg_sentiment'),
        'sentiment_count': r.get('sentiment_count'),
        'sentiment_std': r.get('sentiment_std'),
        'content_count': r.get('content_count'),
        'sentiment_summary': r.get('sentiment_summary')
    } for r in df.to_dict('records')]

    return manager.insert_entity_summaries(
        run_id=run_id,
        summaries=summaries,
        model_used=model_used
    )

def summaries_exist(run_id: int) -> bool:
    """Check if entity summaries exist for a given pipeline run.

    Args:
        run_id: Pipeline run ID.

    Returns:
        bool: True if summaries exist, False otherwise.
    """
    manager = get_bigquery_manager()
    query = f"""
    SELECT COUNT(*) as count
    FROM `{manager.dataset_id}.pipeline_entity_summaries`
    WHERE run_id = {run_id}
    """
    result = list(manager.client.query(query).result())
    return result[0].count > 0


def save_token_usage(run_id: int, stage_name: str, model_name: str, input_tokens: int, output_tokens: int) -> None:
    """Save token usage for a pipeline stage.

    Args:
        run_id: Pipeline run ID.
        stage_name: Name of the pipeline stage (e.g., 'hot_topics', 'sentiments', 'summaries').
        model_name: Name of the model used.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens used.
    """
    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{manager.dataset_id}.pipeline_token_usage`"
    result = list(manager.client.query(query).result())
    next_id = result[0].next_id

    row = {
        'id': next_id,
        'run_id': run_id,
        'stage_name': stage_name,
        'model_name': model_name,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
        'created_at': datetime.now(timezone.utc).isoformat()
    }

    manager._insert_rows('pipeline_token_usage', [row])
    print(f"   ✓ Token usage saved: {stage_name} ({input_tokens:,} in, {output_tokens:,} out)")


def get_pipeline_token_stats(run_id: int) -> dict[str, int]:
    """Aggregate token statistics from pipeline_token_usage table for a pipeline run.

    Args:
        run_id: Pipeline run ID.

    Returns:
        dict[str, int]: Dictionary with aggregated token statistics.
    """
    manager = get_bigquery_manager()

    token_stats = {
        'caption_input': 0,
        'caption_output': 0,
        'sentiments_input': 0,
        'sentiments_output': 0,
        'summary_input': 0,
        'summary_output': 0,
        'total_input': 0,
        'total_output': 0,
    }

    query = f"""
    SELECT stage_name, SUM(input_tokens) as input_tokens, SUM(output_tokens) as output_tokens
    FROM `{manager.dataset_id}.pipeline_token_usage`
    WHERE run_id = {run_id}
    GROUP BY stage_name
    """

    try:
        results = manager.client.query(query).result()
        for row in results:
            stage = row.stage_name
            input_tok = row.input_tokens or 0
            output_tok = row.output_tokens or 0

            if stage == 'hot_topics':
                token_stats['caption_input'] = input_tok
                token_stats['caption_output'] = output_tok
            elif stage == 'sentiments':
                token_stats['sentiments_input'] = input_tok
                token_stats['sentiments_output'] = output_tok
            elif stage == 'summaries':
                token_stats['summary_input'] = input_tok
                token_stats['summary_output'] = output_tok
    except Exception as e:
        print(f"   ⚠️  Could not load token usage: {e}")

    token_stats['total_input'] = (
        token_stats['caption_input'] +
        token_stats['sentiments_input'] +
        token_stats['summary_input']
    )
    token_stats['total_output'] = (
        token_stats['caption_output'] +
        token_stats['sentiments_output'] +
        token_stats['summary_output']
    )

    return token_stats


def checkpoint_exists(stage_name: str) -> bool:
    """Check if a checkpoint exists for a given stage.

    Args:
        stage_name: Stage name ('captions', 'hot_topics', 'entities', 'sentiments', 'sentiments_persisted').

    Returns:
        bool: True if checkpoint exists, False otherwise.
    """
    manager = get_bigquery_manager()
    query = f"""
    SELECT COUNT(*) as count
    FROM `{manager.dataset_id}.pipeline_checkpoints`
    WHERE stage_name = '{stage_name}'
    """
    result = list(manager.client.query(query).result())
    return result[0].count > 0


def save_checkpoint_hot_topics(df_topics: pd.DataFrame) -> int:
    """Save hot topics identification results as a checkpoint.

    Each topic-content_id pair is stored as a separate row.

    Args:
        df_topics: DataFrame with columns: content_id, topic

    Returns:
        int: Number of rows inserted.
    """
    if df_topics.empty:
        return 0

    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) as max_id FROM `{manager.dataset_id}.checkpoint_hot_topics`"
    result = list(manager.client.query(query).result())
    next_id = result[0].max_id + 1

    rows = []
    for idx, row in df_topics.iterrows():
        rows.append({
            'id': next_id + idx,
            'topic': row['topic'],
            'content_id': row['content_id']
        })

    if not rows:
        print(f"   ⚠️  No valid hot topics to save")
        return 0

    manager._insert_rows('checkpoint_hot_topics', rows)
    _record_checkpoint('hot_topics', len(rows))

    return len(rows)


def save_checkpoint_entities(df_entities: Any) -> int:
    """Save entity detection results as a checkpoint.

    The input DataFrame has lists of entities per content item:
    - detected_key_players (list)
    - detected_key_issues (list)

    This function explodes these into individual entity rows.

    Args:
        df_entities: DataFrame with entity detection results.

    Returns:
        int: Number of rows inserted.
    """
    if df_entities.empty:
        return 0

    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) as max_id FROM `{manager.dataset_id}.checkpoint_entities`"
    result = list(manager.client.query(query).result())
    next_id = result[0].max_id + 1

    rows = []
    row_idx = 0

    for _, row_data in df_entities.iterrows():
        content_id = row_data.get('content_id')

        for player in row_data.get('detected_key_players', []):
            if player and isinstance(player, str) and player.strip():
                rows.append({
                    'id': next_id + row_idx,
                    'content_id': content_id,
                    'entity_name': player,
                    'detection_method': 'key_player'
                })
                row_idx += 1

        for issue in row_data.get('detected_key_issues', []):
            if issue and isinstance(issue, str) and issue.strip():
                rows.append({
                    'id': next_id + row_idx,
                    'content_id': content_id,
                    'entity_name': issue,
                    'detection_method': 'key_issue'
                })
                row_idx += 1

    if not rows:
        print(f"   ⚠️  No valid entities to save (all lists were empty)")
        return 0

    manager._insert_rows('checkpoint_entities', rows)
    _record_checkpoint('entities', len(rows))

    return len(rows)


def save_checkpoint_sentiments(df_sentiments: Any) -> int:
    """Save sentiment analysis results as a checkpoint (before smoothing/normalization).

    Args:
        df_sentiments: DataFrame with sentiment analysis results.

    Returns:
        int: Number of rows inserted.
    """
    if df_sentiments.empty:
        return 0

    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) as max_id FROM `{manager.dataset_id}.checkpoint_sentiments`"
    result = list(manager.client.query(query).result())
    next_id = result[0].max_id + 1

    rows = []
    skipped = 0

    for _, row in enumerate(df_sentiments.to_dict('records')):
        entity_name = row.get('entity_name')
        author_id = row.get('author_id')
        sentiment = row.get('sentiment')

        if (
            entity_name is None or (isinstance(entity_name, str) and not entity_name.strip()) or
            author_id is None or (isinstance(author_id, str) and not author_id.strip()) or
            sentiment is None or (isinstance(sentiment, float) and (np.isnan(sentiment) or np.isinf(sentiment)))
        ):
            skipped += 1
            continue

        rows.append({
            'id': next_id + len(rows),
            'content_id': row.get('content_id'),
            'source_type': row.get('source_type'),
            'source_name': row.get('source_name'),
            'author_id': author_id,
            'entity_name': entity_name,
            'sentiment_score': float(sentiment),
            'comment_text': row.get('comment_text', '')
        })

    if not rows:
        print(f"   ⚠️  No valid sentiments to save (all {skipped} rows had NULL/NaN required fields)")
        return 0

    if skipped > 0:
        print(f"   ℹ️  Filtered out {skipped} rows with NULL/NaN values, saving {len(rows)} valid sentiments")

    manager._insert_rows('checkpoint_sentiments', rows)
    _record_checkpoint('sentiments', len(rows))

    return len(rows)


def load_checkpoint_hot_topics() -> pd.DataFrame:
    """Load hot topics from checkpoint as a DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns: content_id, topic (empty DataFrame if no checkpoint exists).
    """
    manager = get_bigquery_manager()

    query = f"""
    SELECT content_id, topic
    FROM `{manager.dataset_id}.checkpoint_hot_topics`
    ORDER BY id
    """

    df = manager.client.query(query).to_dataframe()

    if df.empty:
        return pd.DataFrame(columns=['content_id', 'topic'])

    return df


def load_checkpoint_entities() -> Any:
    """Load entity detection checkpoint data and reconstruct the original DataFrame format.

    Converts individual entity rows back into the format with lists:
    - detected_key_players (list)
    - detected_key_issues (list)

    Returns:
        pd.DataFrame: DataFrame with entity detection results in original format.
    """
    manager = get_bigquery_manager()

    query = f"""
    SELECT content_id, entity_name, detection_method
    FROM `{manager.dataset_id}.checkpoint_entities`
    """
    df_flat = manager.client.query(query).to_dataframe()

    if df_flat.empty:
        return pd.DataFrame(
            columns=[
                'content_id',
                'detected_key_players',
                'detected_key_issues'
            ]
        )

    # Group by content_id and reconstruct lists
    results = []
    for content_id, group in df_flat.groupby('content_id'):
        key_players = group[group['detection_method'] == 'key_player']['entity_name'].tolist()
        key_issues = group[group['detection_method'] == 'key_issue']['entity_name'].tolist()

        results.append({
            'content_id': content_id,
            'detected_key_players': key_players,
            'detected_key_issues': key_issues
        })

    return pd.DataFrame(results)


def load_checkpoint_sentiments() -> Any:
    """Load sentiment analysis checkpoint data.

    Returns:
        pd.DataFrame: DataFrame with sentiment analysis results (pre-smoothing).
    """
    manager = get_bigquery_manager()
    query = f"""
    SELECT content_id, source_type, source_name, author_id, entity_name,
           sentiment_score as sentiment, comment_text
    FROM `{manager.dataset_id}.checkpoint_sentiments`
    """
    return manager.client.query(query).to_dataframe()


def clear_checkpoints() -> None:
    """Clear all checkpoints by dropping and recreating checkpoint tables.

    Called at the start of every pipeline run to ensure clean state.
    """
    manager = get_bigquery_manager()
    client = manager.client
    dataset_id = manager.dataset_id

    tables = ['checkpoint_hot_topics', 'checkpoint_entities', 'checkpoint_sentiments', 'pipeline_checkpoints']
    for table in tables:
        try:
            client.query(f"DROP TABLE IF EXISTS `{dataset_id}.{table}`").result()
        except:
            pass

    queries = {
        'pipeline_checkpoints': f"""
        CREATE TABLE `{dataset_id}.pipeline_checkpoints` (
            id INT64 NOT NULL,
            stage_name STRING NOT NULL,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
            row_count INT64,
            metadata_json STRING
        )
        CLUSTER BY stage_name
        """,
        'checkpoint_hot_topics': f"""
        CREATE TABLE `{dataset_id}.checkpoint_hot_topics` (
            id INT64 NOT NULL,
            topic STRING NOT NULL,
            content_id STRING NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        )
        CLUSTER BY topic
        """,
        'checkpoint_entities': f"""
        CREATE TABLE `{dataset_id}.checkpoint_entities` (
            id INT64 NOT NULL,
            content_id STRING NOT NULL,
            entity_name STRING NOT NULL,
            detection_method STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        )
        CLUSTER BY content_id
        """,
        'checkpoint_sentiments': f"""
        CREATE TABLE `{dataset_id}.checkpoint_sentiments` (
            id INT64 NOT NULL,
            content_id STRING NOT NULL,
            source_type STRING NOT NULL,
            source_name STRING,
            author_id STRING NOT NULL,
            entity_name STRING NOT NULL,
            sentiment_score FLOAT64 NOT NULL,
            comment_text STRING,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        )
        CLUSTER BY entity_name
        """
    }

    for table, query in queries.items():
        client.query(query).result()


def _record_checkpoint(stage_name: str, row_count: int) -> None:
    """Internal helper to record checkpoint metadata.

    Args:
        stage_name: Stage name.
        row_count: Number of rows saved.
    """
    manager = get_bigquery_manager()

    query = f"SELECT COALESCE(MAX(id), 0) as max_id FROM `{manager.dataset_id}.pipeline_checkpoints`"
    result = list(manager.client.query(query).result())
    next_id = result[0].max_id + 1

    row = {
        'id': next_id,
        'stage_name': stage_name,
        'row_count': row_count,
        'metadata_json': None
    }

    manager._insert_rows('pipeline_checkpoints', [row])
    print(f"   ✓ Checkpoint saved: {stage_name} ({row_count} rows)")
