"""BigQuery Database Initialization Script.

Creates tables with partitioning and clustering for optimal performance.
Only includes essential tables for the migration.
"""

import sys

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from config import BIGQUERY_DATASET
from database.bigquery_utils import get_bigquery_client, get_project_id

def create_dataset_if_not_exists(client: bigquery.Client, dataset_id: str) -> None:
    """Create BigQuery dataset if it doesn't exist.

    Args:
        client: BigQuery client instance.
        dataset_id: Fully qualified dataset ID.
    """
    try:
        client.get_dataset(dataset_id)
        print(f"✓ Dataset {dataset_id} already exists")
    except NotFound:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"✓ Created dataset {dataset_id}")

def create_tables(client: bigquery.Client, dataset_id: str, drop_existing: bool = False) -> None:
    """Create all required tables.

    Args:
        client: BigQuery client instance.
        dataset_id: Fully qualified dataset ID.
        drop_existing: Whether to drop existing tables before creating them.
    """
    print("=" * 60)
    print("Creating BigQuery Tables")
    print("=" * 60)

    if drop_existing:
        print("\n⚠️  Dropping existing tables...")
        tables_to_drop = [
            "youtube_videos",
            "youtube_comments",
            "facebook_posts",
            "facebook_comments",
            "pipeline_runs",
            "pipeline_comment_sentiments",
            "pipeline_entity_summaries",
            "pipeline_token_usage",
            "pipeline_checkpoints",
            "checkpoint_captions",
            "checkpoint_entities",
            "checkpoint_sentiments",
        ]
        for table_name in tables_to_drop:
            try:
                client.delete_table(f"{dataset_id}.{table_name}", not_found_ok=True)
                print(f"   ✓ Dropped {table_name}")
            except Exception as e:
                print(f"   ⚠️  Could not drop {table_name}: {e}")
        print("")
    
    # 1. YOUTUBE_VIDEOS - Partitioned by published_date, clustered by channel_name
    print("\n1. Creating youtube_videos table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.youtube_videos` (
        id INT64 NOT NULL,
        channel_id STRING NOT NULL,
        channel_name STRING,
        video_id STRING NOT NULL,
        video_url STRING NOT NULL,
        title STRING NOT NULL,
        published_date TIMESTAMP,
        comment_count INT64 DEFAULT 0,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP,
        pipeline_run_id INT64
    )
    PARTITION BY DATE(published_date)
    CLUSTER BY channel_name, scraped_at
    OPTIONS(
        description="YouTube videos with metadata"
    )
    """
    client.query(query).result()
    print("   ✓ youtube_videos created")
    
    # 2. YOUTUBE_COMMENTS - Partitioned by published_time, clustered by video_id
    print("\n2. Creating youtube_comments table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.youtube_comments` (
        external_comment_id STRING NOT NULL,
        video_id INT64 NOT NULL,
        author_id STRING,
        comment_text STRING NOT NULL,
        votes INT64 DEFAULT 0,
        published_time TIMESTAMP,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        is_reply BOOL DEFAULT FALSE,
        parent_comment_id STRING
    )
    PARTITION BY DATE(published_time)
    CLUSTER BY video_id, is_reply
    OPTIONS(
        description="Comments on YouTube videos - external_comment_id is YouTube's comment ID, parent_comment_id references parent's external_comment_id, author_id is anonymized hash"
    )
    """
    client.query(query).result()
    print("   ✓ youtube_comments created")
    
    # 3. FACEBOOK_POSTS - Partitioned by created_date, clustered by page_name
    print("\n3. Creating facebook_posts table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.facebook_posts` (
        id INT64 NOT NULL,
        page_id STRING NOT NULL,
        page_name STRING,
        post_id STRING NOT NULL,
        message STRING,
        post_url STRING,
        created_date TIMESTAMP,
        comment_count INT64 DEFAULT 0,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        updated_at TIMESTAMP,
        pipeline_run_id INT64
    )
    PARTITION BY DATE(created_date)
    CLUSTER BY page_name, scraped_at
    OPTIONS(
        description="Facebook posts from public pages with metadata"
    )
    """
    client.query(query).result()
    print("   ✓ facebook_posts created")
    
    # 4. FACEBOOK_COMMENTS - Partitioned by published_time, clustered by post_id
    print("\n4. Creating facebook_comments table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.facebook_comments` (
        external_comment_id STRING NOT NULL,
        post_id INT64 NOT NULL,
        author_id STRING,
        comment_text STRING NOT NULL,
        votes INT64 DEFAULT 0,
        published_time TIMESTAMP,
        scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        is_reply BOOL DEFAULT FALSE,
        parent_comment_id STRING
    )
    PARTITION BY DATE(published_time)
    CLUSTER BY post_id, is_reply
    OPTIONS(
        description="Comments on Facebook posts - external_comment_id is Facebook's comment ID, parent_comment_id references parent's external_comment_id, author_id is anonymized hash"
    )
    """
    client.query(query).result()
    print("   ✓ facebook_comments created")
    
    # 5. PIPELINE_RUNS - Partitioned by started_at date, clustered by status
    print("\n5. Creating pipeline_runs table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.pipeline_runs` (
        id INT64 NOT NULL,
        completed_at TIMESTAMP,
        caption_model STRING,
        embedding_model STRING,
        sentiments_model STRING,
        summary_model STRING,
        similarity_threshold FLOAT64,
        max_workers INT64,
        status STRING DEFAULT 'running',
        caption_input_tokens INT64 DEFAULT 0,
        caption_output_tokens INT64 DEFAULT 0,
        sentiments_input_tokens INT64 DEFAULT 0,
        sentiments_output_tokens INT64 DEFAULT 0,
        summary_input_tokens INT64 DEFAULT 0,
        summary_output_tokens INT64 DEFAULT 0,
        total_input_tokens INT64 DEFAULT 0,
        total_output_tokens INT64 DEFAULT 0,
        notes STRING,
        config_json STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    PARTITION BY DATE(completed_at)
    CLUSTER BY status, id
    OPTIONS(
        description="Pipeline execution tracking and token usage"
    )
    """
    client.query(query).result()
    print("   ✓ pipeline_runs created")
    
    # 6. PIPELINE_COMMENT_SENTIMENTS - Partitioned by created_at, clustered by entity_name
    print("\n6. Creating pipeline_comment_sentiments table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.pipeline_comment_sentiments` (
        id INT64 NOT NULL,
        run_id INT64 NOT NULL,
        content_type STRING NOT NULL,
        content_id STRING NOT NULL,
        author_id STRING NOT NULL,
        entity_name STRING NOT NULL,
        sentiment_score FLOAT64,
        smoothed_sentiment_score FLOAT64,
        normalized_sentiment_score FLOAT64,
        model_used STRING,
        source_name STRING,
        is_prod BOOL DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY entity_name, run_id, content_type
    OPTIONS(
        description="Per-comment entity sentiment scores - includes original, smoothed (KNN), and normalized (channel ECDF) scores"
    )
    """
    client.query(query).result()
    print("   ✓ pipeline_comment_sentiments created")
    
    # 7. PIPELINE_ENTITY_SUMMARIES - Partitioned by created_at, clustered by entity_name
    print("\n7. Creating pipeline_entity_summaries table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.pipeline_entity_summaries` (
        id INT64 NOT NULL,
        run_id INT64 NOT NULL,
        entity_name STRING NOT NULL,
        avg_sentiment FLOAT64,
        sentiment_count INT64,
        sentiment_std FLOAT64,
        content_count INT64,
        sentiment_summary STRING,
        model_used STRING,
        is_prod BOOL DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    PARTITION BY DATE(created_at)
    CLUSTER BY entity_name, run_id
    OPTIONS(
        description="Aggregated entity sentiment summaries - used by Streamlit"
    )
    """
    client.query(query).result()
    print("   ✓ pipeline_entity_summaries created")
    
    # 8. PIPELINE_CHECKPOINTS - Track checkpoint status for recovery
    print("\n8. Creating pipeline_checkpoints table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.pipeline_checkpoints` (
        id INT64 NOT NULL,
        stage_name STRING NOT NULL,
        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
        row_count INT64,
        metadata_json STRING
    )
    CLUSTER BY stage_name
    OPTIONS(
        description="Checkpoint metadata for pipeline recovery - tables cleared at start of each run"
    )
    """
    client.query(query).result()
    print("   ✓ pipeline_checkpoints created")
    
    # 9. PIPELINE_TOKEN_USAGE - Track token usage per stage per run
    print("\n9. Creating pipeline_token_usage table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.pipeline_token_usage` (
        id INT64 NOT NULL,
        run_id INT64 NOT NULL,
        stage_name STRING NOT NULL,
        model_name STRING,
        input_tokens INT64 DEFAULT 0,
        output_tokens INT64 DEFAULT 0,
        total_tokens INT64 DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    CLUSTER BY run_id, stage_name
    OPTIONS(
        description="Token usage tracking per pipeline stage - permanent historical data"
    )
    """
    client.query(query).result()
    print("   ✓ pipeline_token_usage created")
    
    # 10. CHECKPOINT_CAPTIONS - Store caption stage results
    print("\n10. Creating checkpoint_captions table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.checkpoint_captions` (
        id INT64 NOT NULL,
        article_title STRING,
        caption_text STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    OPTIONS(
        description="Checkpoint: Caption generation results - cleared at start of each run"
    )
    """
    client.query(query).result()
    print("   ✓ checkpoint_captions created")
    
    # 11. CHECKPOINT_ENTITIES - Store entity detection results
    print("\n11. Creating checkpoint_entities table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.checkpoint_entities` (
        id INT64 NOT NULL,
        content_id STRING NOT NULL,
        news_title STRING,
        source_type STRING NOT NULL,
        source_name STRING,
        entity_name STRING NOT NULL,
        detection_method STRING,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
    )
    CLUSTER BY content_id
    OPTIONS(
        description="Checkpoint: Entity detection results - cleared at start of each run"
    )
    """
    client.query(query).result()
    print("   ✓ checkpoint_entities created")
    
    # 12. CHECKPOINT_SENTIMENTS - Store sentiment analysis results
    print("\n12. Creating checkpoint_sentiments table...")
    query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.checkpoint_sentiments` (
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
    OPTIONS(
        description="Checkpoint: Sentiment analysis results - cleared at start of each run"
    )
    """
    client.query(query).result()
    print("   ✓ checkpoint_sentiments created")

    print("\n" + "=" * 60)
    print("✓ All tables created successfully!")
    print("=" * 60)


def init_bigquery(drop_existing: bool = False) -> None:
    """Initialize BigQuery database with all required tables.

    Args:
        drop_existing: Whether to drop existing tables before creating them.
    """
    print("\n" + "=" * 60)
    print("GPSO BigQuery Database Initialization")
    print("=" * 60)

    # Get client and project info
    client = get_bigquery_client()
    project_id = get_project_id()
    dataset_id = f"{project_id}.{BIGQUERY_DATASET}"

    print(f"\nProject: {project_id}")
    print(f"Dataset: {BIGQUERY_DATASET}")
    print(f"Full dataset ID: {dataset_id}")

    # Create dataset if needed
    create_dataset_if_not_exists(client, dataset_id)

    # Create all tables
    create_tables(client, dataset_id, drop_existing=drop_existing)

    print("\n✓ BigQuery initialization complete!")
    print(f"✓ Dataset: {dataset_id}")
    print()


if __name__ == '__main__':
    drop_existing = '--drop' in sys.argv
    init_bigquery(drop_existing=drop_existing)
