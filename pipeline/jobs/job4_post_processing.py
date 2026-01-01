"""Job 4: Post-processing - Smoothing, normalization, summaries, and persistence."""

import sys

from pipeline.db import (
    checkpoint_exists,
    load_checkpoint_sentiments,
    persist_comment_sentiments,
    persist_entity_summaries,
    save_token_usage,
)
from pipeline.logger import setup_logger
from pipeline.pipeline_config import PipelineConfig
from pipeline.stages.data_processing import process_latest_data
from pipeline.stages.normalization import normalize_channels
from pipeline.stages.smoothing import smooth_sentiments
from pipeline.stages.summaries import generate_summaries

logger = setup_logger("job4")


def main():
    """Execute finalization job.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    config = PipelineConfig()
    
    # Job 4 requires a pipeline run ID from Job 1
    if config.pipeline_run_id is None:
        logger.error("Job 4 requires a pipeline run ID from Job 1")
        logger.error("Usage: python job4_finalization.py --run-id <RUN_ID>")
        return 1
    
    pipeline_run_id = config.pipeline_run_id
    logger.info("="*60)
    logger.info(f"Job 4: Finalization - Run ID: {pipeline_run_id}")
    logger.info("="*60)
    
    # Verify Job 3 completed
    if not checkpoint_exists('sentiments'):
        logger.error("Job 3 (Sentiment Analysis) has not completed yet")
        logger.error("Run Job 3 first to analyze sentiments")
        return 1
    
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
    
    logger.info("Fetching latest data from database...")
    content_items = process_latest_data(
        pipeline_run_id=pipeline_run_id,
        collect_youtube=config.data_collection.collect_youtube,
        collect_facebook=config.data_collection.collect_facebook
    )
    logger.info(f"Retrieved {len(content_items)} unique content items")
    
    # Load sentiments from checkpoint
    logger.info("Loading sentiment analysis results from checkpoint...")
    df_sent = load_checkpoint_sentiments()
    logger.info(f"Loaded {len(df_sent)} sentiment records from checkpoint")
    
    # ========================================================================
    # SMOOTHING
    # ========================================================================
    logger.info("="*60)
    logger.info("SMOOTHING")
    logger.info("="*60)
    logger.info(f"Smoothing sentiments using KNN (k={config.smoothing.k_neighbors}, model={config.smoothing.embedding_model})...")
    df_sent, smooth_stats = smooth_sentiments(
        df_sent, 
        embedding_model=config.smoothing.embedding_model,
        k_neighbors=config.smoothing.k_neighbors,
        min_neighbors=config.smoothing.min_neighbors
    )
    logger.info(f"Smoothed {smooth_stats['valid_sentiments']} sentiment records")
    
    # ========================================================================
    # NORMALIZATION
    # ========================================================================
    logger.info("="*60)
    logger.info("NORMALIZATION")
    logger.info("="*60)
    logger.info("Normalizing sentiments per channel (source_name) using historical ECDFs (BigQuery)...")
    df_sent, norm_stats = normalize_channels(
        df_sent, 
        window_days=config.normalization.window_days, 
        normalization_strength=config.normalization.normalization_strength
    )
    logger.info(f"Channel-normalized sentiments - rows: {norm_stats['total_rows']}, normalized: {norm_stats['rows_normalized']}, channels: {norm_stats['channels_seen']}")
    
    logger.info("Persisting smoothed & normalized comment sentiments to BigQuery...")
    inserted_sent = persist_comment_sentiments(pipeline_run_id, df_sent, config.sentiment.model)
    logger.info(f"Inserted {inserted_sent} comment sentiment rows")
    
    # ========================================================================
    # SUMMARIES
    # ========================================================================
    logger.info("="*60)
    logger.info("SUMMARIES")
    logger.info("="*60)
    logger.info(f"Generating entity summaries using {config.summary.model} (workers: {config.summary.max_workers}, rate: {config.summary.rate_limit}/min)...")
    df_sum, sum_usage = generate_summaries(
        df_sent, 
        content_items, 
        config.summary.model, 
        max_workers=config.summary.max_workers, 
        rate_limit=config.summary.rate_limit
    )
    token_stats['summary_input'] = sum_usage.get('input_tokens', 0)
    token_stats['summary_output'] = sum_usage.get('output_tokens', 0)
    token_stats['total_input'] += token_stats['summary_input']
    token_stats['total_output'] += token_stats['summary_output']
    logger.info(f"Generated {len(df_sum)} entity summaries (Tokens: {token_stats['summary_input']} in, {token_stats['summary_output']} out)")
    
    logger.info("Persisting summary data to BigQuery...")
    inserted_sum = persist_entity_summaries(pipeline_run_id, df_sum, config.summary.model)
    logger.info(f"Inserted {inserted_sum} summary rows")
    
    save_token_usage(
        run_id=pipeline_run_id,
        stage_name='summaries',
        model_name=config.summary.model,
        input_tokens=token_stats['summary_input'],
        output_tokens=token_stats['summary_output']
    )
    
    logger.info("="*60)
    logger.info(f"Job 4 Complete! Pipeline Run ID: {pipeline_run_id}")
    logger.info(f"Sentiment rows: {inserted_sent}, Summary rows: {inserted_sum}")
    logger.info(f"Summary tokens - Input: {token_stats['summary_input']:,}, Output: {token_stats['summary_output']:,}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
