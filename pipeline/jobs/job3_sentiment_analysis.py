"""Job 3: Sentiment Analysis - The most expensive stage."""

import sys
import time

from pipeline.db import (
    checkpoint_exists,
    load_checkpoint_entities,
    load_checkpoint_hot_topics,
    save_checkpoint_sentiments,
    save_token_usage,
)
from pipeline.logger import setup_logger
from pipeline.pipeline_config import PipelineConfig
from pipeline.stages.data_processing import process_latest_data
from pipeline.stages.sentiments import analyze_sentiments

logger = setup_logger("job3")


def main():
    """Execute sentiment analysis job.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    config = PipelineConfig()
    
    # Job 3 requires a pipeline run ID from Job 1
    if config.pipeline_run_id is None:
        logger.error("Job 3 requires a pipeline run ID from Job 1")
        logger.error("Usage: python job3_sentiment_analysis.py --run-id <RUN_ID>")
        return 1
    
    pipeline_run_id = config.pipeline_run_id
    logger.info("="*60)
    logger.info(f"Job 3: Sentiment Analysis - Run ID: {pipeline_run_id}")
    logger.info("="*60)
    
    # Verify Job 2 completed
    if not checkpoint_exists('entities'):
        logger.error("Job 2 (Captions & Entities) has not completed yet")
        logger.error("Run Job 2 first to generate captions and detect entities")
        return 1
    
    if not checkpoint_exists('hot_topics'):
        logger.error("Job 2 (Hot Topics) has not completed yet")
        logger.error("Run Job 2 first to identify hot topics")
        return 1
    
    token_stats = {
        'sentiments_input': 0,
        'sentiments_output': 0,
    }
    
    logger.info("Fetching latest data from database...")
    content_items = process_latest_data(pipeline_run_id=pipeline_run_id)
    logger.info(f"Retrieved {len(content_items)} unique content items")
    
    logger.info("Loading entity detection results from checkpoint...")
    df_entities = load_checkpoint_entities()
    logger.info(f"Loaded {len(df_entities)} entity detections (key players & issues) from checkpoint")
    
    logger.info("Loading hot topics mapping from checkpoint...")
    df_topics = load_checkpoint_hot_topics()
    logger.info(f"Loaded {len(df_topics)} topic-content associations from checkpoint")
    logger.info(f"Total unique topics: {df_topics['topic'].nunique()}")

    # ========================================================================
    # STAGE 3: SENTIMENT ANALYSIS (MOST EXPENSIVE!)
    # ========================================================================
    logger.info("="*60)
    logger.info("STAGE 3: SENTIMENT ANALYSIS")
    logger.info("="*60)
    logger.info(f"Analyzing sentiments using {config.sentiment.model} (workers: {config.sentiment.max_workers}, rate: {config.sentiment.rate_limit}/min)...")

    df_sent, sent_usage = analyze_sentiments(
        df_entities,
        df_topics,
        content_items,
        config.sentiment.model, 
        max_workers=config.sentiment.max_workers, 
        rate_limit=config.sentiment.rate_limit
    )
    token_stats['sentiments_input'] = sent_usage.get('input_tokens', 0)
    token_stats['sentiments_output'] = sent_usage.get('output_tokens', 0)
    logger.info(f"Analyzed {len(df_sent)} sentiment records (Tokens: {token_stats['sentiments_input']} in, {token_stats['sentiments_output']} out)")
    
    # Save checkpoint for job4 to use
    logger.info("Saving sentiment analysis checkpoint...")
    save_checkpoint_sentiments(df_sent)
    logger.info("Sentiment checkpoint saved")
    
    save_token_usage(
        run_id=pipeline_run_id,
        stage_name='sentiments',
        model_name=config.sentiment.model,
        input_tokens=token_stats['sentiments_input'],
        output_tokens=token_stats['sentiments_output']
    )
    
    logger.info("Sleeping 60s to respect rate limits...")
    time.sleep(60)

    logger.info("="*60)
    logger.info(f"Job 3 Complete! Pipeline Run ID: {pipeline_run_id}")
    logger.info(f"Sentiment records: {len(df_sent)}, Tokens - Input: {token_stats['sentiments_input']:,}, Output: {token_stats['sentiments_output']:,}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
