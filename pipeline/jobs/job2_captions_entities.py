"""Job 2: Hot Topics Identification & Entity Detection."""

import sys

import pandas as pd

from pipeline.db import (
    checkpoint_exists,
    load_checkpoint_hot_topics,
    save_checkpoint_entities,
    save_checkpoint_hot_topics,
    save_token_usage,
)
from pipeline.logger import setup_logger
from pipeline.pipeline_config import PipelineConfig
from pipeline.stages.data_processing import process_latest_data
from pipeline.stages.detection import detect_entities_in_all_news_items
from pipeline.stages.topic_discovery import identify_hot_topics

logger = setup_logger("job2")


def main():
    """Execute hot topics identification and entity detection job.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    config = PipelineConfig()
    
    # Job 2 requires a pipeline run ID from Job 1
    if config.pipeline_run_id is None:
        logger.error("Job 2 requires a pipeline run ID from Job 1")
        logger.error("Usage: python job2_captions_entities.py --run-id <RUN_ID>")
        return 1
    
    pipeline_run_id = config.pipeline_run_id
    logger.info("="*60)
    logger.info(f"Job 2: Hot Topics & Entities - Run ID: {pipeline_run_id}")
    logger.info("="*60)
    
    token_stats = {
        'hot_topics_input': 0,
        'hot_topics_output': 0,
    }
    
    logger.info("Fetching latest data from database...")
    content_items = process_latest_data(
        pipeline_run_id=pipeline_run_id,
        collect_youtube=config.data_collection.collect_youtube,
        collect_facebook=config.data_collection.collect_facebook
    )
    
    # Verify Job 1 completed by checking if data was retrieved
    if content_items.empty:
        logger.error("Job 1 (Data Collection) has not completed yet or no data was collected")
        logger.error("Run Job 1 first to create the pipeline run and collect data")
        return 1
    
    logger.info(f"Retrieved {len(content_items)} unique content items")

    # ========================================================================
    # STAGE 1: HOT TOPICS IDENTIFICATION
    # ========================================================================
    df_topics = pd.DataFrame()
    
    if checkpoint_exists('hot_topics'):
        logger.info("="*60)
        logger.info("STAGE 1: HOT TOPICS [LOADING FROM CHECKPOINT]")
        logger.info("="*60)
        df_topics = load_checkpoint_hot_topics()
        num_topics = df_topics['topic'].nunique() if not df_topics.empty else 0
        logger.info(f"Loaded {num_topics} hot topics from checkpoint ({len(df_topics)} topic-content associations)")
        
        if not df_topics.empty:
            topic_counts = df_topics.groupby('topic').size()
            for topic, count in topic_counts.items():
                logger.info(f"  '{topic}': {count} content items")
    else:
        logger.info("="*60)
        logger.info("STAGE 1: HOT TOPICS IDENTIFICATION")
        logger.info("="*60)
        logger.info(f"Identifying top 8 hot topics using {config.caption.model}...")
        df_topics, usage = identify_hot_topics(
            content_items, 
            model=config.caption.model
        )
        
        if not df_topics.empty:
            token_stats['hot_topics_input'] = usage.get('input_tokens', 0)
            token_stats['hot_topics_output'] = usage.get('output_tokens', 0)
            
            num_topics = df_topics['topic'].nunique()
            logger.info(f"Identified {num_topics} hot topics (Tokens: {token_stats['hot_topics_input']} in, {token_stats['hot_topics_output']} out)")
            logger.info(f"Created {len(df_topics)} topic-content associations")
            
            logger.info("Saving hot topics checkpoint...")
            save_checkpoint_hot_topics(df_topics)
            logger.info("Hot topics checkpoint saved")
            
            save_token_usage(
                run_id=pipeline_run_id,
                stage_name='hot_topics',
                model_name=config.caption.model,
                input_tokens=token_stats['hot_topics_input'],
                output_tokens=token_stats['hot_topics_output']
            )
        else:
            logger.error("Failed to identify hot topics")
            return 1

    # ========================================================================
    # STAGE 2: ENTITY DETECTION
    # ========================================================================
    if checkpoint_exists('entities'):
        logger.info("="*60)
        logger.info("STAGE 2: ENTITY DETECTION [ALREADY COMPLETE]")
        logger.info("="*60)
        logger.info("Entity detection checkpoint already exists")
    else:
        logger.info("="*60)
        logger.info("STAGE 2: ENTITY DETECTION (Key Players & Key Issues)")
        logger.info("="*60)
        logger.info(f"Detecting entities in news items using {config.caption.embedding_model}...")

        df_entities = detect_entities_in_all_news_items(
            news_records=content_items,
            embedding_model=config.caption.embedding_model
        )
        logger.info(f"Finished entity detection in {len(df_entities)} news items")
        
        logger.info("Saving entity detection checkpoint...")
        save_checkpoint_entities(df_entities)
        logger.info("Entity checkpoint saved")

    logger.info("="*60)
    logger.info(f"Job 2 Complete! Pipeline Run ID: {pipeline_run_id}")
    num_topics = df_topics['topic'].nunique() if not df_topics.empty else 0
    logger.info(f"Hot Topics: {num_topics}, Tokens - Input: {token_stats['hot_topics_input']:,}, Output: {token_stats['hot_topics_output']:,}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
