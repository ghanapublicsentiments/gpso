"""Job 1: Data Collection and data processing."""

import sys

from pipeline.logger import setup_logger
from pipeline.pipeline_config import PipelineConfig
from pipeline.stages.data_processing import process_latest_data
from sources.facebook import save_facebook_to_database
from sources.youtube import save_youtube_to_database

logger = setup_logger("job1")


def main():
    """Execute data collection job.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    config = PipelineConfig()
    
    # Job 1 requires a pipeline run ID from Job 0
    if config.pipeline_run_id is None:
        logger.error("Job 1 requires a pipeline run ID from Job 0")
        logger.error("Usage: python job1_data_collection.py --run-id <RUN_ID>")
        return 1
    
    pipeline_run_id = config.pipeline_run_id
    logger.info("="*60)
    logger.info(f"Job 1: Data Collection - Run ID: {pipeline_run_id}")
    logger.info("="*60)

    youtube_stats = None
    facebook_stats = None

    # Get YouTube videos using YouTube Data API
    if config.data_collection.collect_youtube:
        logger.info("Collecting YouTube videos...")
        try:
            youtube_stats = save_youtube_to_database(
                pipeline_run_id=pipeline_run_id,
                max_videos_per_channel=config.data_collection.max_videos_per_channel,
                min_comments=config.data_collection.min_youtube_comments,
                max_comments_per_video=config.data_collection.max_comments_per_video,
            )
            logger.info(f"YouTube collection complete: {youtube_stats['videos_inserted']} videos, {youtube_stats['regular_comments']} comments")
        except Exception as e:
            logger.error(f"Error collecting YouTube: {e}")
    else:
        logger.info("YouTube collection disabled in configuration")
    
    # Get Facebook posts using Facebook Graph API
    if config.data_collection.collect_facebook:
        logger.info("Collecting Facebook posts...")
        try:
            facebook_stats = save_facebook_to_database(
                pipeline_run_id=pipeline_run_id,
                max_posts_per_page=config.data_collection.max_posts_per_page,
                min_comments=config.data_collection.min_facebook_comments,
                max_comments_per_post=config.data_collection.max_comments_per_post,
            )
            logger.info(f"Facebook collection complete: {facebook_stats['posts_inserted']} posts, {facebook_stats['comments']} comments")
        except Exception as e:
            logger.error(f"Error collecting Facebook: {e}")
    else:
        logger.info("Facebook collection disabled in configuration")
    
    # Check if at least one source collected data
    if not youtube_stats and not facebook_stats:
        logger.error("No data collected from any source")
        return 1
    
    logger.info("="*60)
    logger.info("DATA COLLECTION STAGE COMPLETE")
    logger.info("="*60)

    logger.info("Fetching latest data from database...")
    content_items = process_latest_data(
        pipeline_run_id=pipeline_run_id,
        collect_youtube=config.data_collection.collect_youtube,
        collect_facebook=config.data_collection.collect_facebook
    )
    logger.info(f"Retrieved {len(content_items)} unique content items after deduplication")
    
    logger.info("="*60)
    logger.info(f"Job 1 Complete! Pipeline Run ID: {pipeline_run_id}")
    logger.info(f"Content items: {len(content_items)}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
