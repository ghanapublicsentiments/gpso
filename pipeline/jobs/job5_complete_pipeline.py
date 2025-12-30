"""Job 5: Complete Pipeline Run - Finalize pipeline with token statistics."""

import sys

from pipeline.db import complete_pipeline_run, get_pipeline_token_stats, summaries_exist
from pipeline.logger import setup_logger
from pipeline.pipeline_config import PipelineConfig

logger = setup_logger("job5")


def main():
    """Complete the pipeline run with final statistics.
    
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    config = PipelineConfig()
    
    # Job 5 requires a pipeline run ID
    if config.pipeline_run_id is None:
        logger.error("Job 5 requires a pipeline run ID")
        logger.error("Usage: python job5_complete_pipeline.py --run-id <RUN_ID>")
        return 1
    
    pipeline_run_id = config.pipeline_run_id
    logger.info("="*60)
    logger.info(f"Job 5: Complete Pipeline - Run ID: {pipeline_run_id}")
    logger.info("="*60)
    
    # Verify Job 4 completed by checking if entity summaries exist
    if not summaries_exist(pipeline_run_id):
        logger.error("Job 4 (post-processing) has not completed yet")
        logger.error("No entity summaries found for this pipeline run")
        logger.error("Run Job 4 first to complete sentiment analysis and summaries")
        return 1
    
    logger.info("Aggregating token statistics from all pipeline stages...")
    token_stats = get_pipeline_token_stats(pipeline_run_id)
    
    logger.info("Token usage summary:")
    logger.info(f"  Hot Topics - Input: {token_stats['caption_input']:,}, Output: {token_stats['caption_output']:,}")
    logger.info(f"  Sentiments - Input: {token_stats['sentiments_input']:,}, Output: {token_stats['sentiments_output']:,}")
    logger.info(f"  Summaries  - Input: {token_stats['summary_input']:,}, Output: {token_stats['summary_output']:,}")
    logger.info(f"  TOTAL      - Input: {token_stats['total_input']:,}, Output: {token_stats['total_output']:,}")
    
    logger.info("Marking pipeline run as completed...")
    complete_pipeline_run(pipeline_run_id, token_stats, status='completed')
    logger.info("Pipeline run marked as completed")
    
    logger.info("="*60)
    logger.info(f"Job 5 Complete! Pipeline Run ID: {pipeline_run_id}")
    logger.info("Status: COMPLETED")
    logger.info("="*60)
    logger.info("✅ FULL PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
