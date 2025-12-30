"""Job 0: Initialize Pipeline Run - Create pipeline run ID and metadata."""

import sys

from pipeline.db import clear_checkpoints, init_pipeline_run
from pipeline.logger import setup_logger

logger = setup_logger("job0")


def main():
    """Initialize a new pipeline run.
    
    Returns:
        int: Exit code (0 for success).
    """
    logger.info("="*60)
    logger.info("Job 0: Initialize Pipeline")
    logger.info("="*60)
    
    # Initialize pipeline run in database
    logger.info("Initializing pipeline run in database...")
    pipeline_run_id = init_pipeline_run()
    logger.info(f"Pipeline run initialized (ID: {pipeline_run_id})")
    
    # Clear any old checkpoints at the start of a new run
    logger.info("Clearing old checkpoints...")
    clear_checkpoints()
    logger.info("Checkpoints cleared")
    
    logger.info("="*60)
    logger.info(f"Job 0 Complete! Pipeline Run ID: {pipeline_run_id}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
