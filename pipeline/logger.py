"""Centralized logging configuration for the pipeline."""

import logging
import sys


class PipelineLogger:
    """Centralized logger for pipeline with hierarchical support for jobs and stages.

    Usage:
        # In job files:
        from logger import get_logger
        logger = get_logger("job1")
        logger.info("Job started")

        # In stage files:
        from logger import get_logger
        logger = get_logger("job1.news_collection")
        logger.info("Data collection started")
    """

    _initialized = False
    _level = logging.INFO

    @classmethod
    def initialize(cls, level: int = logging.INFO) -> None:
        """Initialize the root pipeline logger (call once at startup).

        Args:
            level: Logging level (default: INFO).
        """
        if cls._initialized:
            return

        cls._level = level
        
        # Configure root pipeline logger
        root_logger = logging.getLogger("pipeline")
        root_logger.setLevel(level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter with clean, readable format
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)

        root_logger.addHandler(handler)
        root_logger.propagate = False

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger with the given name.

        Args:
            name: Logger name (e.g., "job1" or "job1.news_collection").

        Returns:
            logging.Logger: Configured logger instance.
        """
        if not cls._initialized:
            cls.initialize()
        
        # Create hierarchical logger name
        full_name = f"pipeline.{name}" if not name.startswith("pipeline") else name
        logger = logging.getLogger(full_name)

        return logger


def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Legacy function for backward compatibility.

    Set up and return a configured logger for pipeline jobs.

    Args:
        name: Logger name (typically the job name).
        level: Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    PipelineLogger.initialize(level)
    return PipelineLogger.get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given component.

    Args:
        name: Component name (e.g., "job1", "job1.news_collection", "stages.smoothing").

    Returns:
        logging.Logger: Configured logger instance.

    Examples:
        logger = get_logger("job1")
        logger = get_logger("job1.news_collection")
        logger = get_logger("stages.sentiments")
    """
    return PipelineLogger.get_logger(name)
