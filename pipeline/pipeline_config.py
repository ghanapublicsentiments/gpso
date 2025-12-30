"""Pipeline configuration with all arguments organized by stage."""

import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CollectionConfig:
    """Configuration for data collection stage - YouTube and Facebook via official APIs."""
    collect_youtube: bool = True
    max_videos_per_channel: int = 100
    min_youtube_comments: int = 1
    max_comments_per_video: int = 200  # Total comments to fetch per video (including replies)
    collect_facebook: bool = False
    max_posts_per_page: int = 100
    min_facebook_comments: int = 1
    max_comments_per_post: int = 200  # Total comments to fetch per post (including replies)


@dataclass
class CaptionConfig:
    """Configuration for caption generation stage."""
    model: str = "gpt-5-nano"
    top_n: int = 10
    headline_similarity_threshold: float = 0.8
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis stage."""
    model: str = "gpt-5-nano"
    max_workers: int = 8
    rate_limit: int = 10  # Max API requests per minute


@dataclass
class SmoothingConfig:
    """Configuration for sentiment smoothing stage."""
    k_neighbors: int = 5
    min_neighbors: int = 1
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class NormalizationConfig:
    """Configuration for channel normalization stage."""
    window_days: int = 30
    normalization_strength: float = 0.8  # 0.1-2.0, higher = stronger compression


@dataclass
class SummaryConfig:
    """Configuration for summary generation stage."""
    model: str = "gpt-5-nano"
    max_workers: int = 8
    rate_limit: int = 10  # Max API requests per minute


@dataclass
class PipelineConfig:
    """Main pipeline configuration containing all stage configs."""

    data_collection: CollectionConfig = field(default_factory=CollectionConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    pipeline_run_id: Optional[int] = None

    def __post_init__(self) -> None:
        """Parse command-line arguments after initialization."""
        if '--run-id' in sys.argv:
            try:
                run_id_idx = sys.argv.index('--run-id')
                if run_id_idx + 1 < len(sys.argv):
                    self.pipeline_run_id = int(sys.argv[run_id_idx + 1])
            except (ValueError, IndexError):
                pass

    def to_dict(self) -> dict:
        """Convert config to dictionary for database storage.

        Returns:
            dict: Configuration as nested dictionary.
        """
        return {
            'data_collection': {
                'collect_youtube': self.data_collection.collect_youtube,
                'max_videos_per_channel': self.data_collection.max_videos_per_channel,
                'min_youtube_comments': self.data_collection.min_youtube_comments,
                'max_comments_per_video': self.data_collection.max_comments_per_video,
                'collect_facebook': self.data_collection.collect_facebook,
                'max_posts_per_page': self.data_collection.max_posts_per_page,
                'min_facebook_comments': self.data_collection.min_facebook_comments,
                'max_comments_per_post': self.data_collection.max_comments_per_post,
            },
            'caption': {
                'model': self.caption.model,
                'top_n': self.caption.top_n,
                'headline_similarity_threshold': self.caption.headline_similarity_threshold,
                'embedding_model': self.caption.embedding_model,
            },
            'sentiment': {
                'model': self.sentiment.model,
                'max_workers': self.sentiment.max_workers,
                'rate_limit': self.sentiment.rate_limit,
            },
            'smoothing': {
                'k_neighbors': self.smoothing.k_neighbors,
                'min_neighbors': self.smoothing.min_neighbors,
                'embedding_model': self.smoothing.embedding_model,
            },
            'normalization': {
                'window_days': self.normalization.window_days,
                'normalization_strength': self.normalization.normalization_strength,
            },
            'summary': {
                'model': self.summary.model,
                'max_workers': self.summary.max_workers,
                'rate_limit': self.summary.rate_limit,
            },
        }
