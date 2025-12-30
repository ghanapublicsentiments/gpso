"""Facebook data collection module."""

from .facebook import (
    get_facebook_client,
    get_page_posts_today,
    retrieve_facebook_comments,
    save_facebook_to_database,
)

__all__ = [
    "get_facebook_client",
    "get_page_posts_today",
    "retrieve_facebook_comments",
    "save_facebook_to_database",
]
