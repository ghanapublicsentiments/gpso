"""Facebook post retrieval stage for sentiment pipeline."""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

from database.bigquery_manager import get_bigquery_manager
from pipeline.constants import FACEBOOK_PAGES, FACEBOOK_POSTS_MAX_RESULTS
from pipeline.logger import get_logger

logger = get_logger("sources.facebook")

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
elif not os.getenv("FACEBOOK_ACCESS_TOKEN"):
    logger.warning(f"⚠️  Warning: .env file not found at {env_path}")
    logger.warning("   Copy .env.sample to .env and fill in your API keys")


def get_facebook_client() -> str:
    """Get Facebook Graph API access token.

    Returns:
        str: Facebook access token.

    Raises:
        ValueError: If FACEBOOK_ACCESS_TOKEN environment variable not set.
    """
    access_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("FACEBOOK_ACCESS_TOKEN environment variable not set")
    return access_token


def get_page_posts_today(page_id: str, max_posts: int = 100) -> list[dict]:
    """Get all posts from T-24h to T (now) from a Facebook page.

    Uses Facebook Graph API to retrieve recent posts from public pages.

    Args:
        page_id: Facebook page ID (e.g., 123456789).
        max_posts: Maximum number of posts to check (default 100).

    Returns:
        list[dict]: List of post info dictionaries with post_id, message, and created_time.
    """
    if not page_id:
        logger.warning("Page ID is required")
        return []

    recent_posts = []

    try:
        access_token = get_facebook_client()
        base_url = f"https://graph.facebook.com/v18.0/{page_id}/posts"

        now = datetime.now(timezone.utc)
        time_24h_ago = now - timedelta(hours=24)

        logger.debug(f"Fetching posts for page ID: {page_id}")

        params = {
            'access_token': access_token,
            'fields': 'id,message,created_time,permalink_url',
            'limit': min(FACEBOOK_POSTS_MAX_RESULTS, max_posts)
        }

        posts_checked = 0
        next_url = base_url

        while posts_checked < max_posts and next_url:
            response = requests.get(
                next_url, 
                params=params if next_url == base_url else None,
                timeout=30  # 30 second timeout to prevent hanging
            )
            response.raise_for_status()
            data = response.json()

            for item in data.get('data', []):
                posts_checked += 1

                post_id = item.get('id')
                message = item.get('message', '')
                created_time = item.get('created_time', '')
                permalink_url = item.get('permalink_url', '')

                try:
                    # Facebook API returns ISO format timestamps
                    created_dt = datetime.fromisoformat(created_time.replace('Z', '+00:00'))

                    if created_dt.tzinfo is None:
                        created_dt_utc = created_dt.replace(tzinfo=timezone.utc)
                    else:
                        created_dt_utc = created_dt.astimezone(timezone.utc)

                    now_utc = datetime.now(timezone.utc)
                    hours_ago = (now_utc - created_dt_utc).total_seconds() / 3600

                    if 0 <= hours_ago <= 24:
                        post_info = {
                            'post_id': post_id,
                            'message': message,
                            'url': permalink_url,
                            'created_at': created_dt_utc.isoformat()
                        }
                        recent_posts.append(post_info)
                        logger.debug(f"Added post: {message[:50]}... (posted {int(hours_ago)}h ago)")
                    elif hours_ago > 24:
                        # Posts are ordered by date, so if we hit something older than 24h, we can stop
                        logger.debug(f"Post is {int(hours_ago)}h old, stopping search")
                        return recent_posts

                except Exception as e:
                    logger.debug(f"Error parsing created time for post {post_id}: {e}")
                    continue

            # Check for pagination
            paging = data.get('paging', {})
            next_url = paging.get('next')
            params = None  # Next URL already contains all parameters

        logger.info(f"Found {len(recent_posts)} posts in last 24 hours for page {page_id}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching posts for page {page_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching posts for page {page_id}: {e}")

    return recent_posts


def retrieve_facebook_comments(post_id: str, max_comments: int = 500) -> list[dict]:
    """Retrieve Facebook comments using Graph API.

    Handles pagination to fetch comments and nested replies.

    Args:
        post_id: Facebook post ID.
        max_comments: Maximum number of comments to extract (will fetch in batches).

    Returns:
        list[dict]: List of comment dictionaries (top-level comments + replies).
    """
    comments = []

    try:
        access_token = get_facebook_client()
        base_url = f"https://graph.facebook.com/v18.0/{post_id}/comments"

        params = {
            'access_token': access_token,
            'fields': 'id,from,message,created_time,like_count,comment_count,comments{id,from,message,created_time,like_count}',
            'limit': min(100, max_comments)
        }

        comments_fetched = 0
        next_url = base_url

        while comments_fetched < max_comments and next_url:
            response = requests.get(
                next_url, 
                params=params if next_url == base_url else None,
                timeout=30  # 30 second timeout to prevent hanging
            )
            response.raise_for_status()
            data = response.json()

            # Extract comments from response
            for item in data.get('data', []):
                comment_id = item.get('id')
                from_info = item.get('from', {})
                author = from_info.get('name', 'Anonymous')
                message = item.get('message', '')
                created_time = item.get('created_time', '')
                like_count = item.get('like_count', 0)
                reply_count = item.get('comment_count', 0)

                # Add top-level comment
                comments.append({
                    'comment_id': comment_id,
                    'author': author,
                    'text': message,
                    'published_time': created_time,
                    'timestamp': created_time,
                    'votes': like_count,
                    'replies': reply_count,
                    'is_reply': False,
                    'parent_id': None
                })
                comments_fetched += 1

                # Add nested replies if they exist
                nested_comments = item.get('comments', {}).get('data', [])
                for reply in nested_comments:
                    reply_id = reply.get('id')
                    reply_from = reply.get('from', {})
                    reply_author = reply_from.get('name', 'Anonymous')
                    reply_message = reply.get('message', '')
                    reply_created_time = reply.get('created_time', '')
                    reply_like_count = reply.get('like_count', 0)

                    comments.append({
                        'comment_id': reply_id,
                        'author': reply_author,
                        'text': reply_message,
                        'published_time': reply_created_time,
                        'timestamp': reply_created_time,
                        'votes': reply_like_count,
                        'replies': 0,
                        'is_reply': True,
                        'parent_id': comment_id
                    })
                    comments_fetched += 1

            # Check for pagination
            paging = data.get('paging', {})
            next_url = paging.get('next')
            params = None  # Next URL already contains all parameters

        logger.debug(f"Completed fetching {len(comments)} comments for post {post_id}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching comments for post {post_id}: {e} (collected {len(comments)} before error)")
    except Exception as e:
        logger.warning(f"Unexpected error fetching comments for post {post_id}: {e} (collected {len(comments)} before error)")

    return comments


def save_facebook_to_database(
    pipeline_run_id: int,
    max_posts_per_page: int = 100,
    min_comments: int = 1,
    max_comments_per_post: int = 500,
    pages_to_scrape: Optional[list[str]] = None
) -> dict:
    """Collect Facebook posts and save directly to database.

    Args:
        pipeline_run_id: Pipeline run ID to associate with scraped data.
        max_posts_per_page: Maximum posts to check per page.
        min_comments: Minimum comments required to include post.
        max_comments_per_post: Maximum total comments to fetch per post (including replies).
        pages_to_scrape: List of page names to scrape (None = all).

    Returns:
        dict: Dictionary with scraping statistics.
    """
    stats = {
        'posts_inserted': 0,
        'posts_skipped': 0,
        'comments': 0,
        'pages_scraped': 0,
        'pages_failed': 0,
        'errors': []
    }

    facebook_pages_config = FACEBOOK_PAGES

    logger.info("="*70)
    logger.info(f"Starting Facebook post scraping for pipeline run {pipeline_run_id}")
    logger.info(f"Pages to scrape: {len(facebook_pages_config)}")
    logger.info(f"Max posts per page: {max_posts_per_page}")
    logger.info(f"Min comments required: {min_comments}")
    logger.info("="*70)

    bq_manager = get_bigquery_manager()

    for page_idx, (page_name, page_info) in enumerate(facebook_pages_config.items(), 1):
        # Skip if specific pages requested and this isn't one
        if pages_to_scrape and page_name not in pages_to_scrape:
            continue

        facebook_page_id = page_info["page_id"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Page {page_idx}/{len(facebook_pages_config)}: {page_name}")
        logger.info(f"Page ID: {facebook_page_id}")
        logger.info(f"{'='*70}")

        try:
            today_posts = get_page_posts_today(
                facebook_page_id,
                max_posts=max_posts_per_page
            )

            if not today_posts:
                logger.warning(f"No recent posts found for {page_name}")
                continue

            logger.info(f"Found {len(today_posts)} recent posts to process")

            for post_idx, post_info in enumerate(today_posts, 1):
                logger.info(f"\n  Post {post_idx}/{len(today_posts)}")

                try:
                    post_id = post_info['post_id']
                    message_preview = post_info['message'][:80] if post_info['message'] else '[No message]'

                    logger.info(f"  📝 Message: {message_preview}...")

                    logger.info(f"  🔍 Fetching comments...")
                    all_comments = retrieve_facebook_comments(post_id, max_comments=max_comments_per_post)

                    total_comments = len(all_comments)
                    logger.info(f"  💬 Comments: {total_comments}")

                    # Only save posts that have enough comments
                    if total_comments < min_comments:
                        logger.warning(f"  ⏭️  Skipped: Only {total_comments} comments (need {min_comments}+)")
                        stats['posts_skipped'] += 1
                        continue

                    logger.info(f"  ✅ Post meets criteria, saving to database...")

                    created_at_str = post_info.get('created_at')
                    created_date = datetime.fromisoformat(created_at_str.replace('Z', '+00:00')) if created_at_str else None

                    post_db_id = bq_manager.insert_facebook_post(
                        page_id=facebook_page_id,
                        page_name=page_name,
                        post_id=post_id,
                        post_message=post_info['message'],
                        post_url=post_info['url'],
                        created_date=created_date,
                        pipeline_run_id=pipeline_run_id
                    )

                    if post_db_id:
                        stats['posts_inserted'] += 1
                        logger.info(f"  ✓ Post saved to database (ID: {post_db_id})")

                        if all_comments:
                            comment_count = bq_manager.insert_facebook_comments(
                                post_id=post_db_id,
                                comments=[
                                    {
                                        'author': c.get('author', 'Anonymous'),
                                        'text': c.get('text', ''),
                                        'votes': c.get('votes', 0),
                                        'published_time': c.get('published_time'),
                                        'comment_id': c.get('comment_id'),
                                        'is_reply': c.get('is_reply', False),
                                        'parent_comment_id': c.get('parent_id')
                                    } for c in all_comments
                                ]
                            )
                            stats['comments'] += comment_count
                            logger.info(f"  ✓ Saved {comment_count} comments")

                    logger.info(f"  ✅ Post processing complete")

                    if post_idx < len(today_posts):
                        time.sleep(1)

                except Exception as e:
                    error_msg = f"Error processing post {post_info.get('url')}: {e}"
                    logger.error(f"  ❌ {error_msg}")
                    stats['errors'].append(error_msg)

            stats['pages_scraped'] += 1
            logger.info(f"\n✅ Page {page_name} complete!")
            logger.info(f"   Posts: {stats['posts_inserted']} saved, {stats['posts_skipped']} skipped")

            if page_idx < len(facebook_pages_config):
                logger.info(f"Waiting 3 seconds before next page...")
                time.sleep(3)

        except Exception as e:
            error_msg = f"Error collecting data for {page_name}: {e}"
            logger.error(f"❌ {error_msg}")
            stats['pages_failed'] += 1
            stats['errors'].append(error_msg)
            continue

    logger.info("\n" + "="*70)
    logger.info("FACEBOOK DATA COLLECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"📊 Summary:")
    logger.info(f"   Pages scraped: {stats['pages_scraped']}/{len(facebook_pages_config)}")
    logger.info(f"   Posts saved: {stats['posts_inserted']}")
    logger.info(f"   Posts skipped: {stats['posts_skipped']}")
    logger.info(f"   Comments: {stats['comments']}")
    if stats['pages_failed'] > 0:
        logger.warning(f"   ⚠️  Pages failed: {stats['pages_failed']}")
    if stats['errors']:
        logger.warning(f"   ⚠️  Errors encountered: {len(stats['errors'])}")
    logger.info("="*70 + "\n")

    return stats
