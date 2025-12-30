"""YouTube video retrieval stage for sentiment pipeline."""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dateutil import parser
from dotenv import load_dotenv
from googleapiclient.discovery import build

from database.bigquery_manager import get_bigquery_manager
from pipeline.constants import (
    YOUTUBE_CHANNELS,
    YOUTUBE_COMMENTS_MAX_RESULTS,
    YOUTUBE_SEARCH_MAX_RESULTS,
)
from pipeline.logger import get_logger

logger = get_logger("sources.youtube")

project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
elif not os.getenv("GOOGLE_API_KEY"):
    logger.warning(f"⚠️  Warning: .env file not found at {env_path}")
    logger.warning("   Copy .env.sample to .env and fill in your API keys")

def get_youtube_client():
    """Get YouTube Data API client.

    Returns:
        YouTube API client instance.

    Raises:
        ValueError: If GOOGLE_API_KEY environment variable not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    return build('youtube', 'v3', developerKey=api_key)


def get_channel_videos_today(channel_id: str, max_videos: int = 100) -> list[dict]:
    """Get all videos uploaded in the window from T-24h to T (now) from a YouTube channel.

    Uses YouTube Data API to retrieve recent videos.

    Args:
        channel_id: YouTube channel ID (e.g., UCxxxxxxxxxxxxxx).
        max_videos: Maximum number of videos to check (default 100).

    Returns:
        list[dict]: List of video info dictionaries with video_id, title, and upload date.
    """
    if not channel_id:
        logger.warning("Channel ID is required")
        return []

    recent_videos = []

    try:
        youtube = get_youtube_client()

        now = datetime.now()
        time_24h_ago = (now - timedelta(hours=24)).isoformat() + 'Z'

        logger.debug(f"Fetching videos for channel ID: {channel_id}")

        next_page_token = None
        videos_checked = 0

        while videos_checked < max_videos:
            page_size = min(YOUTUBE_SEARCH_MAX_RESULTS, max_videos - videos_checked)

            request = youtube.search().list(
                part="snippet",
                channelId=channel_id,
                type="video",
                order="date",
                maxResults=page_size,
                pageToken=next_page_token,
                publishedAfter=time_24h_ago
            )

            response = request.execute()

            for item in response.get('items', []):
                videos_checked += 1

                video_id = item['id']['videoId']
                snippet = item['snippet']
                title = snippet.get('title', 'Unknown')
                published_at = snippet.get('publishedAt', '')

                try:
                    # YouTube API returns UTC timestamps (e.g., "2024-01-15T10:30:00Z")
                    # Parse the ISO format datetime and explicitly convert to UTC
                    published_dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))

                    if published_dt.tzinfo is None:
                        published_dt_utc = published_dt.replace(tzinfo=timezone.utc)
                    else:
                        published_dt_utc = published_dt.astimezone(timezone.utc)

                    now_utc = datetime.now(timezone.utc)
                    hours_ago = (now_utc - published_dt_utc).total_seconds() / 3600

                    if 0 <= hours_ago <= 24:
                        video_info = {
                            'video_id': video_id,
                            'title': title,
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'published_at': published_dt_utc.isoformat()
                        }
                        recent_videos.append(video_info)
                        logger.debug(f"Added video: {title} (published {int(hours_ago)}h ago)")
                    elif hours_ago > 24:
                        # Videos are ordered by date, so if we hit something older than 24h, we can stop
                        logger.debug(f"Video {title} is {int(hours_ago)}h old, stopping search")
                        return recent_videos

                except Exception as e:
                    logger.debug(f"Error parsing published time for video {video_id}: {e}")
                    continue

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        logger.info(f"Found {len(recent_videos)} videos in last 24 hours for channel {channel_id}")

    except Exception as e:
        logger.error(f"Error fetching videos for channel {channel_id}: {e}")

    return recent_videos


def retrieve_youtube_comments(video_id: str, max_comments: int = 500, include_replies: bool = True) -> list[dict]:
    """Retrieve YouTube comments using official YouTube Data API v3.

    Handles pagination to fetch more than 1000 comments if needed.

    Args:
        video_id: YouTube video ID.
        max_comments: Maximum number of comments to extract (will fetch in batches).
        include_replies: Whether to fetch replies to comments (default: True).

    Returns:
        list[dict]: List of comment dictionaries (top-level comments + replies if include_replies=True).
    """
    comments = []

    try:
        youtube = get_youtube_client()

        next_page_token = None
        comments_fetched = 0
        pages_fetched = 0

        while comments_fetched < max_comments:
            page_size = min(YOUTUBE_COMMENTS_MAX_RESULTS, max_comments - comments_fetched)

            request = youtube.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id,
                maxResults=page_size,
                textFormat="plainText",
                pageToken=next_page_token
            )
            response = request.execute()
            pages_fetched += 1
            
            # Extract comments from response
            batch_size = 0
            for item in response.get('items', []):
                top_level_comment = item['snippet']['topLevelComment']
                snippet = top_level_comment['snippet']
                parent_comment_id = top_level_comment['id']
                
                # Add top-level comment
                comments.append({
                    'comment_id': parent_comment_id,
                    'author': snippet.get('authorDisplayName', ''),
                    'text': snippet.get('textDisplay', ''),
                    'published_time': snippet.get('publishedAt', ''),
                    'timestamp': snippet.get('publishedAt', ''),
                    'votes': snippet.get('likeCount', 0),
                    'replies': item['snippet'].get('totalReplyCount', 0),
                    'is_reply': False,
                    'parent_id': None
                })
                comments_fetched += 1
                batch_size += 1
                
                # Add replies if they exist and include_replies is True
                if include_replies and 'replies' in item:
                    reply_comments = item['replies'].get('comments', [])
                    for reply in reply_comments:
                        reply_snippet = reply['snippet']
                        comments.append({
                            'comment_id': reply['id'],
                            'author': reply_snippet.get('authorDisplayName', ''),
                            'text': reply_snippet.get('textDisplay', ''),
                            'published_time': reply_snippet.get('publishedAt', ''),
                            'timestamp': reply_snippet.get('publishedAt', ''),
                            'votes': reply_snippet.get('likeCount', 0),
                            'replies': 0,
                            'is_reply': True,
                            'parent_id': parent_comment_id
                        })
                        comments_fetched += 1
                        batch_size += 1

            logger.debug(f"Fetched page {pages_fetched}: {batch_size} comments (total: {comments_fetched})")

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                logger.debug(f"No more pages available (fetched {pages_fetched} pages total)")
                break  # No more comments available
        
        logger.debug(f"Completed fetching {len(comments)} comments for video {video_id}")

    except Exception as e:
        logger.warning(f"Error fetching comments for video {video_id}: {e} (collected {len(comments)} before error)")

    return comments


def save_youtube_to_database(
    pipeline_run_id: int,
    max_videos_per_channel: int = 100,
    min_comments: int = 1,
    max_comments_per_video: int = 500,
    channels_to_scrape: Optional[list[str]] = None
) -> dict:
    """Collect YouTube videos and save directly to database.

    Args:
        pipeline_run_id: Pipeline run ID to associate with scraped data.
        max_videos_per_channel: Maximum videos to check per channel.
        min_comments: Minimum comments required to include video.
        max_comments_per_video: Maximum total comments to fetch per video (including replies).
        channels_to_scrape: List of channel names to scrape (None = all).

    Returns:
        dict: Dictionary with scraping statistics.
    """
    stats = {
        'videos_inserted': 0,
        'videos_skipped': 0,
        'regular_comments': 0,
        'channels_scraped': 0,
        'channels_failed': 0,
        'errors': []
    }

    youtube_channels_config = YOUTUBE_CHANNELS

    logger.info("="*70)
    logger.info(f"Starting YouTube video scraping for pipeline run {pipeline_run_id}")
    logger.info(f"Channels to scrape: {len(youtube_channels_config)}")
    logger.info(f"Max videos per channel: {max_videos_per_channel}")
    logger.info(f"Min comments required: {min_comments}")
    logger.info("="*70)

    bq_manager = get_bigquery_manager()

    for channel_idx, (channel_name, channel_info) in enumerate(youtube_channels_config.items(), 1):
        # Skip if specific channels requested and this isn't one
        if channels_to_scrape and channel_name not in channels_to_scrape:
            continue

        youtube_channel_id = channel_info["channel_id"]
        channel_handle = channel_info.get("channel_handle", "")

        logger.info(f"\n{'='*70}")
        logger.info(f"Channel {channel_idx}/{len(youtube_channels_config)}: {channel_name}")
        logger.info(f"Channel ID: {youtube_channel_id}")
        if channel_handle:
            logger.info(f"Channel Handle: @{channel_handle}")
        logger.info(f"{'='*70}")

        try:
            today_videos = get_channel_videos_today(
                youtube_channel_id,
                max_videos=max_videos_per_channel
            )

            if not today_videos:
                logger.warning(f"No recent videos found for {channel_name}")
                continue

            logger.info(f"Found {len(today_videos)} recent videos to process")

            for video_idx, video_info in enumerate(today_videos, 1):
                logger.info(f"\n  Video {video_idx}/{len(today_videos)}")

                try:
                    video_id = video_info['video_id']

                    logger.info(f"  📹 Title: {video_info['title'][:80]}...")

                    logger.info(f"  🔍 Fetching comments...")
                    regular = retrieve_youtube_comments(video_id, max_comments=max_comments_per_video, include_replies=True)

                    total_comments = len(regular)
                    logger.info(f"  💬 Comments: {total_comments}")

                    # Only save videos that have enough comments
                    if total_comments < min_comments:
                        logger.warning(f"  ⏭️  Skipped: Only {total_comments} comments (need {min_comments}+)")
                        stats['videos_skipped'] += 1
                        continue

                    logger.info(f"  ✅ Video meets criteria, saving to database...")

                    published_at_str = video_info.get('published_at')
                    publication_date = parser.parse(published_at_str) if published_at_str else None

                    video_db_id = bq_manager.insert_youtube_video(
                        channel_id=youtube_channel_id,
                        channel_name=channel_name,
                        video_id=video_id,
                        video_title=video_info['title'],
                        video_url=video_info['url'],
                        publication_date=publication_date,
                        pipeline_run_id=pipeline_run_id
                    )

                    if video_db_id:
                        stats['videos_inserted'] += 1
                        logger.info(f"  ✓ Video saved to database (ID: {video_db_id})")

                        if regular:
                            regular_count = bq_manager.insert_youtube_comments(
                                video_id=video_db_id,
                                comments=[
                                    {
                                        'author': c.get('author', 'Anonymous'),
                                        'text': c.get('text', ''),
                                        'votes': c.get('votes', 0),
                                        'published_time': c.get('published_time'),
                                        'comment_id': c.get('comment_id'),
                                        'is_reply': c.get('is_reply', False),
                                        'parent_comment_id': c.get('parent_id')
                                    } for c in regular
                                ]
                            )
                            stats['regular_comments'] += regular_count
                            logger.info(f"  ✓ Saved {regular_count} regular comments")

                    logger.info(f"  ✅ Video processing complete")

                    if video_idx < len(today_videos):
                        time.sleep(1)

                except Exception as e:
                    error_msg = f"Error processing video {video_info.get('url')}: {e}"
                    logger.error(f"  ❌ {error_msg}")
                    stats['errors'].append(error_msg)

            stats['channels_scraped'] += 1
            logger.info(f"\n✅ Channel {channel_name} complete!")
            logger.info(f"   Videos: {stats['videos_inserted']} saved, {stats['videos_skipped']} skipped")

            if channel_idx < len(youtube_channels_config):
                logger.info(f"Waiting 3 seconds before next channel...")
                time.sleep(3)

        except Exception as e:
            error_msg = f"Error collecting data for {channel_name}: {e}"
            logger.error(f"❌ {error_msg}")
            stats['channels_failed'] += 1
            stats['errors'].append(error_msg)
            continue

    logger.info("\n" + "="*70)
    logger.info("YOUTUBE DATA COLLECTION COMPLETE")
    logger.info("="*70)
    logger.info(f"📊 Summary:")
    logger.info(f"   Channels scraped: {stats['channels_scraped']}/{len(youtube_channels_config)}")
    logger.info(f"   Videos saved: {stats['videos_inserted']}")
    logger.info(f"   Videos skipped: {stats['videos_skipped']}")
    logger.info(f"   Comments: {stats['regular_comments']}")
    if stats['channels_failed'] > 0:
        logger.warning(f"   ⚠️  Channels failed: {stats['channels_failed']}")
    if stats['errors']:
        logger.warning(f"   ⚠️  Errors encountered: {len(stats['errors'])}")
    logger.info("="*70 + "\n")

    return stats
