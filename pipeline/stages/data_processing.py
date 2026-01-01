"""Data fetching and grouping stage for pipeline."""

import pandas as pd

from database.bigquery_manager import get_bigquery_manager
from pipeline.logger import get_logger

logger = get_logger("stages.data_processing")


def fetch_latest_data(pipeline_run_id: int, collect_youtube: bool = True, collect_facebook: bool = True) -> pd.DataFrame:
    """Fetch latest scraped data from BigQuery by pipeline run ID.

    Retrieves YouTube videos and/or Facebook posts with their comments from 
    the database that were scraped during the specified pipeline run.

    Args:
        pipeline_run_id: Pipeline run ID to fetch data for.
        collect_youtube: Whether to fetch YouTube data (default: True).
        collect_facebook: Whether to fetch Facebook data (default: True).

    Returns:
        pd.DataFrame: DataFrame with combined YouTube and/or Facebook data.
    """
    logger.info(f"Fetching content data for pipeline run {pipeline_run_id}")

    manager = get_bigquery_manager()
    client = manager.client
    dataset_id = manager.dataset_id

    # Fetch YouTube videos
    youtube_query = f"""
    WITH deduplicated AS (
        SELECT 
            v.video_id as content_id,
            v.channel_name as source_name,
            'youtube_video' as source_type,
            v.title,
            v.published_date as published_date, 
            v.scraped_at,
            vc.comment_text, 
            vc.author_id as comment_author_id,
            vc.votes as comment_likes,
            vc.published_time as comment_timestamp, 
            vc.external_comment_id as comment_id,
            vc.is_reply as comment_is_reply,
            vc.parent_comment_id,
            ROW_NUMBER() OVER (
                PARTITION BY 
                v.channel_name, v.title, v.published_date, 
                vc.author_id, vc.published_time, vc.comment_text 
                ORDER BY v.scraped_at DESC
            ) as rn
        FROM `{dataset_id}.youtube_videos` v
        LEFT JOIN `{dataset_id}.youtube_comments` vc ON v.id = vc.video_id
        WHERE v.pipeline_run_id = {pipeline_run_id}
    )
    SELECT 
        content_id,
        source_name,
        source_type,
        title,
        published_date,
        scraped_at,
        comment_text,
        comment_author_id,
        comment_likes,
        comment_timestamp,
        comment_id,
        comment_is_reply,
        parent_comment_id
    FROM deduplicated
    WHERE rn = 1
    ORDER BY published_date DESC
    """
    
    # Fetch Facebook posts
    facebook_query = f"""
    WITH deduplicated AS (
        SELECT 
            p.post_id as content_id,
            p.page_name as source_name,
            'facebook_post' as source_type,
            SUBSTR(p.message, 1, 200) as title,
            p.created_date as published_date,
            p.scraped_at,
            fc.comment_text, 
            fc.author_id as comment_author_id,
            fc.votes as comment_likes,
            fc.published_time as comment_timestamp, 
            fc.external_comment_id as comment_id,
            fc.is_reply as comment_is_reply,
            fc.parent_comment_id,
            ROW_NUMBER() OVER (
                PARTITION BY 
                p.page_name, p.message, p.created_date, 
                fc.author_id, fc.published_time, fc.comment_text 
                ORDER BY p.scraped_at DESC
            ) as rn
        FROM `{dataset_id}.facebook_posts` p
        LEFT JOIN `{dataset_id}.facebook_comments` fc ON p.id = fc.post_id
        WHERE p.pipeline_run_id = {pipeline_run_id}
    )
    SELECT 
        content_id,
        source_name,
        source_type,
        title,
        published_date,
        scraped_at,
        comment_text,
        comment_author_id,
        comment_likes,
        comment_timestamp,
        comment_id,
        comment_is_reply,
        parent_comment_id
    FROM deduplicated
    WHERE rn = 1
    ORDER BY published_date DESC
    """
    
    # Execute queries based on configuration
    dataframes = []
    
    if collect_youtube:
        logger.info("Fetching YouTube data...")
        youtube_df = client.query(youtube_query).to_dataframe()
        logger.info(f"YouTube: {youtube_df['content_id'].nunique() if not youtube_df.empty else 0} videos, {len(youtube_df)} comments (after deduplication)")
        dataframes.append(youtube_df)
    else:
        logger.info("Skipping YouTube data (collect_youtube=False)")
    
    if collect_facebook:
        logger.info("Fetching Facebook data...")
        facebook_df = client.query(facebook_query).to_dataframe()
        logger.info(f"Facebook: {facebook_df['content_id'].nunique() if not facebook_df.empty else 0} posts, {len(facebook_df)} comments (after deduplication)")
        dataframes.append(facebook_df)
    else:
        logger.info("Skipping Facebook data (collect_facebook=False)")
    
    # Combine DataFrames based on what was collected
    if not dataframes:
        logger.warning("No data sources enabled! Both collect_youtube and collect_facebook are False")
        combined_df = pd.DataFrame()
    elif len(dataframes) == 1:
        combined_df = dataframes[0]
    else:
        combined_df = pd.concat(dataframes, ignore_index=True)
    
    logger.info(f"Total: {combined_df['content_id'].nunique() if not combined_df.empty else 0} content items, {len(combined_df)} comments")
    
    return combined_df

def group_comments_by_newsitem(df: pd.DataFrame) -> list[dict]:
    """Group comments by content item (video or post) with detailed conversation history.

    Creates a list of content records, where each record contains:
    - content metadata (content_id, source info, title, etc.)
    - comments_by_author: dictionary keyed by author_id with list of comment objects

    Each comment object includes:
    - timestamp: when the comment was posted
    - comment_text: the comment content
    - conversation_history: if reply, includes parent comment and thread history

    For replies, the conversation history includes:
    - The parent comment with its timestamp and text
    - All other comments in the thread before the current comment (chronologically ordered)

    Args:
        df: DataFrame with content (videos/posts) and comments.

    Returns:
        list[dict]: List of content record dictionaries with nested comment structure.
    """
    content_records = []

    for content_id in df['content_id'].unique():
        # Get all comments for this content item, sorted by timestamp
        content_df = df[(df['content_id'] == content_id) & (df['comment_text'].notnull())].sort_values(
            by='comment_timestamp', ascending=True
        )

        if len(content_df) == 0:
            continue
        
        # Build a map of comment_id -> comment data for building conversation history
        comments_map = {}
        for _, row in content_df.iterrows():
            comment_id = row['comment_id']
            comments_map[comment_id] = {
                'author_id': row['comment_author_id'],
                'timestamp': row['comment_timestamp'],
                'text': row['comment_text'],
                'is_reply': row['comment_is_reply'],
                'parent_comment_id': row['parent_comment_id']
            }

        comments_by_author = {}

        for _, row in content_df.iterrows():
            author_id = row['comment_author_id']
            comment_id = row['comment_id']
            
            # Build conversation history if this is a reply
            conversation_history = None
            if row['comment_is_reply'] and row['parent_comment_id']:
                parent_id = row['parent_comment_id']
                
                # Get the parent comment
                if parent_id in comments_map:
                    parent = comments_map[parent_id]
                    conversation_history = {
                        'parent_comment': {
                            'author_id': parent['author_id'],
                            'timestamp': parent['timestamp'],
                            'text': parent['text']
                        },
                        'thread_comments': []
                    }
                    
                    # Get all comments in this thread that came before the current comment
                    current_timestamp = row['comment_timestamp']
                    for cid, cdata in comments_map.items():
                        # Include comments that are replies to the same parent and came before current comment
                        if (
                            cdata.get('parent_comment_id') == parent_id and
                            cdata['timestamp'] < current_timestamp and
                            cid != comment_id
                        ):
                            conversation_history['thread_comments'].append({
                                'author_id': cdata['author_id'],
                                'timestamp': cdata['timestamp'],
                                'text': cdata['text']
                            })

                    conversation_history['thread_comments'].sort(key=lambda x: x['timestamp'])

            comment_obj = {
                'timestamp': row['comment_timestamp'],
                'comment_text': row['comment_text'],
                'conversation_history': conversation_history
            }
            
            # Add to author's comment list
            if author_id not in comments_by_author:
                comments_by_author[author_id] = []
            comments_by_author[author_id].append(comment_obj)
        
        # Create the content record with the nested structure
        # Calculate total comment count across all authors
        total_comment_count = sum(len(comments) for comments in comments_by_author.values())

        content_record = {
            'content_id': content_id,
            'source_type': content_df.iloc[0]['source_type'],
            'source_name': content_df.iloc[0]['source_name'],
            'news_title': content_df.iloc[0]['title'],
            'comments_by_author': comments_by_author,
            'comment_count': total_comment_count
        }
        
        content_records.append(content_record)
    
    return content_records


def process_latest_data(pipeline_run_id: int, collect_youtube: bool = True, collect_facebook: bool = True) -> pd.DataFrame:
    """Fetch and group latest data from the database by pipeline run ID.
    
    Args:
        pipeline_run_id: Pipeline run ID to fetch data for.
        collect_youtube: Whether to fetch YouTube data (default: True).
        collect_facebook: Whether to fetch Facebook data (default: True).
    
    Returns:
        DataFrame with content records and nested comment structure
        Columns: content_id, source_type, source_name, news_title, 
            comments_by_author, comment_count
    """
    raw_df = fetch_latest_data(
        pipeline_run_id=pipeline_run_id,
        collect_youtube=collect_youtube,
        collect_facebook=collect_facebook
    )
    content_items = group_comments_by_newsitem(raw_df)
    
    # Convert list of dicts to DataFrame
    content_df = pd.DataFrame(content_items)
    
    return content_df