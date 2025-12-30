"""BigQuery Manager for GPSO.

This module provides a simplified interface for read/write operations
on BigQuery tables used in the GPSO pipeline.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import streamlit as st
from google.cloud import bigquery

from config import BIGQUERY_DATASET, BIGQUERY_IS_PROD
from database.bigquery_utils import get_bigquery_client, get_project_id


class BigQueryManager:
    """Manager for BigQuery database operations."""

    def __init__(self, creds_dict: Optional[dict] = None) -> None:
        """Initialize BigQuery manager with client and dataset configuration.
        
        Args:
            creds_dict: Optional credentials dictionary from Streamlit session state.
        """
        self.client = get_bigquery_client(creds_dict)
        self.project_id = get_project_id(creds_dict)
        self.dataset_id = f"{self.project_id}.{BIGQUERY_DATASET}"

    def _query(self, sql: str) -> list[dict]:
        """Execute query and return results as list of dicts.

        Args:
            sql: SQL query string.

        Returns:
            list[dict]: List of row dictionaries.
        """
        query_job = self.client.query(sql)
        results = query_job.result()
        return [dict(row) for row in results]

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a value for BigQuery JSON insertion.

        Handles NaN, Infinity, control characters, and overly long strings.

        Args:
            value: Value to sanitize.

        Returns:
            Any: Sanitized value safe for BigQuery insertion.
        """
        if value is None:
            return None

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None

        if isinstance(value, str):
            value = ''.join(
                char if ord(char) >= 32 or char in '\n\t' else ' '
                for char in value
            )
            # Limit string length to prevent length issues
            if len(value) > 10000:
                value = value[:10000]

        return value

    def _insert_rows(self, table_name: str, rows: list[dict]) -> int:
        """Insert rows using load job (free tier compatible).

        Args:
            table_name: Table name.
            rows: List of row dictionaries.

        Returns:
            int: Number of rows inserted.
        """
        if not rows:
            return 0

        sanitized_rows = []
        for row in rows:
            sanitized_row = {k: self._sanitize_value(v) for k, v in row.items()}
            sanitized_rows.append(sanitized_row)

        table_ref = f"{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )

        job = self.client.load_table_from_json(sanitized_rows, table_ref, job_config=job_config)
        job.result()

        return len(sanitized_rows)

    def insert_comment_sentiments(
        self,
        run_id: int,
        sentiments: list[dict],
        model_used: str,
        is_prod: bool = BIGQUERY_IS_PROD
    ) -> int:
        """Batch insert comment sentiments.

        Args:
            run_id: Pipeline run ID.
            sentiments: List of sentiment dictionaries with keys:
                - content_type, content_id, author_id, entity_name,
                  sentiment_score (original), smoothed_sentiment_score,
                  normalized_sentiment_score, source_name
            model_used: Model name.
            is_prod: True for production data, False for dev/migrated (defaults to config).

        Returns:
            int: Number of rows inserted.
        """
        if not sentiments:
            return 0

        query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{self.dataset_id}.pipeline_comment_sentiments`"
        result = self._query(query)
        next_id = result[0]['next_id']

        rows = []
        for i, s in enumerate(sentiments):
            rows.append({
                'id': next_id + i,
                'run_id': run_id,
                'content_type': s['content_type'],
                'content_id': s['content_id'],
                'author_id': s['author_id'],
                'entity_name': s['entity_name'],
                'sentiment_score': s.get('sentiment_score'),
                'smoothed_sentiment_score': s.get('smoothed_sentiment_score'),
                'normalized_sentiment_score': s.get('normalized_sentiment_score'),
                'model_used': model_used,
                'source_name': s.get('source_name'),
                'is_prod': is_prod,
                'created_at': datetime.now(timezone.utc).isoformat()
            })

        return self._insert_rows('pipeline_comment_sentiments', rows)

    def get_comment_sentiments_by_run(self, run_id: int) -> list[dict]:
        """Get all comment sentiments for a pipeline run.

        Args:
            run_id: Pipeline run ID.

        Returns:
            list[dict]: List of comment sentiment records.
        """
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.pipeline_comment_sentiments`
        WHERE run_id = {run_id}
        ORDER BY entity_name, sentiment_score DESC
        """
        return self._query(query)

    def insert_entity_summaries(
        self,
        run_id: int,
        summaries: list[dict],
        model_used: str,
        is_prod: bool = BIGQUERY_IS_PROD
    ) -> int:
        """Batch insert entity summaries.

        Args:
            run_id: Pipeline run ID.
            summaries: List of summary dictionaries with keys:
                - entity_name, avg_sentiment, sentiment_count,
                  sentiment_std, content_count, sentiment_summary
            model_used: Model name.
            is_prod: True for production data, False for dev/migrated (defaults to config).

        Returns:
            int: Number of rows inserted.
        """
        if not summaries:
            return 0

        query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{self.dataset_id}.pipeline_entity_summaries`"
        result = self._query(query)
        next_id = result[0]['next_id']

        rows = []
        for i, s in enumerate(summaries):
            rows.append({
                'id': next_id + i,
                'run_id': run_id,
                'entity_name': s['entity_name'],
                'avg_sentiment': s.get('avg_sentiment'),
                'sentiment_count': s.get('sentiment_count'),
                'sentiment_std': s.get('sentiment_std'),
                'content_count': s.get('content_count'),
                'sentiment_summary': s.get('sentiment_summary'),
                'model_used': model_used,
                'is_prod': is_prod,
                'created_at': datetime.now(timezone.utc).isoformat()
            })

        return self._insert_rows('pipeline_entity_summaries', rows)

    def get_entity_summaries_by_run(self, run_id: int) -> list[dict]:
        """Get all entity summaries for a pipeline run.

        Args:
            run_id: Pipeline run ID.

        Returns:
            list[dict]: List of entity summary records.
        """
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.pipeline_entity_summaries`
        WHERE run_id = {run_id}
        ORDER BY entity_name
        """
        return self._query(query)
    
    def get_latest_entity_summaries(self, min_sentiment_count: int = 2) -> list[dict]:
        """Get entity summaries from the latest completed run.

        Args:
            min_sentiment_count: Minimum number of sentiments required.

        Returns:
            list[dict]: List of entity summaries.
        """
        query = f"""
        SELECT pes.*
        FROM `{self.dataset_id}.pipeline_entity_summaries` pes
        WHERE pes.run_id = (
            SELECT MAX(run_id)
            FROM `{self.dataset_id}.pipeline_entity_summaries`
        )
        AND pes.sentiment_count >= {min_sentiment_count}
        ORDER BY pes.sentiment_count DESC
        """
        return self._query(query)

    def get_latest_run_info(self) -> Optional[dict]:
        """Get the latest completed pipeline run information.

        Returns:
            Optional[dict]: Dict with run_date and completed_at, or None if no runs found.
        """
        query = f"""
        SELECT 
            DATE(pr.completed_at) as run_date,
            pr.completed_at
        FROM `{self.dataset_id}.pipeline_entity_summaries` pes
        JOIN `{self.dataset_id}.pipeline_runs` pr ON pes.run_id = pr.id
        WHERE pr.status = 'completed'
        AND pr.completed_at IS NOT NULL
        AND pes.run_id = (SELECT MAX(run_id) FROM `{self.dataset_id}.pipeline_entity_summaries`)
        LIMIT 1
        """
        results = self._query(query)
        return results[0] if results else None
    
    def get_current_sentiment_data(self, min_sentiment_count: int = 2, source_filter: Optional[str] = None) -> list[dict]:
        """Get current sentiment data from the latest pipeline run.

        Args:
            min_sentiment_count: Minimum number of sentiments required (default: 2).
            source_filter: Optional filter for content type ('youtube' or 'facebook'). None = all sources.

        Returns:
            list[dict]: List of entity summaries with sentiment data, grouped by entity_name.
        """
        # Build source filter clause
        source_clause = ""
        if source_filter:
            if source_filter.lower() == 'youtube':
                source_clause = "AND pcs.content_type = 'youtube_video'"
            elif source_filter.lower() == 'facebook':
                source_clause = "AND pcs.content_type = 'facebook_post'"
        
        query = f"""
        SELECT 
            pcs.entity_name,
            AVG(pcs.normalized_sentiment_score) as avg_sentiment,
            COUNT(*) as mention_count,
            COUNT(DISTINCT pcs.content_id) as content_count,
            STDDEV(pcs.normalized_sentiment_score) as std_dev,
            ANY_VALUE(pes.sentiment_summary) as sentiment_summary
        FROM `{self.dataset_id}.pipeline_comment_sentiments` pcs
        LEFT JOIN `{self.dataset_id}.pipeline_entity_summaries` pes 
            ON pcs.entity_name = pes.entity_name 
            AND pcs.run_id = pes.run_id
        WHERE pcs.run_id = (SELECT MAX(run_id) FROM `{self.dataset_id}.pipeline_comment_sentiments`)
        {source_clause}
        GROUP BY pcs.entity_name
        HAVING COUNT(*) >= {min_sentiment_count}
        ORDER BY mention_count DESC
        """
        return self._query(query)
    
    def get_sentiment_trends_with_authors(self, min_sentiment_count: int = 2, source_filter: Optional[str] = None) -> list[dict]:
        """Get sentiment trends over time with cumulative unique author counts.

        Args:
            min_sentiment_count: Minimum number of sentiments required (default: 2).
            source_filter: Optional filter for content type ('youtube' or 'facebook'). None = all sources.

        Returns:
            list[dict]: List of daily sentiment aggregates with date, entity_name, avg_sentiment,
                mention_count, sentiment_summary, and cumulative_unique_authors.
        """
        # Build source filter clause
        source_clause = ""
        if source_filter:
            if source_filter.lower() == 'youtube':
                source_clause = "AND pcs.content_type = 'youtube_video'"
            elif source_filter.lower() == 'facebook':
                source_clause = "AND pcs.content_type = 'facebook_post'"
        
        query = f"""
        WITH base AS (
            SELECT 
                DATE(pr.completed_at) as date,
                pcs.entity_name,
                AVG(pcs.normalized_sentiment_score) as avg_sentiment,
                COUNT(*) as mention_count,
                ANY_VALUE(pes.sentiment_summary) as sentiment_summary
            FROM `{self.dataset_id}.pipeline_comment_sentiments` pcs
            JOIN `{self.dataset_id}.pipeline_runs` pr ON pcs.run_id = pr.id
            LEFT JOIN `{self.dataset_id}.pipeline_entity_summaries` pes 
                ON pcs.entity_name = pes.entity_name 
                AND pcs.run_id = pes.run_id
            WHERE pr.status = 'completed'
            AND pr.completed_at IS NOT NULL
            {source_clause}
            AND pcs.run_id = (
                SELECT MAX(pr2.id)
                FROM `{self.dataset_id}.pipeline_runs` pr2
                WHERE DATE(pr2.completed_at) = DATE(pr.completed_at)
                AND pr2.status = 'completed'
                AND pr2.completed_at IS NOT NULL
            )
            GROUP BY date, pcs.entity_name
            HAVING COUNT(*) >= {min_sentiment_count}
        )
        SELECT 
            date,
            entity_name,
            ANY_VALUE(avg_sentiment) as avg_sentiment,
            ANY_VALUE(mention_count) as mention_count,
            ANY_VALUE(sentiment_summary) as sentiment_summary,
            (
                SELECT COUNT(DISTINCT pcs2.author_id)
                FROM `{self.dataset_id}.pipeline_comment_sentiments` pcs2
                JOIN `{self.dataset_id}.pipeline_runs` pr2 ON pcs2.run_id = pr2.id
                WHERE pcs2.entity_name = base.entity_name
                AND DATE(pr2.completed_at) <= base.date
                AND pr2.status = 'completed'
                AND pr2.completed_at IS NOT NULL
                {source_clause.replace('pcs.', 'pcs2.')}
            ) as cumulative_unique_authors
        FROM base
        GROUP BY date, entity_name
        ORDER BY date DESC, mention_count DESC
        """
        return self._query(query)
    
    def get_all_comment_sentiments(self, limit: Optional[int] = None) -> list[dict]:
        """Get all comment sentiments, ordered by most recent.

        Args:
            limit: Optional limit on number of rows to return.

        Returns:
            list[dict]: List of all comment sentiment records.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.pipeline_comment_sentiments`
        ORDER BY created_at DESC
        {limit_clause}
        """
        return self._query(query)
    
    def get_all_entity_summaries(self, limit: Optional[int] = None) -> list[dict]:
        """Get all entity summaries, ordered by most recent.

        Args:
            limit: Optional limit on number of rows to return.

        Returns:
            list[dict]: List of all entity summary records.
        """
        limit_clause = f"LIMIT {limit}" if limit else ""
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.pipeline_entity_summaries`
        ORDER BY created_at DESC
        {limit_clause}
        """
        return self._query(query)

    
    def get_sentiment_trends(
        self,
        entity_names: Optional[list[str]] = None,
        days: int = 30
    ) -> list[dict]:
        """Get sentiment trends over time.

        Args:
            entity_names: Optional list of entity names to filter.
            days: Number of days to look back.

        Returns:
            list[dict]: List of daily aggregates with entity_name, date, avg_sentiment, etc.
        """
        entity_filter = ""
        if entity_names:
            entity_list = "', '".join(entity_names)
            entity_filter = f"AND pes.entity_name IN ('{entity_list}')"
        
        query = f"""
        SELECT 
            DATE(pr.completed_at) as date,
            pes.entity_name,
            pes.avg_sentiment,
            pes.sentiment_count,
            pes.content_count,
            pes.sentiment_summary
        FROM `{self.dataset_id}.pipeline_entity_summaries` pes
        JOIN `{self.dataset_id}.pipeline_runs` pr ON pes.run_id = pr.id
        WHERE pr.status = 'completed'
        AND pr.completed_at IS NOT NULL
        AND DATE(pr.completed_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        {entity_filter}
        ORDER BY DATE(pr.completed_at) DESC, pes.entity_name
        """
        return self._query(query)

    @staticmethod
    def _generate_author_id(author_name: Optional[str]) -> Optional[str]:
        """Generate anonymized author_id from author_name using SHA256 hash.

        Same author name always produces same author_id for tracking patterns
        while maintaining privacy. Uses first 16 characters (64 bits) for strong
        collision resistance while keeping IDs compact.

        Args:
            author_name: Author display name.

        Returns:
            Optional[str]: Hashed author ID or None if author_name is None.
        """
        if not author_name:
            return None
        return hashlib.sha256(author_name.encode('utf-8')).hexdigest()[:16]

    def insert_youtube_video(
        self,
        channel_id: str,
        video_id: str,
        video_title: str,
        video_url: str,
        channel_name: Optional[str] = None,
        publication_date: Optional[datetime] = None,
        pipeline_run_id: Optional[int] = None
    ) -> int:
        """Insert YouTube video, returns video ID.

        Always inserts a new row with the current pipeline_run_id to track across runs.

        Args:
            channel_id: YouTube channel ID string (e.g., "UChd1DEecCRlxaa0-hvPACCw").
            video_id: YouTube video ID.
            video_title: Video title.
            video_url: Video URL.
            channel_name: Human-readable channel name.
            publication_date: When the video was published.
            pipeline_run_id: Pipeline run ID.

        Returns:
            int: Database video ID.
        """
        query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{self.dataset_id}.youtube_videos`"
        result = list(self.client.query(query).result())
        db_video_id = result[0].next_id

        row = {
            'id': db_video_id,
            'channel_id': channel_id,
            'channel_name': channel_name,
            'video_id': video_id,
            'title': video_title,
            'video_url': video_url,
            'published_date': publication_date.isoformat() if publication_date else None,
            'scraped_at': datetime.now().isoformat(),
            'pipeline_run_id': pipeline_run_id
        }

        self._insert_rows('youtube_videos', [row])
        return db_video_id

    def insert_youtube_comments(
        self,
        video_id: int,
        comments: list[dict[str, Any]]
    ) -> int:
        """Batch insert YouTube video comments with author anonymization.

        Args:
            video_id: Database video ID.
            comments: List of comment dictionaries with keys:
                - author: Author display name (will be anonymized via hashing)
                - text: Comment text content
                - votes: Number of likes/votes
                - published_time: When comment was published
                - comment_id: YouTube's external comment ID
                - is_reply: Boolean indicating if this is a reply
                - parent_comment_id: External ID of parent comment (for replies)

        Returns:
            int: Number of comments inserted.
        """
        if not comments:
            return 0

        rows = []
        for comment in comments:
            rows.append({
                'external_comment_id': comment.get('comment_id'),
                'video_id': video_id,
                'author_id': self._generate_author_id(comment.get('author', 'Anonymous')),
                'comment_text': comment.get('text', ''),
                'votes': comment.get('votes', 0),
                'published_time': comment.get('published_time'),
                'is_reply': comment.get('is_reply', False),
                'parent_comment_id': comment.get('parent_comment_id')
            })

        self._insert_rows('youtube_comments', rows)
        return len(rows)

    def insert_facebook_post(
        self,
        page_id: str,
        post_id: str,
        post_message: str,
        post_url: str,
        page_name: Optional[str] = None,
        created_date: Optional[datetime] = None,
        pipeline_run_id: Optional[int] = None
    ) -> int:
        """Insert Facebook post, returns post ID.

        Always inserts a new row with the current pipeline_run_id to track across runs.

        Args:
            page_id: Facebook page ID string.
            post_id: Facebook post ID.
            post_message: Post message/content.
            post_url: Post permalink URL.
            page_name: Human-readable page name.
            created_date: When the post was created.
            pipeline_run_id: Pipeline run ID.

        Returns:
            int: Database post ID.
        """
        query = f"SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM `{self.dataset_id}.facebook_posts`"
        result = list(self.client.query(query).result())
        db_post_id = result[0].next_id

        row = {
            'id': db_post_id,
            'page_id': page_id,
            'page_name': page_name,
            'post_id': post_id,
            'message': post_message,
            'post_url': post_url,
            'created_date': created_date.isoformat() if created_date else None,
            'scraped_at': datetime.now().isoformat(),
            'pipeline_run_id': pipeline_run_id
        }

        self._insert_rows('facebook_posts', [row])
        return db_post_id

    def insert_facebook_comments(
        self,
        post_id: int,
        comments: list[dict[str, Any]]
    ) -> int:
        """Batch insert Facebook post comments with author anonymization.

        Args:
            post_id: Database post ID.
            comments: List of comment dictionaries with keys:
                - author: Author display name (will be anonymized via hashing)
                - text: Comment text content
                - votes: Number of likes/votes
                - published_time: When comment was published
                - comment_id: Facebook's external comment ID
                - is_reply: Boolean indicating if this is a reply
                - parent_comment_id: External ID of parent comment (for replies)

        Returns:
            int: Number of comments inserted.
        """
        if not comments:
            return 0

        rows = []
        for comment in comments:
            rows.append({
                'external_comment_id': comment.get('comment_id'),
                'post_id': post_id,
                'author_id': self._generate_author_id(comment.get('author', 'Anonymous')),
                'comment_text': comment.get('text', ''),
                'votes': comment.get('votes', 0),
                'published_time': comment.get('published_time'),
                'is_reply': comment.get('is_reply', False),
                'parent_comment_id': comment.get('parent_comment_id')
            })

        self._insert_rows('facebook_comments', rows)
        return len(rows)

    def get_content_with_comments(self, limit: int = 1000) -> list[dict]:
        """Get all content (YouTube videos, Facebook posts) with their comments.

        Returns unified structure for playground similarity matching.

        Args:
            limit: Maximum number of content-comment pairs to retrieve.

        Returns:
            list[dict]: List of dicts with:
                - content_id: str (unique identifier)
                - content_type: str ('youtube' or 'facebook')
                - title: str (video title or post text preview)
                - source_name: str (channel name or page name)
                - comment_text: str
        """
        query = f"""
        SELECT 
            CAST(v.id AS STRING) as content_id,
            'youtube' as content_type,
            v.title,
            v.channel_name as source_name,
            c.comment_text
        FROM `{self.dataset_id}.youtube_videos` v
        LEFT JOIN `{self.dataset_id}.youtube_comments` c ON v.id = c.video_id
        WHERE c.comment_text IS NOT NULL
        
        UNION ALL
        
        SELECT 
            CAST(p.id AS STRING) as content_id,
            'facebook' as content_type,
            SUBSTR(p.message, 1, 100) as title,
            p.page_name as source_name,
            c.comment_text
        FROM `{self.dataset_id}.facebook_posts` p
        LEFT JOIN `{self.dataset_id}.facebook_comments` c ON p.id = c.post_id
        WHERE c.comment_text IS NOT NULL
        
        LIMIT {limit}
        """
        return self._query(query)


def get_bigquery_manager() -> BigQueryManager:
    """Get BigQuery manager instance.
    
    Automatically detects if running in Streamlit and uses in-memory credentials
    if available, otherwise falls back to file-based credentials.

    Returns:
        BigQueryManager: Configured BigQuery manager instance.
    """
    # Check if running in Streamlit with in-memory credentials
    try:
        if hasattr(st, 'session_state') and 'gcp_credentials' in st.session_state:
            return BigQueryManager(creds_dict=st.session_state['gcp_credentials'])
    except (ImportError, RuntimeError):
        # Not in Streamlit context, use file-based credentials
        pass
    
    return BigQueryManager()
