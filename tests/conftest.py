"""Pytest configuration and shared fixtures for GPSO tests.

This module provides mock databases, sample data, and utility fixtures
used across all test modules.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_bigquery_manager():
    """Mock BigQuery manager for testing database operations."""
    manager = MagicMock()
    manager.dataset_id = "test_dataset"
    manager.client = MagicMock()

    # Mock query results
    def mock_query(query_string):
        result = MagicMock()
        result.result.return_value = []
        result.to_dataframe.return_value = pd.DataFrame()
        return result

    manager.client.query = mock_query
    manager._insert_rows = MagicMock()

    return manager


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing LLM calls."""
    client = MagicMock()

    # Mock structured output
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.parsed = {}
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    client.beta.chat.completions.parse.return_value = mock_response
    client.chat.completions.create.return_value = mock_response

    return client


@pytest.fixture
def sample_comments():
    """Sample comment data for testing."""
    return [
        {
            "comment_id": "c1",
            "author_id": "user1",
            "comment_text": "This is a positive comment about the economy.",
            "created_at": "2024-01-01T10:00:00Z",
        },
        {
            "comment_id": "c2",
            "author_id": "user1",
            "comment_text": "Healthcare improvements are needed urgently.",
            "created_at": "2024-01-01T10:05:00Z",
        },
        {
            "comment_id": "c3",
            "author_id": "user2",
            "comment_text": "The president is doing a great job on security.",
            "created_at": "2024-01-01T10:10:00Z",
        },
        {
            "comment_id": "c4",
            "author_id": "user3",
            "comment_text": "Corruption is still a major problem in the country.",
            "created_at": "2024-01-01T10:15:00Z",
        },
    ]


@pytest.fixture
def sample_news_items():
    """Sample news item data for testing."""
    return [
        {
            "content_id": "n1",
            "source_name": "Channel A",
            "source_type": "youtube",
            "title": "Economic Update",
            "description": "Latest updates on Ghana's economy",
            "published_at": "2024-01-01T09:00:00Z",
        },
        {
            "content_id": "n2",
            "source_name": "Channel B",
            "source_type": "youtube",
            "title": "Healthcare Crisis",
            "description": "Discussion on healthcare challenges",
            "published_at": "2024-01-01T09:30:00Z",
        },
    ]


@pytest.fixture
def sample_entities():
    """Sample entity lists for testing."""
    return {
        "key_players": ["President Mahama", "Akufo-Addo", "Bawumia"],
        "key_issues": ["Economy", "Healthcare", "Corruption"],
    }


@pytest.fixture
def sample_sentiment_df():
    """Sample sentiment DataFrame for testing."""
    return pd.DataFrame(
        {
            "comment_id": ["c1", "c2", "c3", "c4"],
            "author_id": ["user1", "user1", "user2", "user3"],
            "content_id": ["n1", "n1", "n2", "n2"],
            "source_name": ["Channel A", "Channel A", "Channel B", "Channel B"],
            "source_type": ["youtube", "youtube", "youtube", "youtube"],
            "entity_name": ["Economy", "Healthcare", "President Mahama", "Corruption"],
            "sentiment": [0.7, -0.3, 0.8, -0.6],
            "comment_text": [
                "This is a positive comment about the economy.",
                "Healthcare improvements are needed urgently.",
                "The president is doing a great job on security.",
                "Corruption is still a major problem in the country.",
            ],
            "created_at": pd.to_datetime(
                [
                    "2024-01-01T10:00:00Z",
                    "2024-01-01T10:05:00Z",
                    "2024-01-01T10:10:00Z",
                    "2024-01-01T10:15:00Z",
                ]
            ),
        }
    )


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    # Simple 384-dimensional embeddings (matching all-MiniLM-L6-v2)
    np.random.seed(42)
    return np.random.randn(4, 384)


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model for testing."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        np.random.seed(42)

        # Mock encode to return random embeddings
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.randn(len(texts), 384)

        mock_model.encode = mock_encode
        mock_st.return_value = mock_model

        yield mock_st


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch("pipeline.logger.get_logger") as mock_get_logger:
        logger = MagicMock()
        mock_get_logger.return_value = logger
        yield logger


@pytest.fixture
def sample_entity_summaries():
    """Sample entity summaries for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "run_id": [1, 1, 1],
            "entity_name": ["Economy", "Healthcare", "President Mahama"],
            "avg_sentiment": [0.65, -0.25, 0.75],
            "sentiment_count": [120, 85, 95],
            "sentiment_std": [0.25, 0.30, 0.20],
            "content_count": [15, 12, 18],
            "sentiment_summary": [
                "Overall positive sentiment about economic improvements",
                "Mixed reactions with concerns about healthcare access",
                "Strong positive support for the president's security policies",
            ],
            "model_used": ["gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
        }
    )


@pytest.fixture
def sample_plotly_data():
    """Sample data for creating Plotly charts."""
    return {
        "x_values": ["Entity A", "Entity B", "Entity C"],
        "y_values": [0.5, -0.2, 0.8],
        "labels": ["Positive", "Negative", "Very Positive"],
    }


class MockBigQueryResult:
    """Mock BigQuery query result for testing."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __iter__(self):
        return iter([type("Row", (), row) for row in self.data])

    def result(self):
        return self

    def to_dataframe(self):
        return pd.DataFrame(self.data) if self.data else pd.DataFrame()


@pytest.fixture
def mock_bigquery_result():
    """Factory for creating mock BigQuery results."""
    return MockBigQueryResult
