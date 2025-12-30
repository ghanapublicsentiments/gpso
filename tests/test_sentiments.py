"""Tests for sentiment analysis stage."""

from unittest.mock import MagicMock, patch

import pandas as pd

from pipeline.stages.sentiments import analyze_sentiments, create_sentiments_schema, sanitize_field_name


class TestSentimentHelpers:
    """Test sentiment helper functions."""

    def test_sanitize_field_name(self):
        """Test field name sanitization."""
        result = sanitize_field_name("John Doe", "e", {})
        assert result == "John_Doe"

    def test_create_schema(self):
        """Test schema creation."""
        schema, _, _ = create_sentiments_schema(["user1"], ["Economy"])
        assert schema is not None


class TestAnalyzeSentiments:
    """Test sentiment analysis."""

    @patch("pipeline.stages.sentiments.get_client")
    def test_analyze(self, mock_get_client):
        """Test analyze sentiments."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_get_client.return_value = mock_client

        df_entities = pd.DataFrame(
            {"content_id": ["n1"], "detected_key_players": [["Economy"]], "detected_key_issues": [[]]}
        )
        df_topics = pd.DataFrame({"content_id": ["n1"], "topic": ["Test"]})
        news_records = pd.DataFrame(
            {
                "content_id": ["n1"],
                "comments_by_author": [{"user1": [{"comment_id": "c1", "comment_text": "Test"}]}],
                "news_title": ["News"],
                "source_type": ["youtube"],
                "source_name": ["Channel"],
            }
        )

        result_df, usage = analyze_sentiments(df_entities, df_topics, news_records, "gpt-4o-mini")
        assert not result_df.empty
