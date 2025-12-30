"""Tests for database operations."""

from unittest.mock import MagicMock, patch

from pipeline.db import (
    complete_pipeline_run,
    init_pipeline_run,
    persist_comment_sentiments,
)


class TestPipelineRunOps:
    """Test suite for pipeline run operations."""

    @patch("pipeline.db.get_bigquery_manager")
    def test_init_pipeline_run(self, mock_get_manager, mock_bigquery_result):
        """Test initialization of pipeline run."""
        manager = MagicMock()
        manager.dataset_id = "test_dataset"
        mock_result = mock_bigquery_result([{"next_id": 1}])
        manager.client.query.return_value = mock_result
        mock_get_manager.return_value = manager

        run_id = init_pipeline_run()
        assert run_id == 1


class TestCompletePipelineRun:
    """Test suite for completing pipeline runs."""

    @patch("pipeline.db.get_bigquery_manager")
    @patch("pipeline.db.PipelineConfig")
    def test_complete_run(self, mock_config_class, mock_get_manager):
        """Test completing a pipeline run."""
        mock_config = MagicMock()
        mock_config.caption.model = "gpt-4o"
        mock_config.to_dict.return_value = {}
        mock_config_class.return_value = mock_config

        manager = MagicMock()
        mock_get_manager.return_value = manager

        token_stats = {"total_input": 6800, "total_output": 2800}
        complete_pipeline_run(run_id=1, token_stats=token_stats, status="completed")

        assert manager._insert_rows.called


class TestPersistCommentSentiments:
    """Test suite for persisting comment sentiments."""

    @patch("pipeline.db.get_bigquery_manager")
    def test_persist_sentiments(self, mock_get_manager, sample_sentiment_df):
        """Test persisting comment sentiments to BigQuery."""
        manager = MagicMock()
        manager.insert_comment_sentiments.return_value = 4
        mock_get_manager.return_value = manager

        result = persist_comment_sentiments(run_id=1, df=sample_sentiment_df, model_used="gpt-4o-mini")

        assert result == 4
        assert manager.insert_comment_sentiments.called
