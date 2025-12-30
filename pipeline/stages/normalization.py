"""Channel normalization stage for sentiment scores.

This stage normalizes sentiment scores per ``source_name`` (channel) using
empirical CDFs over recent history, then maps them to z-scores via the
standard normal inverse CDF.

Steps:

1. Read historical sentiment scores for the past N days from
   ``pipeline_comment_sentiments``.
2. For each ``source_name`` build an empirical CDF F_s based on the
   historical scores (and current batch scores).
3. For each row in ``df_sent`` compute the channel-specific quantile
   ``q = F_s(score)``.
4. Convert ``q`` to a z-score via ``z = norm.ppf(q)``.
5. Attach the z-score as ``normalized_sentiment`` and return the updated
   DataFrame.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import ecdf, norm

from database.bigquery_manager import get_bigquery_manager
from pipeline.logger import get_logger

logger = get_logger("stages.normalization")


@dataclass
class NormalizationStats:
    """Statistics for normalization operations."""
    
    total_rows: int
    rows_with_score: int
    rows_normalized: int
    channels_seen: int
    window_days: int


def _load_historical_sentiments(window_days: int = 30) -> pd.DataFrame:
    """Load historical sentiments from the pipeline table for a time window.

    Args:
        window_days: Number of days to look back for historical data.

    Returns:
        pd.DataFrame: DataFrame with source_name and smoothed_sentiment columns.
    """
    manager = get_bigquery_manager()
    client = manager.client
    dataset_id = manager.dataset_id
    
    cutoff_date = (datetime.now() - timedelta(days=window_days)).strftime('%Y-%m-%d')
    
    query = f"""
        SELECT source_name, smoothed_sentiment_score as smoothed_sentiment
        FROM `{dataset_id}.pipeline_comment_sentiments`
        WHERE created_at >= '{cutoff_date}'
          AND smoothed_sentiment_score IS NOT NULL
          AND source_name IS NOT NULL
    """
    
    df_hist = client.query(query).to_dataframe()
    return df_hist


def _build_channel_cdfs(
    df_hist: pd.DataFrame,
    df_current: pd.DataFrame,
) -> dict[str, ecdf]:
    """Build empirical CDF objects per channel using scipy.stats.ecdf.

    Pools historical scores with current-batch scores, then creates an
    ECDF object for each source_name.
    
    Args:
        df_hist: Historical sentiment data.
        df_current: Current batch sentiment data.
    
    Returns:
        dict[str, ecdf]: Mapping from source_name to ECDF object.
    """
    if df_hist.empty and df_current.empty:
        return {}

    cols = ["source_name", "smoothed_sentiment"]
    frames = []
    if not df_hist.empty:
        frames.append(df_hist[cols])
    if not df_current.empty:
        mask = df_current["source_name"].notna() & df_current["smoothed_sentiment"].notna()
        if mask.any():
            frames.append(df_current.loc[mask, cols])

    if not frames:
        return {}

    df_all = pd.concat(frames, ignore_index=True)

    cdfs: dict[str, ecdf] = {}
    for source_name, grp in df_all.groupby("source_name"):
        scores = grp["smoothed_sentiment"].values.astype(float)
        if scores.size == 0:
            continue
        cdfs[source_name] = ecdf(scores)

    return cdfs


def _apply_channel_normalization(
    df_sent: pd.DataFrame,
    channel_cdfs: dict[str, ecdf],
    normalization_strength: float = 0.8,
) -> tuple[pd.DataFrame, NormalizationStats]:
    """Apply per-channel ECDF normalization and attach z-scores.
    
    Uses smoothed_sentiment as input and creates normalized_sentiment as output.
    Preserves both sentiment and smoothed_sentiment columns.
    
    Args:
        df_sent: Sentiment DataFrame with smoothed_sentiment column.
        channel_cdfs: Dictionary of ECDF objects per channel.
        normalization_strength: Strength of normalization via tanh.
    
    Returns:
        tuple[pd.DataFrame, NormalizationStats]: Normalized DataFrame and stats.
    """
    df = df_sent.copy()

    total_rows = len(df)
    mask_score = df["smoothed_sentiment"].notna()
    rows_with_score = int(mask_score.sum())

    z_scores = np.full(total_rows, np.nan, dtype=float)

    for idx, row in df[mask_score].iterrows():
        src = row.get("source_name")
        score = row["smoothed_sentiment"]

        if src is None or pd.isna(src) or src not in channel_cdfs:
            continue

        cdf_func = channel_cdfs[src]
        q = cdf_func.cdf.evaluate(float(score))
        z = norm.ppf(q)
        z_scores[df.index.get_loc(idx)] = z

    df["normalized_sentiment"] = np.tanh(normalization_strength * z_scores)

    stats = NormalizationStats(
        total_rows=total_rows,
        rows_with_score=rows_with_score,
        rows_normalized=int(np.isfinite(z_scores).sum()),
        channels_seen=len(channel_cdfs),
        window_days=30,
    )

    return df, stats


def normalize_channels(
	df_sent: pd.DataFrame,
	window_days: int = 30,
	normalization_strength: float = 0.8,
) -> tuple[pd.DataFrame, dict[str, float]]:
	"""Normalize sentiment scores per channel using historical ECDFs.
	
	Uses smoothed_sentiment as input and creates normalized_sentiment as output.
	Preserves both sentiment and smoothed_sentiment columns.

	Args:
		df_sent: DataFrame after smoothing; must include columns
			``smoothed_sentiment`` and ``source_name``.
		window_days: Lookback window (in days) for historical scores.
		normalization_strength: Strength of normalization (default 0.8).

	Returns:
		(df_with_normalized, stats_dict)
	"""

	if df_sent.empty:
		return df_sent, {
			"total_rows": 0,
			"rows_with_score": 0,
			"rows_normalized": 0,
			"channels_seen": 0,
			"window_days": window_days,
		}

	if "source_name" not in df_sent.columns:
		return df_sent, {
			"total_rows": len(df_sent),
			"rows_with_score": int(df_sent["smoothed_sentiment"].notna().sum()) if "smoothed_sentiment" in df_sent.columns else 0,
			"rows_normalized": 0,
			"channels_seen": 0,
			"window_days": window_days,
		}

	# 1) Load historical sentiments
	df_hist = _load_historical_sentiments(window_days=window_days)

	# 2) Build per-channel CDFs using historical + current batch data
	channel_cdfs = _build_channel_cdfs(df_hist, df_sent)

	# 3) Apply normalization to current df_sent
	df_norm, stats = _apply_channel_normalization(df_sent, channel_cdfs, normalization_strength=normalization_strength)

	return df_norm, {
		"total_rows": stats.total_rows,
		"rows_with_score": stats.rows_with_score,
		"rows_normalized": stats.rows_normalized,
		"channels_seen": stats.channels_seen,
		"window_days": stats.window_days,
	}

