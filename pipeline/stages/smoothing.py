"""Sentiment smoothing stage using KNN with embeddings."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.logger import get_logger

logger = get_logger("stages.smoothing")


def embed_comments(df_sentiments: pd.DataFrame, embedding_model: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """Generate embeddings for unique comment texts in the sentiment dataframe.
    
    Args:
        df_sentiments: DataFrame with sentiment data including comment_text column.
        embedding_model: Name of the sentence transformer model to use.
    
    Returns:
        pd.DataFrame: DataFrame with added embedding column containing embeddings.
    """
    model = SentenceTransformer(embedding_model)
    
    # Get unique comments to avoid redundant embeddings
    unique_comments = df_sentiments['comment_text'].unique()
    
    # Generate embeddings
    embeddings = model.encode(unique_comments, show_progress_bar=True, convert_to_numpy=True)
    
    # Create mapping from comment text to embedding
    comment_to_embedding = {comment: emb for comment, emb in zip(unique_comments, embeddings)}
    
    # Add embeddings to dataframe
    df_with_embeddings = df_sentiments.copy()
    df_with_embeddings['embedding'] = df_with_embeddings['comment_text'].map(comment_to_embedding)
    
    return df_with_embeddings


def smooth_sentiments_knn(
    df_sentiments: pd.DataFrame,
    k_neighbors: int = 5,
    min_neighbors: int = 1
) -> pd.DataFrame:
    """Apply KNN smoothing to sentiment scores within each entity and content group.
    
    For each comment's sentiment towards an entity:
    1. Find K nearest neighbor comments (by cosine similarity of embeddings) 
       that also mention the same entity AND belong to the same content item
    2. Weight each neighbor's sentiment by its cosine similarity
    3. Add the smoothed sentiment as smoothed_sentiment (keeps original sentiment)
    
    Args:
        df_sentiments: DataFrame with author_id, content_id, source_type,
            entity_name, sentiment, comment_text, embedding columns.
        k_neighbors: Number of nearest neighbors to use for smoothing.
        min_neighbors: Minimum number of neighbors required to apply smoothing.
    
    Returns:
        pd.DataFrame: DataFrame with added smoothed_sentiment column.
    """
    df_to_smooth = df_sentiments[df_sentiments['sentiment'].notna()].copy()
    
    if df_to_smooth.empty:
        logger.warning("No valid sentiments to smooth")
        df_result = df_sentiments.copy()
        df_result['smoothed_sentiment'] = df_result['sentiment']
        return df_result
    
    df_to_smooth['smoothed_sentiment'] = df_to_smooth['sentiment']
    
    # Group by entity AND content_id to smooth within each entity per news item
    groups = df_to_smooth.groupby(['entity_name', 'content_id'])
    total_groups = len(groups)
    logger.info(f"Smoothing sentiments for {total_groups} entity-content groups (k={k_neighbors})...")
    
    processed_groups = 0
    
    for (entity, content_id), group_df in groups:
        # Skip if not enough data points to smooth
        if len(group_df) < min_neighbors + 1:
            processed_groups += 1
            continue
        
        group_indices = group_df.index
        
        embeddings = np.vstack(group_df['embedding'].values)
        sentiments = group_df['sentiment'].values.astype(np.float64)
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # For each comment, find K nearest neighbors and compute weighted average
        for i, idx in enumerate(group_indices):
            # Get similarity scores for this comment
            similarities = similarity_matrix[i]
            
            # Get indices of K+1 most similar comments (including itself)
            top_k_indices = np.argsort(similarities)[::-1][:k_neighbors + 1]
            
            # Remove self from neighbors
            top_k_indices = top_k_indices[top_k_indices != i][:k_neighbors]
            
            if len(top_k_indices) < min_neighbors:
                continue
            
            # Get similarities and sentiments of neighbors
            neighbor_similarities = similarities[top_k_indices]
            neighbor_sentiments = sentiments[top_k_indices]
            
            # Compute weighted average (weight by cosine similarity)
            weights = neighbor_similarities + 1e-10  # Add small epsilon to avoid division by zero
            weighted_sentiment = np.sum(weights * neighbor_sentiments) / np.sum(weights)
            
            df_to_smooth.loc[idx, 'smoothed_sentiment'] = weighted_sentiment
        
        processed_groups += 1
        if processed_groups % 20 == 0 or processed_groups == total_groups:
            logger.info(f"Progress: {processed_groups}/{total_groups} groups processed ({processed_groups*100//total_groups}%)")
    
    logger.info(f"Smoothing complete: {len(df_to_smooth)} sentiments processed")
    
    return df_to_smooth


def smooth_sentiments(
    df_sentiments: pd.DataFrame,
    embedding_model: str = "all-MiniLM-L6-v2",
    k_neighbors: int = 5,
    min_neighbors: int = 1
) -> tuple[pd.DataFrame, dict[str, any]]:
    """Main smoothing pipeline: embed comments and apply KNN smoothing.
    
    Args:
        df_sentiments: DataFrame from sentiment analysis with author_id, content_id, 
            source_type, entity_name, sentiment, comment_text columns.
        embedding_model: Sentence transformer model name for embeddings.
        k_neighbors: Number of nearest neighbors for smoothing.
        min_neighbors: Minimum neighbors required to apply smoothing.
    
    Returns:
        tuple[pd.DataFrame, dict[str, any]]: DataFrame with smoothed sentiments and stats dict.
    """
    df_valid_for_stats = df_sentiments[df_sentiments['sentiment'].notna()]
    num_groups = len(df_valid_for_stats.groupby(['entity_name', 'content_id'])) if not df_valid_for_stats.empty else 0
    
    unique_author_entity = len(df_sentiments.groupby(['author_id', 'entity_name'])) if not df_sentiments.empty else 0
    
    stats = {
        'total_comments': unique_author_entity,
        'total_sentiments': len(df_sentiments),
        'valid_sentiments': df_sentiments['sentiment'].notna().sum(),
        'entity_content_groups': num_groups,
        'k_neighbors': k_neighbors,
        'embedding_model': embedding_model
    }
    
    # Step 1: Embed comments
    df_with_embeddings = embed_comments(df_sentiments, embedding_model)
    
    # Step 2: Apply KNN smoothing
    df_smoothed = smooth_sentiments_knn(
        df_with_embeddings,
        k_neighbors=k_neighbors,
        min_neighbors=min_neighbors
    )
    
    # Calculate smoothing statistics
    if stats['valid_sentiments'] > 0:
        original_sentiments = df_sentiments[df_sentiments['sentiment'].notna()]['sentiment'].values
        smoothed_sentiments_vals = df_smoothed[df_smoothed['smoothed_sentiment'].notna()]['smoothed_sentiment'].values
        
        if len(original_sentiments) == len(smoothed_sentiments_vals):
            sentiment_changes = np.abs(smoothed_sentiments_vals - original_sentiments)
            stats['avg_sentiment_change'] = float(np.mean(sentiment_changes))
            stats['max_sentiment_change'] = float(np.max(sentiment_changes))
            stats['min_sentiment_change'] = float(np.min(sentiment_changes))
    
    # Remove embedding column before returning (not needed for downstream stages)
    df_final = df_smoothed.drop(columns=['embedding'])
    
    return df_final, stats
