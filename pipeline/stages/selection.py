"""Article selection stage."""

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def select_top_distinct_titles(
    df: pd.DataFrame,
    title_column: str='news_title',
    count_column: str='news_comment_count',
    n_select: int=10,
    embedding_model: str='all-minilm',
    similarity_threshold: float=0.8,
    batch_embed_size: int=50
) -> pd.DataFrame:
    """Select top N distinct news titles based on comment count and semantic similarity.
    
    Args:
        df: DataFrame with news articles.
        title_column: Name of the title column.
        count_column: Name of the comment count column.
        n_select: Number of articles to select.
        embedding_model: Ollama embedding model name.
        similarity_threshold: Maximum similarity between selected articles.
        batch_embed_size: Batch size for embedding operations.
    
    Returns:
        pd.DataFrame: Selected articles with selection_order column.
    """
    df_sorted = df.sort_values(by=count_column, ascending=False).reset_index(drop=True)
    df_candidates = df_sorted.copy()
    titles = df_candidates[title_column].tolist()
    
    all_embeddings = []
    for i in range(0, len(titles), batch_embed_size):
        batch = titles[i:i+batch_embed_size]
        try:
            response = ollama.embed(model=embedding_model, input=batch)
            all_embeddings.extend(response['embeddings'])
        except Exception:
            all_embeddings.extend([[0]*768 for _ in batch])
    
    embeddings = np.array(all_embeddings)
    selected_indices = []
    selected_embeddings = []
    
    for idx in range(len(df_candidates)):
        if len(selected_indices) >= n_select:
            break
        
        current_embedding = embeddings[idx]
        
        if not selected_embeddings:
            selected_indices.append(idx)
            selected_embeddings.append(current_embedding)
            continue
        
        sims = cosine_similarity([current_embedding], selected_embeddings)[0]
        if sims.max() < similarity_threshold:
            selected_indices.append(idx)
            selected_embeddings.append(current_embedding)
    
    selected_df = df_candidates.iloc[selected_indices].copy()
    selected_df['selection_order'] = range(1, len(selected_indices)+1)
    return selected_df
