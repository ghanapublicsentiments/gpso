"""Entity detection stage for news items."""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.entities import KEY_ISSUES, KEY_PLAYERS
from pipeline.logger import get_logger

logger = get_logger("stages.detection")


def detect_entities_in_news_item(
    news_record: dict,
    embedding_model: str = 'all-MiniLM-L6-v2',
    fuzzy_threshold: int = 80,
    embed_threshold_key_issues: float = 0.5,
    key_issue_chunk_size: int = 5,
    top_k_key_issues: int = 3,
    batch_embed_size: int = 1000
) -> dict[str, list[str]]:
    """Detect key players and key issues in a news item.
    
    Logic:
    1. Key Players: Fuzzy search for names in news content
    2. Key Issues: Fuzzy search + embedding similarity for top matches
    
    Args:
        news_record: Dictionary containing news item data (from group_comments_by_newsitem).
        embedding_model: SentenceTransformer model for semantic matching.
        fuzzy_threshold: Minimum fuzzy match score (0-100) for keyword matching.
        embed_threshold_key_issues: Minimum similarity for key issues embedding matches.
        key_issue_chunk_size: Words per chunk for key issues embedding.
        top_k_key_issues: Number of top key issues to return (for embed_match).
        batch_embed_size: Batch size for embedding operations.
    
    Returns:
        dict[str, list[str]]: Dictionary with detected_key_players and detected_key_issues.
    """
    model = SentenceTransformer(embedding_model)

    def create_word_chunks(text: str, chunk_size: int) -> list[str]:
        """Create fixed-size overlapping word chunks from text.
        
        Args:
            text: Input text to chunk.
            chunk_size: Number of words per chunk.
        
        Returns:
            list[str]: List of text chunks.
        """
        if not text:
            return []
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
        return chunks
    
    def fuzzy_match_entities(entities: list[str], text: str, threshold: int) -> list[str]:
        """Find entities that fuzzy match in the text.
        
        Args:
            entities: List of entity names to search for.
            text: Text to search in.
            threshold: Minimum fuzzy match score (0-100).
        
        Returns:
            list[str]: List of matched entities.
        """
        matched = []
        text_lower = text.lower()
        
        for entity in entities:
            entity_lower = entity.lower()
            score = fuzz.partial_ratio(entity_lower, text_lower)
            if score >= threshold:
                matched.append(entity)
        
        return matched
    
    def find_top_embedding_matches(
        candidates: list[str],
        text_chunks: list[str],
        threshold: float,
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Find top-k candidates that match text chunks via embeddings.
        
        Args:
            candidates: List of candidate entities.
            text_chunks: List of text chunks to compare against.
            threshold: Minimum similarity threshold.
            top_k: Number of top matches to return.
        
        Returns:
            list[tuple[str, float]]: List of (candidate_name, max_similarity) tuples.
        """
        if not text_chunks or not candidates:
            return []
        
        # Embed candidates
        candidate_embeddings = []
        for i in range(0, len(candidates), batch_embed_size):
            batch = candidates[i: i+batch_embed_size]
            response = model.encode(batch, show_progress_bar=False)
            candidate_embeddings.extend(response)
        
        # Embed text chunks
        chunk_embeddings = []
        for i in range(0, len(text_chunks), batch_embed_size):
            batch = text_chunks[i:i+batch_embed_size]
            response = model.encode(batch, show_progress_bar=False)
            chunk_embeddings.extend(response)
        
        candidate_embeddings = np.array(candidate_embeddings)
        chunk_embeddings = np.array(chunk_embeddings)
        
        # Calculate similarities
        results = []
        for idx, candidate in enumerate(candidates):
            candidate_embedding = candidate_embeddings[idx].reshape(1, -1)
            similarities = cosine_similarity(candidate_embedding, chunk_embeddings)[0]
            max_similarity = similarities.max()
            
            if max_similarity >= threshold:
                results.append((candidate, float(max_similarity)))
        
        # Sort by similarity (descending) and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    # Step 1: Prepare combined text from title and all comments
    title = news_record.get('news_title', '') or ''
    comments_by_author = news_record.get('comments_by_author', {})
    
    # Extract all comment texts from the comments_by_author structure
    comment_texts = []
    for _, comments in comments_by_author.items():
        for comment in comments:
            ct = comment.get('comment_text')
            if ct:
                comment_texts.append(ct)
    
    # Combine all text (filter out empty strings)
    text_parts = [title] + comment_texts
    combined_text = ' '.join(part for part in text_parts if part)
    
    # Step 2: Detect Key Players (all use keyword_match)
    keyword_players = [p for p, match_type in KEY_PLAYERS.items() if match_type == "keyword_match"]
    detected_key_players = fuzzy_match_entities(keyword_players, combined_text, fuzzy_threshold)
    
    # Step 3: Detect Key Issues
    detected_key_issues = []
    
    # 3a. Keyword match key issues
    keyword_issues = [i for i, match_type in KEY_ISSUES.items() if match_type == "keyword_match"]
    detected_key_issues.extend(fuzzy_match_entities(keyword_issues, combined_text, fuzzy_threshold))
    
    # 3b. Embed match key issues - create chunks and find top semantic matches
    embed_issues = [i for i, match_type in KEY_ISSUES.items() if match_type == "embed_match"]
    if embed_issues:
        # Create 5-word chunks
        chunks = create_word_chunks(combined_text, key_issue_chunk_size)
        if chunks:
            top_issues = find_top_embedding_matches(
                candidates=embed_issues,
                text_chunks=chunks,
                threshold=embed_threshold_key_issues,
                top_k=top_k_key_issues
            )
            detected_key_issues.extend([issue for issue, _ in top_issues])
    
    return {
        'detected_key_players': detected_key_players,
        'detected_key_issues': detected_key_issues
    }


def detect_entities_in_all_news_items(
    news_records: pd.DataFrame,
    embedding_model: str = 'all-MiniLM-L6-v2',
    fuzzy_threshold: int = 80,
    embed_threshold_key_issues: float = 0.5,
    key_issue_chunk_size: int = 5,
    top_k_key_issues: int = 3,
    batch_embed_size: int = 1000
) -> pd.DataFrame:
    """Detect key players and key issues across all news items.
    
    Args:
        news_records: DataFrame of news records (from process_latest_data).
        embedding_model: SentenceTransformer model for semantic matching.
        fuzzy_threshold: Minimum fuzzy match score (0-100) for keyword matching.
        embed_threshold_key_issues: Minimum similarity for key issues embedding matches.
        key_issue_chunk_size: Words per chunk for key issues embedding.
        top_k_key_issues: Number of top key issues to return (for embed_match).
        batch_embed_size: Batch size for embedding operations.
    
    Returns:
        pd.DataFrame: DataFrame with content_id, detected_key_players, detected_key_issues.
    """
    results = []
    
    for idx, record in news_records.iterrows():
        try:
            # Convert row to dict for the detection function
            record_dict = record.to_dict()
            
            detection_result = detect_entities_in_news_item(
                news_record=record_dict,
                embedding_model=embedding_model,
                fuzzy_threshold=fuzzy_threshold,
                embed_threshold_key_issues=embed_threshold_key_issues,
                key_issue_chunk_size=key_issue_chunk_size,
                top_k_key_issues=top_k_key_issues,
                batch_embed_size=batch_embed_size
            )
            
            results.append({
                'content_id': record.get('content_id'),
                'detected_key_players': detection_result['detected_key_players'],
                'detected_key_issues': detection_result['detected_key_issues']
            })
                
        except Exception:
            results.append({
                'content_id': record.get('content_id'),
                'detected_key_players': [],
                'detected_key_issues': []
            })
    
    df_results = pd.DataFrame(results)
    
    return df_results