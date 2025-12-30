"""Mapping stage utilities.

Distinct strategies are applied for subjects vs entities to improve recall and
precision:
1. Subjects (generated captions) -> multi-surface matching: title + each comment +
   fixed-size content word chunks (greedy exact caption match boost).
2. Entities (canonical list) -> adaptive overlapping chunks sized to entity word
   count with optional keyword fallback when semantic similarity is weak.

Returned mapping objects contain list of dicts with keys:
    record: original content item dict
    similarity: float similarity score (0-1)
    match_type: 'similarity' | 'exact_caption' | 'keyword'

These richer structures support downstream sentiment filtering and summary
generation with transparency of match rationale.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBED_FALLBACK_DIM = 384  # dimension assumed if embedding fails


def _batch_embed(texts: list[str], model: str, batch_size: int) -> list[List[float]]:
    """Embed texts in batches using SentenceTransformer.
    
    Args:
        texts: List of texts to embed.
        model: Model name (unused, hardcoded to all-MiniLM-L6-v2).
        batch_size: Batch size for encoding.
    
    Returns:
        list[List[float]]: List of embeddings.
    """
    embeddings: list[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            embeddings.extend(model.encode(batch, show_progress_bar=False))
        except Exception:
            embeddings.extend([[0]*EMBED_FALLBACK_DIM for _ in batch])
    return embeddings


def deduplicate_content_items(items: list[dict]) -> list[dict]:
    """Deduplicate news_article & youtube_post by title, merging comments.

    Combines duplicate titles' comments_by_author (unique by comment_id) and updates comment count.
    Keeps each youtube_video as unique.
    
    Args:
        items: List of content item dictionaries.
    
    Returns:
        list[dict]: Deduplicated list of content items.
    """
    videos: list[dict] = []
    articles_posts: dict[str, Dict] = {}
    for record in items:
        stype = record.get('source_type', '')
        if stype == 'youtube_video':
            videos.append(record)
            continue
        title = record.get('news_title', '') or record.get('article_title', '')
        if not title:
            articles_posts[f"{id(record)}"] = record
            continue
        existing = articles_posts.get(title)
        if not existing:
            articles_posts[title] = record
        else:
            # Merge comments_by_author from both records
            existing_cba = existing.get('comments_by_author', {})
            new_cba = record.get('comments_by_author', {})
            
            # Merge by author_id
            for author_id, comments in new_cba.items():
                if author_id not in existing_cba:
                    existing_cba[author_id] = comments
                else:
                    # Deduplicate comments by comment_text
                    existing_comment_ids = {c.get('comment_text'): c for c in existing_cba[author_id]}
                    for comment in comments:
                        comment_text = comment.get('comment_text')
                        if comment_text not in existing_comment_ids:
                            existing_cba[author_id].append(comment)
            
            existing['comments_by_author'] = existing_cba
            # Recalculate comment count
            existing['comment_count'] = sum(len(cmts) for cmts in existing_cba.values())
    return videos + list(articles_posts.values())

def map_subjects_to_items(
    subjects: list[str], 
    items: list[dict], 
    embedding_model: str, 
    similarity_threshold: float,
    batch_embed_size: int = 50, 
    content_chunk_words: int = 10,
    caption_map: dict[str, str] | None = None
) -> dict[str, List[Dict]]:
    """Map generated subject captions to content items.

    Strategy:
      1. Embed all subjects.
      2. For each item build surface texts: title, each comment text, fixed-size word chunks
         of news_content (content_chunk_words).
      3. Embed surface texts, compute cosine similarities, take max per item.
      4. If caption_map provided and item title's caption exactly equals subject -> add
         match with similarity=1.0 and match_type='exact_caption' (deduplicated by content_id).
    
    Args:
        subjects: List of subject captions.
        items: List of content item dictionaries.
        embedding_model: Model name for embeddings.
        similarity_threshold: Minimum similarity for matching.
        batch_embed_size: Batch size for embedding operations.
        content_chunk_words: Number of words per content chunk.
        caption_map: Optional mapping from title to caption for exact matching.
    
    Returns:
        dict[str, List[Dict]]: Mapping from subject to list of matched items.
    """
    subject_embeds = np.array(_batch_embed(subjects, embedding_model, batch_embed_size))
    item_surfaces: list[List[str]] = []
    for rec in items:
        texts: list[str] = []
        title = rec.get('news_title')
        if title:
            texts.append(title)
        # Iterate over comments_by_author structure
        comments_by_author = rec.get('comments_by_author', {})
        for author_id, comments in comments_by_author.items():
            for comment in comments:
                ct = comment.get('comment_text')
                if ct:
                    texts.append(ct)
        content = rec.get('news_content')
        if content:
            words = content.split()
            for i in range(0, len(words), content_chunk_words):
                chunk = words[i:i+content_chunk_words]
                if chunk:
                    texts.append(' '.join(chunk))
        item_surfaces.append(texts)
    
    # Embed all item texts per record
    item_embeds: list[np.ndarray] = []
    for texts in item_surfaces:
        emb_list = _batch_embed(texts, embedding_model, batch_embed_size) if texts else []
        item_embeds.append(np.array(emb_list))

    mapping: dict[str, List[Dict]] = {s: [] for s in subjects}
    for si, subject in enumerate(subjects):
        s_vec = subject_embeds[si].reshape(1, -1)
        seen_ids = set()
        # caption exact matches first
        if caption_map:
            for rec in items:
                title = rec.get('news_title')
                cid = rec.get('content_id')
                if not title or not cid or cid in seen_ids:
                    continue
                if title in caption_map and caption_map[title] == subject:
                    mapping[subject].append({'record': rec, 'similarity': 1.0, 'match_type': 'exact_caption'})
                    seen_ids.add(cid)
        for idx, emb_arr in enumerate(item_embeds):
            if emb_arr.size == 0:
                continue
            cid = items[idx].get('content_id')
            if cid in seen_ids:
                continue
            sims = cosine_similarity(s_vec, emb_arr)[0]
            max_sim = sims.max()
            if max_sim >= similarity_threshold:
                mapping[subject].append({'record': items[idx], 'similarity': float(max_sim), 'match_type': 'similarity'})
                seen_ids.add(cid)
        # deterministic order: similarity desc then content_id
        mapping[subject].sort(key=lambda x: (x['similarity'], x['record'].get('content_id')), reverse=True)
    return mapping

def map_entities_to_items(
    entities: list[str], 
    items: list[dict], 
    embedding_model: str, 
    similarity_threshold: float,
    batch_embed_size: int = 50, 
    overlap_words: int = 1, 
    keyword_fallback: bool = True,
    max_workers: int = 5
) -> dict[str, List[Dict]]:
    """Map canonical entities using adaptive overlapping content chunks and optional keyword fallback.

    For each unique entity word count w:
      1. Build overlapping chunks of size w+overlap_words across each item's combined text
         (title + content + comments). Stride = chunk_size - overlap_words.
      2. Embed all chunks once per word count group for reuse.
      3. Embed entities; compute max similarity per item.
      4. If similarity below threshold and keyword_fallback enabled, check keyword presence.

    keyword_fallback extracts non-stopword tokens (>2 chars) from entity and tests substring
    presence in combined item text.
    
    Args:
        entities: List of entity names.
        items: List of content item dictionaries.
        embedding_model: Model name for embeddings.
        similarity_threshold: Minimum similarity for matching.
        batch_embed_size: Batch size for embedding operations.
        overlap_words: Number of overlapping words in chunks.
        keyword_fallback: Whether to use keyword matching as fallback.
        max_workers: Number of parallel workers.
    
    Returns:
        dict[str, List[Dict]]: Mapping from entity to list of matched items.
    """
    stop_words = {'the','of','in','on','at','to','for','a','an','and','or','but','is','are','was','were','be','been','being','with','by','from'}

    def extract_keywords(e: str) -> set:
        return {w for w in e.lower().split() if w not in stop_words and len(w) > 2}

    combined_texts: list[str] = []
    for rec in items:
        parts: list[str] = []
        t = rec.get('news_title')
        if t: parts.append(t)
        c = rec.get('news_content')
        if c: parts.append(c)
        # Iterate over comments_by_author structure
        comments_by_author = rec.get('comments_by_author', {})
        for author_id, comments in comments_by_author.items():
            for comment in comments:
                tx = comment.get('comment_text')
                if tx: parts.append(tx)
        combined_texts.append(' '.join(parts))

    entity_embeds = np.array(_batch_embed(entities, embedding_model, batch_embed_size))
    # group entities by word count
    wc_groups: dict[int, List[int]] = {}
    for idx, e in enumerate(entities):
        wc = len(e.split()) or 1
        wc_groups.setdefault(wc, []).append(idx)

    def create_chunks(text: str, wc: int) -> list[str]:
        if not text:
            return []
        words = text.split()
        chunk_size = wc + overlap_words
        stride = max(1, chunk_size - overlap_words)
        chunks: list[str] = []
        for i in range(0, len(words), stride):
            chunk = words[i:i+chunk_size]
            if chunk:
                chunks.append(' '.join(chunk))
            if i + chunk_size >= len(words):
                break
        return chunks

    def process_wc(wc: int, entity_indices: list[int]) -> dict[str, List[Dict]]:
        """Process entities with given word count.
        
        Args:
            wc: Word count for chunk size.
            entity_indices: List of entity indices to process.
        
        Returns:
            dict[str, List[Dict]]: Mapping from entity to matched items.
        """
        # precompute all chunks for this word count
        all_chunks: list[str] = []
        chunk_to_item: list[int] = []
        for item_idx, text in enumerate(combined_texts):
            chs = create_chunks(text, wc)
            all_chunks.extend(chs)
            chunk_to_item.extend([item_idx]*len(chs))
        chunk_embeds = np.array(_batch_embed(all_chunks, embedding_model, batch_embed_size)) if all_chunks else np.empty((0,EMBED_FALLBACK_DIM))
        results: dict[str, List[Dict]] = {}
        for ent_idx in entity_indices:
            entity = entities[ent_idx]
            e_vec = entity_embeds[ent_idx].reshape(1,-1)
            sims = cosine_similarity(e_vec, chunk_embeds)[0] if chunk_embeds.size else np.array([])
            # compute max similarity per item
            item_best: dict[int, float] = {}
            for ci, item_idx in enumerate(chunk_to_item):
                sim = sims[ci]
                prev = item_best.get(item_idx)
                item_best[item_idx] = sim if prev is None or sim > prev else prev
            kw = extract_keywords(entity) if keyword_fallback else set()
            matches: list[dict] = []
            seen_ids = set()
            for item_idx, max_sim in item_best.items():
                rec = items[item_idx]
                cid = rec.get('content_id')
                if cid in seen_ids:
                    continue
                match_type = None
                if max_sim >= similarity_threshold:
                    match_type = 'similarity'
                elif keyword_fallback and kw and any(k in combined_texts[item_idx].lower() for k in kw):
                    match_type = 'keyword'
                if match_type:
                    matches.append({'record': rec, 'similarity': float(max_sim), 'match_type': match_type})
                    seen_ids.add(cid)
            matches.sort(key=lambda x: (x['match_type'] != 'similarity', x['similarity']), reverse=True)
            results[entity] = matches
        return results

    mapping: dict[str, List[Dict]] = {e: [] for e in entities}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_wc, wc, idxs) for wc, idxs in wc_groups.items()]
        for fut in as_completed(futures):
            partial = fut.result()
            for ent, lst in partial.items():
                mapping[ent] = lst
    return mapping
