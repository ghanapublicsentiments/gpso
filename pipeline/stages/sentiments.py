"""Comment-level sentiment analysis stage."""

import re
import threading
from typing import Optional

import pandas as pd
from pydantic import Field, create_model

from config import get_client
from pipeline.logger import get_logger
from pipeline.utils import RateLimiter, process_in_parallel

logger = get_logger("stages.sentiments")


def sanitize_field_name(name: str, prefix: str, existing_fields: dict) -> str:
    """Sanitize a name to be a valid Pydantic field name and ensure uniqueness.
    
    Args:
        name: Original name to sanitize.
        prefix: Prefix to add if name starts with a digit (e.g., 'e_' for entities, 'a_' for authors).
        existing_fields: Dictionary of existing field names to check for uniqueness.
    
    Returns:
        str: Sanitized, unique field name.
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    
    # Add prefix if it starts with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f'{prefix}_{sanitized}'
    
    # Ensure uniqueness
    original = sanitized
    counter = 1
    while sanitized in existing_fields:
        sanitized = f"{original}_{counter}"
        counter += 1
    
    return sanitized


def create_sentiments_schema(author_list: list[str], entity_list: list[str]):
    """Create a nested Pydantic schema where each author has sentiment fields for each entity.
    
    Args:
        author_list: List of author IDs.
        entity_list: List of entity names (combined key players, key issues, major news subject).
    
    Returns:
        tuple: (Pydantic model class, field_to_author dict, field_to_entity dict).
    """
    entity_fields = {}
    field_to_entity = {}
    
    for entity in entity_list:
        sanitized = sanitize_field_name(entity, 'e', entity_fields)
        entity_fields[sanitized] = (Optional[float], Field(None, ge=-1, le=1))
        field_to_entity[sanitized] = entity
    
    EntitySentiments = create_model('EntitySentiments', **entity_fields)
    
    author_fields = {}
    field_to_author = {}
    
    for author_id in author_list:
        sanitized = sanitize_field_name(author_id, 'a', author_fields)
        author_fields[sanitized] = (EntitySentiments, Field(...))
        field_to_author[sanitized] = str(author_id)
    
    Sentiments = create_model('Sentiments', **author_fields)
    
    return Sentiments, field_to_author, field_to_entity


def build_prompt(comments_by_author: dict, entity_list: list[str], news_item: Optional[str] = None):
    """Generate the prompt for sentiment analysis of public comments.
    
    Args:
        comments_by_author: Dictionary mapping author_id -> list of comment objects.
        entity_list: List of entity names to analyze sentiment for.
        news_item: Optional news item string for context.
    
    Returns:
        str: Formatted prompt for sentiment analysis.
    """
    num_comments = sum(len(comments) for comments in comments_by_author.values())
    num_entities = len(entity_list)
    entities_str = ', '.join(entity_list)
    
    prompt = f"""
        You are an expert Ghanaian news analyst specializing in objective public sentiment analysis in news items.

        Your task is to analyze {num_comments} public comments associated with a news item 
        and provide sentiment scores for each author towards the {num_entities} entities.

        ENTITIES:
        {entities_str}

        STEP 1: UNDERSTAND THE CONTEXT
        • Analyze the comments as a whole in chronological order to understand the conversation context and dynamics.
        • Consider any provided news item for additional context.

        STEP 2: ANALYZE COMMENTS FOR SENTIMENT
        For each of the {num_comments} comments (identified by author_id):
        • *Consider the comment very carefully to determine if each of the {num_entities} entities is mentioned*
        • Score sentiment towards EACH entity from -1 (very negative) to +1 (very positive)
            • Set score to null if the entity is NOT mentioned in the comment
        • Base scores ONLY on what's expressed in the comment, not on any assumptions as your factual knowledge may be outdated.
            For example, do not assume current political appointments or recent events unless explicitly mentioned in the comment.

        
        CRITICAL RULES:
        ✗ NEVER base sentiments on the news item - it may contain reporting bias
        ✗ ONLY analyze what commenters explicitly express
        ✗ Return null for all entities if the comment contains ads, spam, irrelevant or inappropriate content
        ✓ Consider Ghanaian cultural context and local nuances
        ✓ Each sentiment score must be objective and independent, and not relative to other entities or comments

        IMPORTANT TO NOTE:
        • A comment may be a reply to a parent or preceding comment. Consider the parent or preceding comment for context,
            but analyze the current comment on its own merit.
        • Some comments may not mention any entities - in such cases, return null for all entities.
        • If an entity is mentioned but no clear sentiment is expressed, return a neutral score of 0 for that entity.
        • Instead of their name, some authors use the author name field to preface their main comment.

        PUBLIC COMMENTS:
        {comments_by_author}

        NEWS ITEM FOR CONTEXT:
        {news_item if news_item else 'N/A'}

        OUTPUT FORMAT:
        Return the sentiment for each entity for each author_id.
        Structure: For each author_id, provide sentiment scores for all {num_entities} entities.
    """.strip()
    return prompt

def analyze_sentiments(
    df_entities: pd.DataFrame,
    df_topics: pd.DataFrame,
    news_records: pd.DataFrame, 
    model: str, 
    max_workers: int=2, 
    rate_limit: int=10
) -> tuple[pd.DataFrame, dict[str,int]]:
    """Analyze sentiments for detected entities in news items.
    
    Args:
        df_entities: DataFrame with content_id, detected_key_players, detected_key_issues.
        df_topics: DataFrame with content_id, topic (hot topic name).
        news_records: DataFrame of news records (from process_latest_data).
        model: LLM model name for sentiment analysis.
        max_workers: Number of parallel workers.
        rate_limit: Maximum API requests per minute.
    
    Returns:
        tuple[pd.DataFrame, dict[str, int]]: DataFrame with sentiment results, token usage dict.
    """
    client = get_client(model)
    rate = RateLimiter(max_requests_per_minute=rate_limit)
    all_results = []
    usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    lock = threading.Lock()
    
    # Create a mapping from content_id to news_record (convert DataFrame to dict)
    content_id_to_record = news_records.set_index('content_id').to_dict('index')
    
    # Filter df_entities to only items with comments
    tasks_to_process = []
    for _, row in df_entities.iterrows():
        content_id = row['content_id']
        record = content_id_to_record.get(content_id)
        if not record:
            continue
        
        comments_by_author = record.get('comments_by_author', {})
        if not comments_by_author:
            continue
        
        # Combine all detected entities into one list
        entity_list = []
        
        # Add key players
        if row['detected_key_players']:
            entity_list.extend(row['detected_key_players'])
        
        # Add key issues
        if row['detected_key_issues']:
            entity_list.extend(row['detected_key_issues'])
        
        # Add hot topics if this content_id is associated with any
        content_topics = df_topics[df_topics['content_id'] == content_id]['topic'].tolist()
        entity_list.extend(content_topics)
        
        # Skip if no entities detected
        if not entity_list:
            continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entity_list:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)
        
        tasks_to_process.append({
            'content_id': content_id,
            'record': record,
            'entities': unique_entities,
            'source_type': record.get('source_type', ''),
            'source_name': record.get('source_name', '')
        })
    
    total_tasks = len(tasks_to_process)
    logger.info(f"Analyzing sentiments for {total_tasks} content items...")
    
    def task(task_data: dict) -> list[dict]:
        """Process a single news item with its detected entities."""
        record = task_data['record']
        entities = task_data['entities']
        content_id = task_data['content_id']
        source_type = task_data['source_type']
        source_name = task_data['source_name']

        comments_by_author = record.get('comments_by_author', {})
        if not comments_by_author:
            return None
        
        authors = list(comments_by_author.keys())
        
        # Create schema with both authors and entities
        Sentiments, field_to_author, field_to_entity = create_sentiments_schema(authors, entities)
        
        # Build prompt with entity list
        prompt = build_prompt(comments_by_author, entities, {'title': record.get('news_title', '')})
        
        rate.wait_if_needed()
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                response_format=Sentiments,
            )
            
            with lock:
                usage['input_tokens'] += getattr(completion.usage, 'prompt_tokens', 0)
                usage['output_tokens'] += getattr(completion.usage, 'completion_tokens', 0)
                usage['total_tokens'] += getattr(completion.usage, 'total_tokens', 0)
            
            parsed = completion.choices[0].message.parsed
            
            # Reverse mappings for easier lookup
            rev_author = {v: k for k, v in field_to_author.items()}
            rev_entity = {v: k for k, v in field_to_entity.items()}
            
            # Extract results: for each author, get sentiments for each entity
            out = []
            for author_id in authors:
                author_field = rev_author.get(str(author_id))
                if not author_field:
                    continue
                
                author_sentiments = getattr(parsed, author_field, None)
                if not author_sentiments:
                    continue
                
                # Get comment texts for this author from comments_by_author
                author_comments = comments_by_author.get(author_id, [])
                # Combine all comment texts for this author
                comment_text = ' | '.join([c.get('comment_text', '') for c in author_comments])
                
                # Extract sentiment for each entity
                for entity in entities:
                    entity_field = rev_entity.get(entity)
                    if not entity_field:
                        continue
                    
                    sentiment_score = getattr(author_sentiments, entity_field, None)
                    
                    out.append({
                        'author_id': author_id,
                        'content_id': content_id,
                        'source_type': source_type,
                        'source_name': source_name,
                        'entity_name': entity,
                        'sentiment': sentiment_score,
                        'comment_text': comment_text
                    })
            
            return out
            
        except Exception:
            return None
    
    # Process tasks in parallel using utility function
    all_results = []
    results = process_in_parallel(
        tasks_to_process,
        task,
        max_workers=max_workers,
        progress_name="content items",
        progress_interval=10
    )
    
    # Flatten results (each task returns a list of dicts)
    for res in results:
        if res:
            all_results.extend(res)
    
    # Convert results to DataFrame
    df = pd.DataFrame(all_results)
    logger.info(f"Sentiment analysis complete: {len(df)} sentiment records generated")
    
    return df, usage
