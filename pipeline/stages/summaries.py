"""Entity-level sentiment summary stage."""

import pandas as pd
from pydantic import BaseModel, Field

from config import get_client
from pipeline.logger import get_logger
from pipeline.utils import RateLimiter, process_in_parallel

logger = get_logger("stages.summaries")


def build_summary_prompt(entity: str, avg_sentiment: float, records: list[dict]):
    """Generate the prompt for sentiment summary of an entity in public comments.

    Args:
        entity: Entity name to analyze.
        avg_sentiment: Average sentiment score for the entity.
        records: List of dictionaries containing news records with comments.
    
    Returns:
        str: Formatted prompt for summary generation.
    """
    prompt = f"""
        You are an expert Ghanaian news analyst specializing in objective public sentiment analysis.

        You're given a set of news records, each of which contains multiple public comments.

        Your task is to analyze the comments and provide a reason for the average sentiment score of {avg_sentiment} for the following entity: {entity}.

        STEP 1: UNDERSTAND THE CONTEXT FOR EACH NEWS RECORD
        • Analyze the comments in each news record as a whole in chronological order to understand the conversation context and dynamics.
        • Consider any provided news item for additional context.

        STEP 2: ANALYZE COMMENTS FOR SENTIMENT ACROSS ALL RECORDS
        • Consider the comments very carefully and provide reasoning (no more than 2 sentences) for the average sentiment score of {avg_sentiment} on a scale of -0.89 (very negative) to +0.89 (very positive) for the entity.
        • Base your reasoning ONLY on what's expressed in the comment, not on any assumptions as your factual knowledge may be outdated.
            For example, do not assume current political appointments or recent events unless explicitly mentioned in the comment.
        • Focus only on the substantive reasons for the sentiment, not the general direction of the sentiment.

        CRITICAL RULES:
        ✗ NEVER base your reasoning on the news item - it may contain reporting bias
        ✗ NEVER base your reasoning on any other entity in the comments or news item, other than the specified entity
        ✗ NEVER quote any comment verbatim in your reasoning, especially since they may contain inappropriate content
        ✗ Comments that contain ads, spam, irrelevant or inappropriate content should be ignored and excluded from reasoning
        ✓ ONLY analyze what commenters explicitly express
        ✓ In your reasoning, go straight to actual points raised in the comments.
        ✓ Consider Ghanaian cultural context and local nuances

        IMPORTANT TO NOTE:
        • Individual comments may reference other comments - analyze each comment on its own merit.
        • Instead of their name, some authors use the author name field to preface their main comment.

        *ENTITY: {entity}*

        *AVERAGE SENTIMENT SCORE: {avg_sentiment}*

        NEWS RECORDS:
        {records}

        OUTPUT FORMAT:
        Return a two-sentence reasoning for the average sentiment score.
    """.strip()
    return prompt


class SentimentReasoning(BaseModel):
    """Model for sentiment reasoning extraction."""
    
    reasoning: str = Field(...)


def generate_summaries(
    df_sentiments: pd.DataFrame, 
    content_items: pd.DataFrame, 
    model: str, 
    max_workers: int=2, 
    rate_limit: int=10
) -> tuple[pd.DataFrame, dict[str,int]]:
    """Generate sentiment summaries for all entities.
    
    Args:
        df_sentiments: DataFrame with sentiment data.
        content_items: DataFrame with news records and comments.
        model: LLM model name for summary generation.
        max_workers: Number of parallel workers.
        rate_limit: Maximum API requests per minute.
    
    Returns:
        tuple[pd.DataFrame, dict[str,int]]: Summary DataFrame and token usage.
    """
    client = get_client(model)
    limiter = RateLimiter(rate_limit)
    usage = {'input_tokens':0,'output_tokens':0,'total_tokens':0}
    entities = df_sentiments['entity_name'].unique()
    
    def entity_records(entity: str) -> list[dict]:
        """Get content items with full comments_by_author structure for this entity.
        
        Args:
            entity: Entity name to filter by.
        
        Returns:
            list[dict]: List of content records with filtered comments.
        """
        df_e = df_sentiments[(df_sentiments['entity_name']==entity) & (df_sentiments['normalized_sentiment'].notna())]

        # Get set of (content_id, author_id) pairs that mentioned this entity
        entity_authors = set(zip(df_e['content_id'], df_e['author_id']))
        content_ids = set(df_e['content_id'].unique())
        
        subset = []
        for _, r in content_items.iterrows():
            content_id = r.get('content_id')
            if content_id in content_ids:
                # Filter comments_by_author to only include authors who mentioned this entity
                comments_by_author = r.get("comments_by_author", {})
                filtered_cba = {}
                for author_id, comments in comments_by_author.items():
                    # Include this author's comments if they mentioned the entity in this content
                    if (content_id, author_id) in entity_authors:
                        filtered_cba[author_id] = comments
                
                if filtered_cba:
                    r_copy = r.to_dict()
                    r_copy["comments_by_author"] = filtered_cba
                    subset.append(r_copy)
        return subset
    
    def task(entity: str):
        df_e = df_sentiments[(df_sentiments['entity_name']==entity) & (df_sentiments['sentiment'].notna())]
        if df_e.empty:
            return None
        avg_sent = df_e['normalized_sentiment'].mean()
        prompt = build_summary_prompt(entity, avg_sent, entity_records(entity))
        limiter.wait_if_needed()
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[{'role':'user','content':prompt}],
                response_format=SentimentReasoning,
            )
            usage['input_tokens'] += getattr(completion.usage,'prompt_tokens',0)
            usage['output_tokens'] += getattr(completion.usage,'completion_tokens',0)
            usage['total_tokens'] += getattr(completion.usage,'total_tokens',0)
            reasoning = completion.choices[0].message.parsed.reasoning
            return {
                'entity_name': entity,
                'avg_sentiment': avg_sent,
                'sentiment_count': len(df_e),
                'sentiment_std': df_e['normalized_sentiment'].std(),
                'content_count': df_e['content_id'].nunique(),
                'sentiment_summary': reasoning
            }
        except Exception as e:
            logger.warning(f"Failed to generate summary for '{entity}': {str(e)[:80]}")
            return None
    
    results = process_in_parallel(
        entities,
        task,
        max_workers=max_workers,
        progress_name="entities",
        progress_interval=10
    )
    
    df_sum = pd.DataFrame(results)
    if not df_sum.empty:
        df_sum.sort_values('sentiment_count', ascending=False, inplace=True)
    return df_sum, usage
