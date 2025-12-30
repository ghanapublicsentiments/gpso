"""Hot topics identification stage - identifies top discussed topics from video comments."""

import pandas as pd
from pydantic import BaseModel, Field

from config import get_client
from pipeline.logger import get_logger

logger = get_logger("stages.hot_topics")


class TopicMapping(BaseModel):
    """Mapping of a single topic to content IDs that discuss it."""
    
    topic: str = Field(..., description="The hot topic (5 words or less)")
    content_ids: list[str] = Field(..., description="List of content_ids that discuss this topic")


class HotTopics(BaseModel):
    """Model for the top 8 most discussed topics mapped to their content IDs."""
    
    topic_1: TopicMapping = Field(..., description="Most discussed topic and its content_ids")
    topic_2: TopicMapping = Field(..., description="Second most discussed topic and its content_ids")
    topic_3: TopicMapping = Field(..., description="Third most discussed topic and its content_ids")
    topic_4: TopicMapping = Field(..., description="Fourth most discussed topic and its content_ids")
    topic_5: TopicMapping = Field(..., description="Fifth most discussed topic and its content_ids")
    topic_6: TopicMapping = Field(..., description="Sixth most discussed topic and its content_ids")
    topic_7: TopicMapping = Field(..., description="Seventh most discussed topic and its content_ids")
    topic_8: TopicMapping = Field(..., description="Eighth most discussed topic and its content_ids")


def identify_hot_topics(
    content_items: pd.DataFrame, 
    model: str = "gpt-4o-mini"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Identify the top 8 most discussed topics from public comments and map them to content IDs.
    
    Args:
        content_items: DataFrame of news records with comments_by_author structure.
        model: LLM model to use for topic identification and mapping.
    
    Returns:
        tuple[pd.DataFrame, dict[str, int]]: DataFrame with topic-content_id mappings and token usage.
            DataFrame has columns: content_id, topic. Returns empty DataFrame on error.
    """
    logger.info(f"Identifying hot topics from {len(content_items)} items using {model}")
    
    df_news = content_items[['content_id', 'news_title', 'source_name', 'comment_count', 'comments_by_author']].copy()
    
    news_context = df_news.to_json(orient='records', indent=2)
    
    prompt = f"""
    Identify the top 8 topics most discussed in the following public comments, and for each topic, 
    identify which content items (by content_id) discuss that topic.

    PUBLIC COMMENTS DATA:
    {news_context}

    CONSIDERATIONS:
    1. Prioritize topics based on comment volume and frequency of discussion.
    2. Each topic should be 5 words or less and should be neutral in tone.
    3. Topics should not be generic but be specific and capture the essence of the discussions.
    4. Topics must be sufficiently distinct from each other.
    5. News titles are included for context but focus on the comments themselves for topic identification.
    6. Look for recurring themes, concerns, or subjects across multiple items and authors.
    7. For each topic, include the content_ids of ALL news items where that topic is discussed 
       (look at the title, content, and comments).
    8. A content item can be associated with multiple topics if it discusses multiple topics.
    """.strip()
    
    usage = {'input_tokens': 0, 'output_tokens': 0}
    
    try:
        client = get_client(model)
        
        logger.info("Sending request to LLM for hot topic identification and mapping...")
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            response_format=HotTopics,
        )
        
        parsed = completion.choices[0].message.parsed
        
        usage['input_tokens'] = getattr(completion.usage, 'prompt_tokens', 0)
        usage['output_tokens'] = getattr(completion.usage, 'completion_tokens', 0)
        
        topic_to_content_ids = {
            parsed.topic_1.topic: parsed.topic_1.content_ids,
            parsed.topic_2.topic: parsed.topic_2.content_ids,
            parsed.topic_3.topic: parsed.topic_3.content_ids,
            parsed.topic_4.topic: parsed.topic_4.content_ids,
            parsed.topic_5.topic: parsed.topic_5.content_ids,
            parsed.topic_6.topic: parsed.topic_6.content_ids,
            parsed.topic_7.topic: parsed.topic_7.content_ids,
            parsed.topic_8.topic: parsed.topic_8.content_ids,
        }
        
        logger.info(f"Hot topics identified successfully (Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out)")
        
        rows = []
        for topic, content_ids in topic_to_content_ids.items():
            for content_id in content_ids:
                rows.append({
                    'content_id': content_id,
                    'topic': topic
                })
        
        df_topics = pd.DataFrame(rows)
        
        for topic, content_ids in topic_to_content_ids.items():
            logger.info(f"  '{topic}': {len(content_ids)} content items")
        
        return df_topics, usage
        
    except Exception as e:
        logger.error(f"Error identifying hot topics: {e}")
        return pd.DataFrame(columns=['content_id', 'topic']), usage

