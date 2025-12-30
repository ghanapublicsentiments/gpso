"""Core entity list for sentiment mapping.

This module centralizes the curated list of political figures, institutions
and national issues tracked by the sentiment pipeline. Editing this list
changes which baseline entities are embedded and mapped each run.

Guidelines for updates:
- Keep names concise and canonical (e.g., 'John Mahama', not 'H.E John Dramani Mahama')
- Prefer singular over plural unless plurality conveys a distinct concept
- Avoid adding ephemeral news cycle phrases; use enduring issue labels
- When removing an entity ensure historical reporting implications are considered

The constant `KEY_PLAYERS_AND_ISSUES` is imported by `sentiment_pipeline.py`.
"""

KEY_PLAYERS: dict[str, str] = {
    "President Mahama": "keyword_match",
    "Naana Jane Opoku-Agyemang": "keyword_match",
    "Sammy Gyamfi": "keyword_match",
    "Akufo-Addo": "keyword_match",
    "Bawumia": "keyword_match",
    "Kennedy Agyapong": "keyword_match",
    "Parliament": "keyword_match",
    "Police": "keyword_match",
    "NPP": "keyword_match",
    "NDC": "keyword_match",
    "Current Government": "embed_match",
    "Previous Government": "embed_match",
}


KEY_ISSUES: dict[str, str] = {
    "Galamsey": "keyword_match",
    "Dumsor": "keyword_match",
    "Economy": "embed_match",
    "Standard of Living": "embed_match",
    "Education": "embed_match",
    "Healthcare": "embed_match",
    "Corruption": "embed_match",
    "National Security": "embed_match",
    "Unemployment": "embed_match",
    "Human Rights": "embed_match",
    "Justice": "embed_match",
}
