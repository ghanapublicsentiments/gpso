"""UI component functions for the Streamlit sentiment analysis dashboard."""

import numpy as np


def interpolate_gradient_color(position_percent: float) -> str:
    """Interpolate color from the sentiment gradient (0-100%).

    Gradient: #d4a89a (0%) -> #ecebe3 (50%) -> #c8ddc8 (100%)
    
    Args:
        position_percent: Position in the gradient (0-100).
    
    Returns:
        str: Hex color code.
    """
    color1 = (0xD4, 0xA8, 0x9A)  # Negative
    color2 = (0xEC, 0xEB, 0xE3)  # Neutral
    color3 = (0xC8, 0xDD, 0xC8)  # Positive

    position = position_percent / 100.0

    if position <= 0.5:
        # Interpolate between color1 and color2
        t = position * 2
        r = int(color1[0] + (color2[0] - color1[0]) * t)
        g = int(color1[1] + (color2[1] - color1[1]) * t)
        b = int(color1[2] + (color2[2] - color1[2]) * t)
    else:
        # Interpolate between color2 and color3
        t = (position - 0.5) * 2
        r = int(color2[0] + (color3[0] - color2[0]) * t)
        g = int(color2[1] + (color3[1] - color2[1]) * t)
        b = int(color2[2] + (color3[2] - color2[2]) * t)

    return f"#{r:02x}{g:02x}{b:02x}"


def render_entity_card(
    entity_name: str,
    entity_data: dict,
    is_best: bool = False,
    is_worst: bool = False,
    is_popular: bool = False,
    *,
    clamp_se: bool = True
) -> str:
    """Render a sentiment entity card used across pages.

    Expected keys in ``entity_data``:
      - 'Avg Sentiment': float in [-1, 1]
      - 'Std Dev': float
      - 'Mentions': int
      - 'Is New': bool

    Args:
        entity_name: Name of the entity.
        entity_data: Dictionary containing sentiment statistics.
        is_best: Whether this is the most positive entity.
        is_worst: Whether this is the most negative entity.
        is_popular: Whether this is a popular entity.
        clamp_se: Whether to clamp the standard error.
    
    Returns:
        str: HTML string for the entity card.
    """
    sentiment_score = float(entity_data.get("Avg Sentiment", 0.0))
    std_dev = float(entity_data.get("Std Dev", 0.0))
    mentions = max(1, int(entity_data.get("Mentions", 1)))
    is_new = bool(entity_data.get("Is New", False))

    entity_name_display = str(entity_name).upper()

    # Position on scale (0-100%) from [-1, 1]
    position = ((sentiment_score + 1.0) / 2.0) * 100.0

    # Standard error and 95% CI
    se_raw = std_dev / np.sqrt(mentions) if mentions > 0 else 0.0
    if clamp_se:
        standard_error = float(np.clip(se_raw, 0.0, np.sqrt(2.0)))
    else:
        standard_error = float(np.clip(se_raw, 0.0, np.sqrt(2.0)))

    ci_half_width = 1.96 * standard_error

    left_sentiment = float(np.clip(sentiment_score - ci_half_width, -1.0, 1.0))
    right_sentiment = float(np.clip(sentiment_score + ci_half_width, -1.0, 1.0))

    left_position = ((left_sentiment + 1.0) / 2.0) * 100.0
    right_position = ((right_sentiment + 1.0) / 2.0) * 100.0

    # Ensure minimum visible width
    if right_position - left_position < 0.5:
        left_position = max(0.0, position - 0.25)
        right_position = min(100.0, position + 0.25)

    marker_color = interpolate_gradient_color(position)

    # Badges
    badges_html = ""
    if is_popular and not is_new:
        badges_html += '<div style="font-size: 10px; padding: 2px 6px; background-color: #c2185b; color: white; border-radius: 4px; font-weight: 600;">⭐ Popular</div>'
    if is_best:
        badges_html += '<div style="font-size: 10px; padding: 2px 6px; background-color: #059669; color: white; border-radius: 4px;">😊 Most Positive</div>'
    elif is_worst:
        badges_html += '<div style="font-size: 10px; padding: 2px 6px; background-color: #bb5a38; color: white; border-radius: 4px;">😟 Most Negative</div>'

    html = '<div style="margin-bottom: 10px;">'
    html += '<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px; gap: 8px;">'
    html += f'<div style="font-size: 14px; font-weight: bold; flex-shrink: 0;">{entity_name_display}</div>'
    html += f'<div style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px; justify-content: flex-end;">{badges_html}</div>'
    html += '</div>'

    html += (
        '<div style="position: relative; background: linear-gradient(to right, '
        '#d4a89a 0%, #ecebe3 50%, #c8ddc8 100%); border-radius: 10px; height: 24px; '
        'border: 1px solid #d3d2ca;">'
    )

    # Confidence interval band
    html += (
        f'<div style="position: absolute; left: {left_position}%; right: {100 - right_position}%; '
        'top: 50%; transform: translateY(-50%); height: 28px; background-color: '
        f'{marker_color}; border-radius: 2px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);"></div>'
    )

    # Mean tick
    html += (
        f'<div style="position: absolute; left: {position}%; top: 50%; transform: translate(-50%, -50%); '
        'width: 3px; height: 32px; background-color: '
        f'{marker_color}; border-radius: 1px; box-shadow: 0 1px 3px rgba(0,0,0,0.5); '
        'opacity: 0.8; z-index: 1; border: 1px solid rgba(255,255,255,0.3);"></div>'
    )

    # Vertical tick marks at -1, 0, +1
    html += '<div style="position: absolute; left: 0%; top: 50%; transform: translateY(-50%); width: 1px; height: 16px; background-color: rgba(61,58,42,0.25);"></div>'
    html += '<div style="position: absolute; left: 50%; top: 50%; transform: translateY(-50%); width: 1px; height: 20px; background-color: rgba(61,58,42,0.35);"></div>'
    html += '<div style="position: absolute; left: 100%; top: 50%; transform: translateY(-50%); width: 1px; height: 16px; background-color: rgba(61,58,42,0.25);"></div>'
    html += '</div>'

    # Axis labels and score label row
    html += '<div style="position: relative; display: flex; justify-content: space-between; font-size: 10px; margin-top: 2px;">'

    # Left label
    if position < 15:
        html += '<div style="text-align: left; opacity: 0;">'
    else:
        html += '<div style="text-align: left;">'
    html += '<div style="color: #8b8577; font-weight: 600;">Negative</div>'
    html += '<div style="color: #8b8577;">-1.0</div>'
    html += '</div>'

    # Center label
    html += '<div style="text-align: center;">'
    html += '<div style="color: #8b8577; font-weight: 600;">Neutral</div>'
    html += '<div style="color: #8b8577;">0</div>'
    html += '</div>'

    # Right label
    if position > 85:
        html += '<div style="text-align: right; opacity: 0;">'
    else:
        html += '<div style="text-align: right;">'
    html += '<div style="color: #8b8577; font-weight: 600;">Positive</div>'
    html += '<div style="color: #8b8577;">+1.0</div>'
    html += '</div>'

    # Score label anchored at tick
    html += (
        f'<span style="position: absolute; left: {position}%; transform: translateX(-50%); '
        'font-size: 12px; font-weight: bold; color: #3d3a2a; white-space: nowrap; '
        'top: 50%; margin-top: -6px;">'
        f'{sentiment_score:.2f}</span>'
    )

    html += '</div>'
    html += '</div>'

    return html
