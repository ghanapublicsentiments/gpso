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
      - 'Content Items': int (optional)
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
    content_items = int(entity_data.get("Content Items", 0))
    is_new = bool(entity_data.get("Is New", False))

    entity_name_display = str(entity_name)

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
    
    # Determine sentiment category and color - Using gradient colors
    # Gradient: #d4a89a (negative) -> #ecebe3 (neutral) -> #c8ddc8 (positive)
    if sentiment_score > 0.3:
        sentiment_category = "Positive"
        category_color = "#c8ddc8"  # Positive end of gradient
        category_bg = "#5a7a5a"  # Darker shade of positive color
        category_icon = "😊"
    elif sentiment_score < -0.3:
        sentiment_category = "Negative"
        category_color = "#d4a89a"  # Negative end of gradient
        category_bg = "#8a6a5a"  # Darker shade of negative color
        category_icon = "😟"
    else:
        sentiment_category = "Neutral"
        category_color = "#ecebe3"  # Neutral center of gradient
        category_bg = "#9a9990"  # Darker shade of neutral color
        category_icon = "😐"

    # Badges - using gradient colors
    badges_html = ""
    if is_popular and not is_new:
        badges_html += '<span style="display: inline-block; font-size: 9px; padding: 3px 8px; background: linear-gradient(135deg, #ecebe3, #d0cfc5); color: #3d3a2a; border-radius: 12px; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 4px rgba(236,235,227,0.3);">Popular</span>'
    if is_best:
        badges_html += '<span style="display: inline-block; font-size: 9px; padding: 3px 8px; background: linear-gradient(135deg, #c8ddc8, #a8c8a8); color: #2d4a2d; border-radius: 12px; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 4px rgba(200,221,200,0.3);">Most Positive</span>'
    elif is_worst:
        badges_html += '<span style="display: inline-block; font-size: 9px; padding: 3px 8px; background: linear-gradient(135deg, #d4a89a, #c4988a); color: #4a3a2a; border-radius: 12px; font-weight: 600; margin-left: 6px; box-shadow: 0 2px 4px rgba(212,168,154,0.3);">Most Negative</span>'

    # Start card with responsive container, border, and dark blue-gray background
    html = '<div style="margin-bottom: 12px; width: 100%; box-sizing: border-box; padding: 16px; border: 2px solid #134e4a; border-radius: 12px; background: #134e4a; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">'
    
    # Header section with entity name and badges
    html += '<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; margin-bottom: 12px; gap: 8px;">'
    html += f'<div style="font-size: 25px; font-weight: 400; color: #f9fafb; flex-shrink: 1; min-width: 0; word-wrap: break-word;">{entity_name_display}</div>'
    html += f'<div style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">{badges_html}</div>'
    html += '</div>'
    
    # Stats row (sentiment category, score, mentions, comments)
    html += f'<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; align-items: center;">'
    html += f'<div style="display: inline-flex; align-items: center; padding: 4px 10px; background: {category_bg}; border-radius: 16px; border: 1px solid {category_color}20;">'
    html += f'<span style="font-size: 14px; margin-right: 4px;">{category_icon}</span>'
    html += f'<span style="font-size: 12px; font-weight: 600; color: {category_color};">{sentiment_category}</span>'
    html += f'</div>'
    html += f'<div style="font-size: 12px; color: #d1d5db;"><strong style="color: #f3f4f6;">Score:</strong> {sentiment_score:.2f}</div>'
    html += f'<div style="font-size: 12px; color: #d1d5db;"><strong style="color: #f3f4f6;">💬 {mentions:,}</strong> mentions</div>'
    if content_items > 0:
        html += f'<div style="font-size: 12px; color: #d1d5db;"><strong style="color: #f3f4f6;">📄 {content_items:,}</strong> items</div>'
    html += '</div>'

    # Gradient gauge - App theme colors (rust to yellow to green)
    html += (
        '<div style="position: relative; background: linear-gradient(to right, '
        '#d4a89a 0%, #ecebe3 50%, #c8ddc8 100%); border-radius: 12px; height: 28px; '
        'box-shadow: 0 2px 6px rgba(0,0,0,0.3); overflow: visible;">'
    )

    # Confidence interval band with enhanced styling
    html += (
        f'<div style="position: absolute; left: {left_position}%; right: {100 - right_position}%; '
        'top: 50%; transform: translateY(-50%); height: 32px; background-color: '
        f'{marker_color}; border-radius: 6px; box-shadow: 0 3px 8px rgba(0,0,0,0.25); opacity: 0.85;"></div>'
    )

    # Mean tick with enhanced styling
    html += (
        f'<div style="position: absolute; left: {position}%; top: 50%; transform: translate(-50%, -50%); '
        'width: 4px; height: 36px; background-color: '
        f'{marker_color}; border-radius: 2px; box-shadow: 0 2px 6px rgba(0,0,0,0.6); '
        'z-index: 2; border: 2px solid white;"></div>'
    )

    html += '</div>'

    # Axis labels with improved typography (light colors for dark background)
    html += '<div style="position: relative; display: flex; justify-content: space-between; font-size: 10px; margin-top: 6px; padding: 0 4px;">'

    # Left label
    html += '<div style="text-align: left; min-width: 50px;">'
    html += '<div style="color: #f4a582; font-weight: 700; font-size: 9px;">NEGATIVE</div>'
    html += '<div style="color: #d1d5db; font-size: 9px;">-1.0</div>'
    html += '</div>'

    # Center label
    html += '<div style="text-align: center; min-width: 50px;">'
    html += '<div style="color: #fde68a; font-weight: 700; font-size: 9px;">NEUTRAL</div>'
    html += '<div style="color: #d1d5db; font-size: 9px;">0</div>'
    html += '</div>'

    # Right label
    html += '<div style="text-align: right; min-width: 50px;">'
    html += '<div style="color: #6ee7b7; font-weight: 700; font-size: 9px;">POSITIVE</div>'
    html += '<div style="color: #d1d5db; font-size: 9px;">+1.0</div>'
    html += '</div>'

    # Score label anchored at tick with enhanced styling
    html += (
        f'<span style="position: absolute; left: {position}%; transform: translateX(-50%); '
        'font-size: 13px; font-weight: 800; color: white; white-space: nowrap; '
        f'top: 50%; margin-top: -7px; text-shadow: 0 2px 4px rgba(0,0,0,0.5); '
        f'background: {marker_color}; padding: 2px 8px; border-radius: 8px; '
        'border: 2px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">'
        f'{sentiment_score:.2f}</span>'
    )

    html += '</div>'
    html += '</div>'

    return html
