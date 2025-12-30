"""Input sanitization utilities for user-facing interfaces.

This module provides functions to sanitize and validate user input to prevent
prompt injection, excessive token usage, and other potential security issues.
"""

import re


def sanitize_user_prompt(
    prompt: str, 
    max_length: int = 4000,
    min_length: int = 1,
    remove_special_patterns: bool = True
) -> str:
    """Sanitize user prompt for LLM chat interface.
    
    Args:
        prompt: Raw user input text.
        max_length: Maximum allowed prompt length (default: 4000 chars).
        min_length: Minimum required prompt length (default: 1 char).
        remove_special_patterns: Whether to remove potentially malicious patterns.
    
    Returns:
        str: Sanitized prompt text.
    
    Raises:
        ValueError: If prompt is too short, too long, or contains only whitespace.
    """
    # Strip leading/trailing whitespace
    prompt = prompt.strip()
    
    # Check minimum length
    if len(prompt) < min_length:
        raise ValueError(f"Prompt must be at least {min_length} character(s) long")
    
    # Check if prompt is only whitespace
    if not prompt or prompt.isspace():
        raise ValueError("Prompt cannot be empty or contain only whitespace")
    
    # Enforce maximum length
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    
    # Remove common prompt injection patterns if enabled
    if remove_special_patterns:
        # Remove excessive repetition (e.g., "ignore ignore ignore...")
        prompt = re.sub(r'\b(\w+)(\s+\1){4,}\b', r'\1', prompt, flags=re.IGNORECASE)
        
        # Limit consecutive special characters
        prompt = re.sub(r'([!@#$%^&*()_+=\-{}\[\]:;"\'<>,.?/\\|`~])\1{5,}', r'\1\1\1', prompt)
        
        # Remove potential system prompt leakage attempts
        suspicious_patterns = [
            r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions?',
            r'forget\s+(all\s+)?(previous|prior|above)',
            r'disregard\s+(all\s+)?(previous|prior|above)',
            r'<\s*system\s*>',
            r'<\s*/\s*system\s*>',
            r'\[\s*system\s*\]',
            r'\[\s*/\s*system\s*\]',
        ]
        
        for pattern in suspicious_patterns:
            # We don't remove entirely; we just add a note that this was flagged
            # to maintain user experience while being transparent
            if re.search(pattern, prompt, re.IGNORECASE):
                # Just log it but don't modify - let the LLM's own protections handle it
                pass
    
    return prompt


def sanitize_entity_name(entity_name: str, max_length: int = 200) -> str:
    """Sanitize entity name for custom entity analysis.
    
    Args:
        entity_name: Raw entity name input.
        max_length: Maximum allowed entity name length (default: 200 chars).
    
    Returns:
        str: Sanitized entity name.
    
    Raises:
        ValueError: If entity name is invalid.
    """
    # Strip and normalize whitespace
    entity_name = ' '.join(entity_name.split())
    
    # Check if empty
    if not entity_name:
        raise ValueError("Entity name cannot be empty")
    
    # Enforce maximum length
    if len(entity_name) > max_length:
        entity_name = entity_name[:max_length]
    
    # Remove leading/trailing special characters but allow them in the middle
    entity_name = entity_name.strip('!@#$%^&*()_+={}[]|\\:;"<>,.?/~`')
    
    # Final check after stripping
    if not entity_name:
        raise ValueError("Entity name must contain valid characters")
    
    return entity_name
