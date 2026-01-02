# Gemini Tool Calling Fix

## Problem
When using Gemini models (gemini-3-flash-preview, gemini-3-pro-preview, gemini-2.5-flash-lite) through the OpenAI compatibility layer, we were getting this error:

```
BadRequestError: error code: 400 - function call is missing a thought_signature in functioncall parts. 
this is required for tools to work correctly, and missing thought_signature may lead to degraded model performance.
```

## Root Cause
Gemini 3 models require **thought signatures** to be preserved when using function calling (tool calling). When the model returns a tool call, it includes a `thought_signature` in the response (stored in `extra_content.google.thought_signature`). This signature **must** be included when sending the conversation history back to the model in subsequent turns.

From the [Gemini documentation](https://ai.google.dev/gemini-api/docs/thought-signatures):
- Thought signatures are encrypted representations of the model's internal thought process
- For Gemini 3 models, you MUST pass back thought signatures during function calling
- The signature is returned in `extra_content.google.thought_signature` when using OpenAI compatibility
- Failure to include signatures results in a 400 error

## Solution
Modified `chat_orchestrator.py` to preserve the `extra_content` field (which contains the thought_signature) when storing assistant messages with tool calls:

### Before:
```python
updated_messages.append({
    "role": "assistant",
    "content": assistant_message.content or "",
    "tool_calls": [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        }
        for tc in assistant_message.tool_calls
    ]
})
```

### After:
```python
# Preserve tool calls with thought_signature for Gemini models
tool_calls_data = []
for tc in assistant_message.tool_calls:
    tool_call_dict = {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        }
    }
    
    # Preserve thought_signature for Gemini models (in extra_content)
    # This is required for Gemini 3 models when using OpenAI compatibility
    if hasattr(tc, 'extra_content') and tc.extra_content:
        tool_call_dict["extra_content"] = tc.extra_content
    
    tool_calls_data.append(tool_call_dict)

updated_messages.append({
    "role": "assistant",
    "content": assistant_message.content or "",
    "tool_calls": tool_calls_data
})
```

## Key Points
- This fix is **non-breaking** - it only adds the `extra_content` field when it exists
- Works with all models (OpenAI, Anthropic, Grok, etc.) that don't use thought signatures
- Specifically enables Gemini 3 and 2.5 models to work correctly with tool calling
- For parallel function calls, only the first tool call has a thought signature
- For sequential function calls, each step has its own thought signature

## Testing
To test this fix:
1. Use a Gemini model (e.g., `gemini-3-flash-preview`)
2. Ask a question that requires tool calling (e.g., "Show me the top 10 entities by sentiment")
3. The model should now successfully call tools without the 400 error

## Reference
- [Gemini Thought Signatures Documentation](https://ai.google.dev/gemini-api/docs/thought-signatures)
- [OpenAI Compatibility Guide](https://ai.google.dev/gemini-api/docs/openai)
