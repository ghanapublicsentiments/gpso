"""Tool orchestration and execution logic for the sentiment analysis chatbot.

This module handles the execution of tool calls from the LLM, including error handling,
retry logic, and response processing.
"""

import json

from chat_tools import (
    AVAILABLE_TOOLS,
    TOOL_FUNCTIONS,
    AnalyzeCustomEntityParams,
    CreateDataframeParams,
    CreatePlotlyFigureParams,
    GetDataframeParams,
)
from config import get_client


def _get_error_suggestion(function_name: str, error_msg: str) -> str:
    """Generate helpful suggestions based on the error type.
    
    Args:
        function_name: Name of the function that failed.
        error_msg: The error message.
    
    Returns:
        str: A helpful suggestion for fixing the error.
    """
    error_lower = error_msg.lower()
    
    # Dataframe not found errors
    if "not found" in error_lower or "empty" in error_lower:
        if function_name == "get_dataframe":
            return """
            The dataframe_id may be invalid. Use 'df_entity_summaries' for entity summaries.
            If using a custom dataframe, ensure it was created successfully first.
            Note: Individual comment sentiments are not directly accessible for privacy protection.
            Use 'analyze_custom_entity_sentiment' to analyze custom entities.
            """
        elif function_name in ["create_plotly_figure"]:
            return """
            The dataframe_id may be invalid. Make sure to use a valid dataframe_id from 
            get_dataframe or create_dataframe results.
            """
    
    # Column errors
    if "column" in error_lower or "key" in error_lower:
        if function_name == "get_dataframe":
            return """
            Check column names - available columns in df_entity_summaries: 
            id, run_id, entity_name, avg_sentiment, sentiment_count, sentiment_std, 
            content_count, sentiment_summary, model_used, created_at.
            """
        elif function_name == "create_plotly_figure":
            return """
            Ensure x_column and y_column exist in the referenced dataframe. 
            You can use get_dataframe first to check available columns, or provide 
            x_values and y_values directly instead of column names.
            """
    
    # Missing required arguments
    if "required" in error_lower or "missing" in error_lower:
        if function_name == "get_dataframe":
            return """
            The 'dataframe_id' parameter is required. 
            Specify 'df_entity_summaries' for entity summary data.
            """
        elif function_name == "create_dataframe":
            return """
            The 'records' parameter is required. 
            Provide a list of dictionaries representing rows.
            """
        elif function_name == "create_plotly_figure":
            return """
            Required: 'chart_type' (line/bar/scatter/pie/histogram). 
            Also need either (dataframe_id + x_column + y_column) or (x_values + y_values).
            """
    
    # Data type errors
    if "type" in error_lower or "expected" in error_lower:
        return f"""
        Check argument types for {function_name}. Filters should be lists of dicts, 
        limit should be an integer, dataframe_id should be a string.
        """
    
    # Generic fallback
    return f"""
    Review the {function_name} parameters. Ensure all required arguments are provided 
    with correct types and valid values. Check the function documentation for examples.
    """


def execute_tool_calls(
    messages: list[dict],
    selected_model: str,
    get_system_message_fn,
    api_key: str | None = None,
    max_iterations: int = 5
) -> tuple[str, list[dict]]:
    """Orchestrate tool calling: prompt LLM, execute tools, return final response.
    
    Args:
        messages: The conversation history.
        selected_model: The model to use for chat completions.
        get_system_message_fn: Function that returns the system message.
        api_key: Optional API key for the OpenAI client.
        max_iterations: Maximum number of tool call cycles to prevent infinite loops.
    
    Returns:
        tuple[str, list[dict]]: Tuple of (final_response_text, updated_messages).
        
    Raises:
        ValueError: If there's a configuration error (API key missing, invalid model, etc.)
        ConnectionError: If there's a network connectivity issue.
    """
    try:
        openai_client = get_client(selected_model, api_key=api_key)
    except ValueError as e:
        # Re-raise ValueError so it can be caught in the chat UI with specific handling
        raise
    except Exception as e:
        # Convert other client creation errors to ConnectionError
        raise ConnectionError(f"Failed to initialize AI client: {str(e)}")
    
    tools = AVAILABLE_TOOLS
    
    iteration = 0
    updated_messages = messages.copy()
    
    if not updated_messages or updated_messages[0].get("role") != "system":
        updated_messages.insert(0, {
            "role": "system",
            "content": get_system_message_fn()
        })
    
    while iteration < max_iterations:
        iteration += 1
        
        try:
            response = openai_client.chat.completions.create(
                model=selected_model,
                messages=updated_messages,
                tools=tools,
                tool_choice="auto",
            )
            
            assistant_message = response.choices[0].message
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e).lower()
            
            # Provide user-friendly error message based on error type
            if "APIConnectionError" in error_type or "ConnectionError" in error_type:
                friendly_msg = "I'm having trouble connecting to the AI service. Please check your internet connection and try again."
            elif "APITimeoutError" in error_type or "timeout" in error_msg:
                friendly_msg = "The AI service took too long to respond. Please wait a moment and try again."
            elif "RateLimitError" in error_type or "rate limit" in error_msg:
                friendly_msg = "We've hit the rate limit from too many requests. Please wait a few moments before trying again."
            elif "AuthenticationError" in error_type or "authentication" in error_msg or "unauthorized" in error_msg:
                friendly_msg = "Your API key seems to be invalid or expired. Please check your API key configuration."
            elif "InvalidRequestError" in error_type:
                friendly_msg = "The request was rejected by the AI service. Try selecting a different model or refreshing the page."
            else:
                friendly_msg = "Something went wrong while processing your request. Please try again or rephrase your question."
            
            # Add error message to conversation
            error_response = {
                "role": "assistant",
                "content": friendly_msg
            }
            updated_messages.append(error_response)
            
            return error_response["content"], updated_messages
        
        if assistant_message.tool_calls:
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
            
            # Track created figures and dataframes in this iteration
            figure_ids = []
            dataframe_ids = []
            
            # Execute each tool call with error handling and retry logic
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    function_response = json.dumps({
                        "error": "Invalid JSON in function arguments",
                        "details": str(e),
                        "received_arguments": tool_call.function.arguments,
                        "suggestion": "Please ensure all arguments are valid JSON format"
                    })
                    updated_messages.append({
                        "role": "tool",
                        "content": function_response,
                        "tool_call_id": tool_call.id,
                    })
                    continue
                
                # Execute the function with error handling
                if function_name in TOOL_FUNCTIONS:
                    try:
                        param_models = {
                            "get_dataframe": GetDataframeParams,
                            "create_dataframe": CreateDataframeParams,
                            "create_plotly_figure": CreatePlotlyFigureParams,
                            "analyze_custom_entity_sentiment": AnalyzeCustomEntityParams,
                        }
                        
                        if function_name in param_models:
                            params = param_models[function_name](**function_args)
                            function_response = TOOL_FUNCTIONS[function_name](params)
                        else:
                            function_response = TOOL_FUNCTIONS[function_name](**function_args)
                        
                        # Check if the function returned an error
                        try:
                            result = json.loads(function_response)
                            if "error" in result:
                                # Function executed but returned an error
                                # Enhance error message with helpful suggestions
                                error_msg = result.get("error", "Unknown error")
                                enhanced_error = {
                                    "error": error_msg,
                                    "function": function_name,
                                    "arguments_received": function_args,
                                    "suggestion": _get_error_suggestion(function_name, error_msg)
                                }
                                function_response = json.dumps(enhanced_error)
                            else:
                                # Track IDs for rendering if successful
                                if function_name == "create_plotly_figure" and "figure_id" in result:
                                    figure_ids.append(result["figure_id"])
                                elif function_name == "create_dataframe" and "dataframe_id" in result:
                                    dataframe_ids.append(result["dataframe_id"])
                        except (json.JSONDecodeError, TypeError):
                            # Response is not JSON, treat as success
                            pass
                            
                    except TypeError as e:
                        # Wrong arguments passed to function
                        function_response = json.dumps({
                            "error": "Invalid function arguments",
                            "function": function_name,
                            "details": str(e),
                            "arguments_received": function_args,
                            "suggestion": _get_error_suggestion(function_name, str(e))
                        })
                    except Exception as e:
                        # Other execution errors
                        function_response = json.dumps({
                            "error": "Function execution failed",
                            "function": function_name,
                            "details": str(e),
                            "arguments_received": function_args,
                            "suggestion": "Please check the arguments and try again with valid values"
                        })
                else:
                    function_response = json.dumps({
                        "error": f"Unknown function: {function_name}",
                        "available_functions": list(TOOL_FUNCTIONS.keys()),
                        "suggestion": "Please use one of the available functions"
                    })
                
                # Add function response to messages
                updated_messages.append({
                    "role": "tool",
                    "content": function_response,
                    "tool_call_id": tool_call.id,
                })
            
            # Store metadata about created artifacts
            if figure_ids:
                updated_messages[-1]["figure_ids"] = figure_ids
            if dataframe_ids:
                updated_messages[-1]["dataframe_ids"] = dataframe_ids
            
            continue
        else:
            # No tool calls, we have the final response
            final_response = assistant_message.content or ""
            final_message = {"role": "assistant", "content": final_response}
            
            # Check if any figures or dataframes were created in previous iterations
            # by looking through the messages
            all_figure_ids = []
            all_dataframe_ids = []
            for msg in updated_messages:
                if msg.get("role") == "tool":
                    if "figure_ids" in msg:
                        all_figure_ids.extend(msg["figure_ids"])
                    if "dataframe_ids" in msg:
                        all_dataframe_ids.extend(msg["dataframe_ids"])
            
            if all_figure_ids:
                final_message["figure_ids"] = all_figure_ids
            if all_dataframe_ids:
                final_message["dataframe_ids"] = all_dataframe_ids
            
            updated_messages.append(final_message)
            return final_response, updated_messages
    
    # If we hit max iterations, check if we're stuck in an error loop
    error_count = sum(
        1 for msg in updated_messages
        if msg.get("role") == "tool" and "error" in msg.get("content", "")
    )

    if error_count >= 3:
        return (
            """
            I apologize, but I'm having trouble executing the requested operations correctly. 
            The tools are returning errors even after multiple attempts. 
            Please try rephrasing your question or asking for something different.

            If you're asking about specific data, try being more explicit about:
            - Which dataset to use (comment sentiments or entity summaries)
            - What columns or fields you're interested in
            - What time range or filters to apply
            - What type of visualization you'd like (bar chart, line graph, etc.)
            """,
            updated_messages
        )
    
    return """
    I apologize, but I've reached the maximum number of tool calls. 
    Please try rephrasing your question.
    """, updated_messages
