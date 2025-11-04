#!/usr/bin/env python3
"""
Example custom budget checker function for testing.
This demonstrates how to create custom budget checking logic.

Budget checker functions should have this signature:
    def my_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int, **kwargs) -> bool

Example usage in config:
    actor_rollout_ref.rollout.budget_checker.name=custom_math_budget_checker
    actor_rollout_ref.rollout.budget_checker.kwargs='{"max_chars": 1000, "max_tool_calls": 5}'
"""


def custom_math_budget_checker(
    text: str,
    token_ids: list[int],
    total_tokens_generated: int,
    max_chars: int = 1000,
    max_tool_calls: int = 5,
) -> bool:
    """
    Example budget checker for math problems.
    Stops when:
    - Text exceeds max_chars characters, OR
    - More than max_tool_calls tool calls have been made (detected by counting </tool_call>)
    
    Args:
        text: The generated text so far
        token_ids: The generated token IDs so far
        total_tokens_generated: Total number of tokens generated
        max_chars: Maximum character limit (default: 1000)
        max_tool_calls: Maximum tool calls allowed (default: 5)
        
    Returns:
        True if should continue, False to stop
    """
    # Check character limit
    if len(text) >= max_chars:
        print(f"Budget exhausted: text length {len(text)} >= {max_chars}")
        return False

    # Check tool call count
    tool_call_count = text.count("</tool_call>")
    if tool_call_count >= max_tool_calls:
        print(f"Budget exhausted: {tool_call_count} tool calls >= {max_tool_calls}")
        return False

    return True



