#!/usr/bin/env python3
"""
Example custom budget checker function for testing.
This demonstrates how to create custom budget checking logic.

Budget checker functions should have this signature:
    def my_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int) -> bool

All parameters (like limits) should be hardcoded in the function.

Example usage in config:
    actor_rollout_ref.rollout.budget_checker.path=/root/verl/dev_files/custom_budget_checker.py
    actor_rollout_ref.rollout.budget_checker.name=custom_math_budget_checker
"""


def custom_math_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int) -> bool:
    """
    Example budget checker for math problems.
    Stops when:
    - Text exceeds 1000 characters, OR
    - More than 5 tool calls have been made (detected by counting </tool_call>)
    
    Args:
        text: The generated text so far
        token_ids: The generated token IDs so far
        total_tokens_generated: Total number of tokens generated
        
    Returns:
        True if should continue, False to stop
    """
    MAX_CHARS = 1000
    MAX_TOOL_CALLS = 5
    
    # Check character limit
    if len(text) >= MAX_CHARS:
        print(f"Budget exhausted: text length {len(text)} >= {MAX_CHARS}")
        return False

    # Check tool call count
    tool_call_count = text.count("</tool_call>")
    if tool_call_count >= MAX_TOOL_CALLS:
        print(f"Budget exhausted: {tool_call_count} tool calls >= {MAX_TOOL_CALLS}")
        return False

    return True



