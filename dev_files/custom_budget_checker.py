#!/usr/bin/env python3
"""
Example custom budget checker for testing.
This demonstrates how to create custom budget checking logic.
"""

from verl.workers.rollout.budget_checker import BaseBudgetChecker


class CustomMathBudgetChecker(BaseBudgetChecker):
    """
    Example budget checker for math problems.
    Stops when:
    - Text exceeds max_chars characters, OR
    - More than max_tool_calls tool calls have been made (detected by counting </tool_call>)
    
    Example usage in config:
        actor_rollout_ref.rollout.budget_checker.name=CustomMathBudgetChecker
        actor_rollout_ref.rollout.budget_checker.kwargs='{"max_chars": 1000, "max_tool_calls": 5}'
    """

    def __init__(self, max_chars: int = 1000, max_tool_calls: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.max_chars = max_chars
        self.max_tool_calls = max_tool_calls

    def check(
        self,
        text: str,
        token_ids: list[int],
        total_tokens_generated: int,
    ) -> bool:
        # Check character limit
        if len(text) >= self.max_chars:
            print(f"Budget exhausted: text length {len(text)} >= {self.max_chars}")
            return False

        # Check tool call count
        tool_call_count = text.count("</tool_call>")
        if tool_call_count >= self.max_tool_calls:
            print(f"Budget exhausted: {tool_call_count} tool calls >= {self.max_tool_calls}")
            return False

        return True



