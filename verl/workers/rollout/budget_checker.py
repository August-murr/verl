# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example budget checker functions for interval-based generation.

Budget checkers are simple functions that determine when to stop generation.
They receive the generated text and token IDs, and return True to continue or False to stop.

Function signature:
    def my_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int) -> bool:
        # Return True to continue, False to stop
        pass

Note: All parameters (like limits) should be hardcoded in the function definition.
"""


def character_count_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int) -> bool:
    """Stop generation when text exceeds 500 characters.
    
    Args:
        text: The generated text so far
        token_ids: The generated token IDs so far
        total_tokens_generated: Total number of tokens generated
        
    Returns:
        True if should continue (text < 500 chars), False to stop
    """
    MAX_CHARS = 50000
    return len(text) < MAX_CHARS


def token_count_budget_checker(text: str, token_ids: list[int], total_tokens_generated: int) -> bool:
    """Stop generation after 1000 tokens.
    
    Args:
        text: The generated text so far
        token_ids: The generated token IDs so far
        total_tokens_generated: Total number of tokens generated
        
    Returns:
        True if should continue (tokens < 1000), False to stop
    """
    MAX_TOKENS = 1000
    return total_tokens_generated < MAX_TOKENS
