# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Optional


class BaseBudgetChecker(ABC):
    """Base class for budget checking during interval-based generation.
    
    Budget checkers are used to determine when to stop generation in 
    interval-based rollouts. They receive the generated text (and optionally
    token IDs) and return True if generation should continue, False otherwise.
    
    Example:
        class MyBudgetChecker(BaseBudgetChecker):
            def __init__(self, max_chars: int = 1000, **kwargs):
                super().__init__(**kwargs)
                self.max_chars = max_chars
            
            def check(self, text, token_ids, total_tokens_generated):
                return len(text) < self.max_chars
    """

    def __init__(self, **kwargs):
        """Initialize budget checker with optional kwargs from config."""
        self.kwargs = kwargs

    @abstractmethod
    def check(
        self,
        text: str,
        token_ids: list[int],
        total_tokens_generated: int,
    ) -> bool:
        """Check if generation should continue.

        Args:
            text: The generated text so far (decoded from token_ids)
            token_ids: The generated token IDs so far
            total_tokens_generated: Total number of tokens generated so far

        Returns:
            bool: True if generation should continue, False to stop
        """
        raise NotImplementedError


class CharacterCountBudgetChecker(BaseBudgetChecker):
    """Simple test budget checker that stops when text exceeds character limit.
    
    Example usage in config:
        actor_rollout_ref.rollout.budget_checker.name=CharacterCountBudgetChecker
        actor_rollout_ref.rollout.budget_checker.kwargs='{"max_chars": 500}'
    """

    def __init__(self, max_chars: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.max_chars = max_chars

    def check(
        self,
        text: str,
        token_ids: list[int],
        total_tokens_generated: int,
    ) -> bool:
        """Stop generation when text exceeds max_chars."""
        return len(text) < self.max_chars


class TokenCountBudgetChecker(BaseBudgetChecker):
    """Simple budget checker that stops after N tokens.
    
    Example usage in config:
        actor_rollout_ref.rollout.budget_checker.name=TokenCountBudgetChecker
        actor_rollout_ref.rollout.budget_checker.kwargs='{"max_tokens": 1000}'
    """

    def __init__(self, max_tokens: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens

    def check(
        self,
        text: str,
        token_ids: list[int],
        total_tokens_generated: int,
    ) -> bool:
        """Stop generation after max_tokens."""
        return total_tokens_generated < self.max_tokens
