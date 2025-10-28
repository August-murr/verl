#!/usr/bin/env python3
"""
Simple custom reward function that rewards shorter answers.

Reward = 1 / length of response
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, 
                 reward_router_address=None, reward_model_tokenizer=None, **kwargs):
    """
    Simple reward function that rewards shorter answers.
    
    Args:
        data_source (str): The dataset name (e.g., "openai/gsm8k")
        solution_str (str): The model's response/solution
        ground_truth (str): The correct answer
        extra_info (dict, optional): Additional information
        reward_router_address (str, optional): Address of reward router (for remote evaluation)
        reward_model_tokenizer (optional): Tokenizer for reward model
        **kwargs: Additional keyword arguments (for flexibility)
    
    Returns:
        float: Reward score (inverse of response length)
    """
    # Simple reward: 1 / length of response
    response_length = len(solution_str)
    
    # Avoid division by zero
    if response_length == 0:
        return 0.0
    
    # Simple inverse relationship: reward = 1 / length
    reward = 1.0 / response_length
    
    return reward