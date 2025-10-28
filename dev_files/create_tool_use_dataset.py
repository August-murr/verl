#!/usr/bin/env python3
"""
Create a dataset for math calculation tasks with tool agent loop support.
This script creates a parquet dataset compatible with VERL GRPO agentic RL training.

Based on the VERL documentation, the dataset needs these exact columns:
1. data_source: Dataset identifier
2. agent_name: "tool_agent" for agentic RL training
3. prompt: Chat template format with system + user messages
4. ability: Task category 
5. reward_model: Contains ground_truth for evaluation
6. extra_info: Additional metadata including tool configuration
"""

import argparse
import os
import random
import pandas as pd
from datasets import Dataset

def generate_math_question():
    """Generate a random math question with multiplication and addition."""
    # Generate random numbers for the calculation
    num1 = random.randint(1000, 9999)
    num2 = random.randint(1000, 9999)
    num3 = random.randint(1000, 9999)
    
    question = f"What's {num1} times {num2} plus {num3}?"
    # Calculate the correct answer
    answer = num1 * num2 + num3
    
    return question, str(answer)

def create_tool_use_dataset(num_samples=64, local_save_dir="~/data/tool_use_math"):
    """
    Create a dataset for math calculation tasks with tool agent loop support.
    
    Args:
        num_samples: Number of samples to create
        local_save_dir: Directory to save the parquet files
    """
    
    # Create the dataset structure required by VERL with tool agent loop support
    dataset_data = {
        "data_source": [],
        "agent_name": [],
        "prompt": [], 
        "ability": [],
        "reward_model": [],
        "extra_info": []
    }
    
    print(f"Creating {num_samples} samples for math calculation task with tool agent loop...")
    
    for i in range(num_samples):
        # Generate a random math question and answer
        question, correct_answer = generate_math_question()
        
        # Create prompt in chat template format with system message (required by tool agent)
        prompt_chat_format = [
            {
                "role": "system",
                "content": (
                    "You are a math expert. Solve problems step by step using your reasoning abilities. "
                    "An `execute_code` tool is available if you need to perform complex calculations, you may be able to use it rarely."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Add to dataset with tool agent loop configuration
        dataset_data["data_source"].append("custom/tool_use_math")
        dataset_data["agent_name"].append("tool_agent")  # Required for agentic RL
        dataset_data["prompt"].append(prompt_chat_format)
        dataset_data["ability"].append("math")
        
        # Reward model with ground truth for evaluation
        dataset_data["reward_model"].append({
            "style": "rule",
            "ground_truth": correct_answer
        })
        
        # Extra info with tool configuration (using execute_code tool)
        dataset_data["extra_info"].append({
            "index": i,
            "task": "math_calculation",
            "question": question,
            "answer": correct_answer,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "execute_code": {
                    "create_kwargs": {"dummy": "placeholder"},  # Parquet needs non-empty struct
                    # "execute_kwargs": {},
                    # "calc_reward_kwargs": {},
                    # "release_kwargs": {},
                },
            },
            "interaction_kwargs": {
                "query": question,
                "ground_truth": correct_answer,
            },
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(dataset_data)
    
    # Expand the local save directory
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Split into train/test (80/20 split)
    train_size = int(0.8 * num_samples)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Save parquet files
    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"✅ Tool use math dataset created successfully!")
    print(f"📁 Saved to: {local_save_dir}")
    print(f"📊 Train samples: {len(train_df)}")
    print(f"📊 Test samples: {len(test_df)}")
    print(f"📄 Files created:")
    print(f"   - {train_path}")
    print(f"   - {test_path}")
    
    return train_path, test_path

def verify_dataset(parquet_path):
    """Verify the dataset format matches VERL tool agent loop requirements."""
    print(f"\n🔍 Verifying dataset: {parquet_path}")
    
    # Load with Hugging Face datasets
    dataset = Dataset.from_parquet(parquet_path)
    
    print(f"✅ Dataset loaded successfully")
    print(f"📊 Number of rows: {len(dataset)}")
    print(f"📋 Columns: {dataset.column_names}")
    
    # Check required columns for tool agent loop
    required_columns = ["data_source", "agent_name", "prompt", "ability", "reward_model", "extra_info"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    else:
        print(f"✅ All required columns present")
    
    # Verify agent_name field
    sample = dataset[0]
    if sample["agent_name"] != "tool_agent":
        print(f"❌ agent_name should be 'tool_agent', got: {sample['agent_name']}")
        return False
    else:
        print(f"✅ agent_name correctly set to 'tool_agent'")
    
    # Verify tool configuration in extra_info
    extra_info = sample["extra_info"]
    if "need_tools_kwargs" not in extra_info or not extra_info["need_tools_kwargs"]:
        print(f"❌ need_tools_kwargs not properly set in extra_info")
        return False
    else:
        print(f"✅ Tool configuration properly set")
    
    # Show sample data
    print(f"\n📝 Sample row:")
    for key, value in sample.items():
        if key == "extra_info":
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create tool use math dataset for VERL GRPO agentic RL training")
    parser.add_argument("--num_samples", type=int, default=64, 
                       help="Number of samples to create (default: 64)")
    parser.add_argument("--local_save_dir", default="~/data/tool_use_math",
                       help="Directory to save the dataset (default: ~/data/tool_use_math)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the created dataset")
    
    args = parser.parse_args()
    
    # Create the dataset
    train_path, test_path = create_tool_use_dataset(
        num_samples=args.num_samples,
        local_save_dir=args.local_save_dir
    )
    
    # Verify if requested
    if args.verify:
        verify_dataset(train_path)
        verify_dataset(test_path)
    
    print(f"\n🚀 Ready for GRPO agentic RL training!")
    print(f"💡 Update your training script to use:")
    print(f"   data.train_files={train_path}")
    print(f"   data.val_files={test_path}")
    print(f"🔧 The agent_name field is set to 'tool_agent' for agentic RL training")