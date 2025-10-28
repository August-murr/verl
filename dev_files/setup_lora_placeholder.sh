#!/bin/bash
# Workaround for vLLM V1 LoRA placeholder path issue
# This creates a dummy adapter_config.json that vLLM V1 expects to find

set -e

echo "Creating placeholder directory for vLLM V1 LoRA workaround..."

# Create the placeholder directory
mkdir -p simon_lora_path

# Get LoRA configuration from the training script
LORA_RANK=8
LORA_ALPHA=32
TARGET_MODULES="all-linear"

# Create a minimal adapter_config.json that vLLM expects
# This matches the LoRA configuration from run_qwen2-0.5b_tool_use.sh
cat > simon_lora_path/adapter_config.json <<EOF
{
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  "peft_type": "LORA",
  "task_type": "CAUSAL_LM",
  "r": ${LORA_RANK},
  "lora_alpha": ${LORA_ALPHA},
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": false,
  "modules_to_save": null
}
EOF

# Create dummy adapter_model.safetensors to satisfy vLLM's check for tensor files
# This is just an empty placeholder - actual tensors are passed via TensorLoRARequest
python3 << 'PYEOF'
import torch
from safetensors.torch import save_file

# Create minimal dummy tensors (vLLM just checks if the file exists)
dummy_tensors = {
    "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((8, 1)),
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((1, 8)),
}

save_file(dummy_tensors, "simon_lora_path/adapter_model.safetensors")
print("✅ Created simon_lora_path/adapter_model.safetensors")
PYEOF

echo "✅ Created simon_lora_path/adapter_config.json"
echo "📄 Contents:"
cat simon_lora_path/adapter_config.json

echo ""
echo "📁 Directory structure:"
ls -lh simon_lora_path/

echo ""
echo "✅ Workaround setup complete!"
echo "💡 This allows vLLM V1 to find the expected config and tensor files."
echo "💡 Actual LoRA weights are still passed as tensors via TensorLoRARequest at runtime."
