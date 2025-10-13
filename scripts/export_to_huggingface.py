"""
Export nanochat model to HuggingFace format for deployment on HF Inference Endpoints

Usage:
    python scripts/export_to_huggingface.py --source sft --output-dir ./hf_model
    
Then upload to HuggingFace:
    huggingface-cli login
    huggingface-cli upload your-username/model-name ./hf_model
"""

import argparse
import json
import os
import torch
from pathlib import Path

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init


def convert_nanochat_to_hf(model, tokenizer, meta, output_dir):
    """
    Convert nanochat model to HuggingFace format.
    
    nanochat uses a custom GPT architecture similar to GPT-2 but with:
    - RoPE (Rotary Position Embeddings) instead of learned positional embeddings
    - RMSNorm instead of LayerNorm
    - Untied embeddings
    - ReLU^2 activation
    - QK norm
    - Multi-Query Attention (MQA)
    
    We'll create a custom HF config and model class.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = model.config
    
    print(f"Converting model with config:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Heads: {config.n_head}")
    print(f"  KV Heads: {config.n_kv_head}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Max sequence length: {config.sequence_len}")
    
    # Create HuggingFace-style config
    hf_config = {
        "architectures": ["NanoChatGPTModel"],
        "model_type": "nanochat-gpt",
        "vocab_size": config.vocab_size,
        "n_positions": config.sequence_len,
        "n_embd": config.n_embd,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "activation_function": "relu_squared",
        "use_rope": True,
        "use_qk_norm": True,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.36.0",
    }
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(hf_config, f, indent=2)
    print(f"✓ Saved config to {config_path}")
    
    # Save model weights
    # HuggingFace expects pytorch_model.bin or model.safetensors
    state_dict = model.state_dict()
    
    # Remove non-persistent buffers (cos/sin are computed on-the-fly)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('cos') and not k.startswith('sin')}
    
    model_path = output_dir / "pytorch_model.bin"
    torch.save(state_dict, model_path)
    print(f"✓ Saved model weights to {model_path}")
    
    # Save tokenizer files
    save_tokenizer(tokenizer, output_dir)
    
    # Create modeling file
    create_modeling_file(output_dir)
    
    # Create README
    create_readme(hf_config, meta, output_dir)
    
    print(f"\n✓ Model exported successfully to {output_dir}")
    print(f"\nNext steps:")
    print(f"1. Test the model locally:")
    print(f"   python scripts/test_hf_model.py --model-dir {output_dir}")
    print(f"\n2. Upload to HuggingFace:")
    print(f"   huggingface-cli login")
    print(f"   huggingface-cli upload your-username/model-name {output_dir}")
    print(f"\n3. Deploy on HF Inference Endpoints via the web UI")


def save_tokenizer(tokenizer, output_dir):
    """Save tokenizer in HuggingFace format."""
    
    # Get vocab
    vocab = {tokenizer.decode([i]): i for i in range(tokenizer.get_vocab_size())}
    
    # Save vocab.json
    vocab_path = output_dir / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved vocabulary to {vocab_path}")
    
    # Save tokenizer config
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_type": "nanochat-gpt",
        "bos_token": tokenizer.decode([tokenizer.get_bos_token_id()]),
        "eos_token": tokenizer.decode([tokenizer.get_bos_token_id()]),  # nanochat uses bos as eos
        "unk_token": tokenizer.decode([tokenizer.get_bos_token_id()]),
        "clean_up_tokenization_spaces": False,
    }
    
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"✓ Saved tokenizer config to {tokenizer_config_path}")
    
    # Save special tokens map
    special_tokens = {
        "bos_token": tokenizer.decode([tokenizer.get_bos_token_id()]),
        "eos_token": tokenizer.decode([tokenizer.get_bos_token_id()]),
        "unk_token": tokenizer.decode([tokenizer.get_bos_token_id()]),
    }
    
    special_tokens_path = output_dir / "special_tokens_map.json"
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens, f, indent=2)
    print(f"✓ Saved special tokens to {special_tokens_path}")


def create_modeling_file(output_dir):
    """Create modeling_nanochat_gpt.py for HuggingFace compatibility."""
    
    modeling_code = '''"""
NanoChatGPT model for HuggingFace Transformers.

This is a wrapper around the nanochat GPT implementation to make it
compatible with HuggingFace Inference Endpoints.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


class NanoChatGPTConfig(PretrainedConfig):
    model_type = "nanochat-gpt"
    
    def __init__(
        self,
        vocab_size=50304,
        n_positions=2048,
        n_embd=768,
        n_layer=12,
        n_head=6,
        n_kv_head=6,
        activation_function="relu_squared",
        use_rope=True,
        use_qk_norm=True,
        tie_word_embeddings=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.activation_function = activation_function
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.tie_word_embeddings = tie_word_embeddings


class NanoChatGPTModel(PreTrainedModel):
    config_class = NanoChatGPTConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Import nanochat GPT here to avoid circular imports
        from nanochat.gpt import GPT, GPTConfig
        
        # Create nanochat config
        gpt_config = GPTConfig(
            sequence_len=config.n_positions,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd
        )
        
        # Create the actual model
        self.model = GPT(gpt_config)
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # nanochat's forward returns loss during training, logits during inference
        logits = self.model(input_ids, targets=None)
        return {"logits": logits}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, **kwargs):
        """Generate text using nanochat's generate method."""
        # Convert to list for nanochat
        tokens = input_ids[0].tolist()
        
        # Use nanochat's generator
        generated = []
        for token in self.model.generate(tokens, max_tokens=max_length-len(tokens), 
                                        temperature=temperature, top_k=top_k):
            generated.append(token)
        
        # Convert back to tensor
        all_tokens = tokens + generated
        return torch.tensor([all_tokens])
'''
    
    modeling_path = output_dir / "modeling_nanochat_gpt.py"
    with open(modeling_path, 'w') as f:
        f.write(modeling_code)
    print(f"✓ Created modeling file at {modeling_path}")


def create_readme(config, meta, output_dir):
    """Create model card README.md."""
    
    num_params = sum(config['n_embd'] * config['n_embd'] * 4 for _ in range(config['n_layer']))
    num_params += config['vocab_size'] * config['n_embd'] * 2  # embeddings
    num_params_m = num_params / 1e6
    
    readme = f"""---
license: mit
language:
- en
tags:
- nanochat
- gpt
- text-generation
inference: true
---

# NanoChat GPT Model

This model was trained using [nanochat](https://github.com/karpathy/nanochat), a minimal, hackable implementation of a full-stack ChatGPT clone.

## Model Details

- **Architecture**: Custom GPT with RoPE, RMSNorm, and Multi-Query Attention
- **Parameters**: ~{num_params_m:.1f}M
- **Layers**: {config['n_layer']}
- **Embedding Dimension**: {config['n_embd']}
- **Attention Heads**: {config['n_head']} (KV heads: {config['n_kv_head']})
- **Context Length**: {config['n_positions']} tokens
- **Training**: Trained on Hyperbolic Labs GPU infrastructure

## Key Features

- **RoPE**: Rotary Position Embeddings for better length generalization
- **RMSNorm**: More efficient than LayerNorm
- **MQA**: Multi-Query Attention for efficient inference
- **Untied Embeddings**: Separate input and output embeddings
- **ReLU²**: Squared ReLU activation in MLP

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/model-name", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Training Details

This model was trained using nanochat's efficient training pipeline:
- Optimizer: Muon for linear layers, AdamW for embeddings
- Precision: BF16
- Training Infrastructure: Hyperbolic Labs

## Limitations

This is a small language model trained on limited compute. It has the following limitations:
- May generate incorrect or nonsensical information
- Limited knowledge cutoff
- Not aligned for safety
- Should not be used for production applications without further fine-tuning

## Citation

```bibtex
@misc{{nanochat,
  author = {{Andrej Karpathy}},
  title = {{nanochat: The best ChatGPT that $100 can buy}},
  year = {{2025}},
  publisher = {{GitHub}},
  url = {{https://github.com/karpathy/nanochat}}
}}
```
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"✓ Created README at {readme_path}")


def main():
    parser = argparse.ArgumentParser(description='Export nanochat model to HuggingFace format')
    parser.add_argument('-i', '--source', type=str, default="sft", 
                       help="Source of the model: sft|mid|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None, 
                       help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, 
                       help='Step to load')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Output directory for HuggingFace model')
    args = parser.parse_args()
    
    print("=" * 80)
    print("NanoChat to HuggingFace Converter")
    print("=" * 80)
    
    # Initialize
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    
    # Load model
    print(f"\nLoading nanochat model from '{args.source}'...")
    model, tokenizer, meta = load_model(
        args.source, 
        device, 
        phase="eval",
        model_tag=args.model_tag, 
        step=args.step
    )
    print("✓ Model loaded successfully")
    
    # Convert and save
    print(f"\nConverting to HuggingFace format...")
    convert_nanochat_to_hf(model, tokenizer, meta, args.output_dir)
    

if __name__ == "__main__":
    main()
