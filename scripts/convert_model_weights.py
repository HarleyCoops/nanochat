"""
Convert nanochat model weights from training format to HuggingFace format.
This script downloads the existing pytorch_model.bin, converts weight names, and re-uploads.
"""

import torch
from huggingface_hub import hf_hub_download, HfApi
import tempfile
import os


def convert_weight_name(old_name: str) -> str:
    """Convert old nanochat weight names to HuggingFace format."""
    # Handle embeddings
    if old_name == 'transformer.wte.weight':
        return 'model.embed_tokens.weight'
    if old_name == 'lm_head.weight':
        return 'model.lm_head.weight'
    
    # Handle transformer blocks: transformer.h.{i}.* -> model.blocks.{i}.*
    if old_name.startswith('transformer.h.'):
        parts = old_name.split('.')
        layer_num = parts[2]  # Get layer number
        rest = '.'.join(parts[3:])  # Get rest of path
        
        # Convert attention weight names
        rest = rest.replace('attn.c_q.weight', 'attn.q_proj.weight')
        rest = rest.replace('attn.c_k.weight', 'attn.k_proj.weight')
        rest = rest.replace('attn.c_v.weight', 'attn.v_proj.weight')
        rest = rest.replace('attn.c_proj.weight', 'attn.out_proj.weight')
        
        # Convert MLP weight names
        rest = rest.replace('mlp.c_fc.weight', 'mlp.fc.weight')
        rest = rest.replace('mlp.c_proj.weight', 'mlp.proj.weight')
        
        return f'model.blocks.{layer_num}.{rest}'
    
    # Return unchanged if no conversion needed
    return old_name


def main():
    model_id = "HarleyCooper/nanochat561"
    
    print("=" * 80)
    print("Converting nanochat model weights to HuggingFace format")
    print("=" * 80)
    
    # Download the current model
    print(f"\n[1/4] Downloading pytorch_model.bin from {model_id}...")
    model_path = hf_hub_download(
        repo_id=model_id,
        filename="pytorch_model.bin",
        repo_type="model"
    )
    
    print(f"[2/4] Loading model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    
    print(f"[3/4] Converting {len(state_dict)} weight tensors...")
    converted_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = convert_weight_name(old_key)
        converted_state_dict[new_key] = value
        if old_key != new_key:
            print(f"  {old_key} -> {new_key}")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
        temp_path = tmp.name
        print(f"\n[4/4] Saving converted weights to {temp_path}...")
        torch.save(converted_state_dict, temp_path)
    
    # Upload to HuggingFace
    print(f"\n[5/5] Uploading converted model to {model_id}...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=temp_path,
        path_in_repo="pytorch_model.bin",
        repo_id=model_id,
        repo_type="model",
        commit_message="Convert weight names to HuggingFace format"
    )
    
    # Cleanup
    os.unlink(temp_path)
    
    print("\n" + "=" * 80)
    print("✓ Conversion complete!")
    print("=" * 80)
    print("\nThe model weights have been converted and uploaded.")
    print("Your HuggingFace Space should now load the trained weights correctly.")


if __name__ == "__main__":
    main()
