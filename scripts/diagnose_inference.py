"""
Diagnostic script to identify why the model produces gibberish output.
This script checks various potential issues:
1. Tokenizer configuration mismatch
2. Model architecture/config mismatch
3. Weight loading issues
4. Logits distribution issues
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine


def check_tokenizer(model, tokenizer):
    """Check if tokenizer matches model vocab size"""
    print("\n" + "="*80)
    print("TOKENIZER CHECK")
    print("="*80)
    
    vocab_size = tokenizer.get_vocab_size()
    model_vocab_size = model.config.vocab_size
    
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Model vocab size: {model_vocab_size}")
    
    if vocab_size != model_vocab_size:
        print(f"❌ MISMATCH: Tokenizer vocab size ({vocab_size}) != Model vocab size ({model_vocab_size})")
        return False
    else:
        print("✓ Tokenizer vocab size matches model")
    
    # Test encoding/decoding
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest encoding/decoding:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded: {encoded[:10]}... (showing first 10 tokens)")
    print(f"  Decoded: '{decoded}'")
    
    if test_text not in decoded:
        print(f"⚠️  WARNING: Round-trip encoding/decoding doesn't match exactly")
    
    return True


def check_model_weights(model):
    """Check if model weights are properly initialized"""
    print("\n" + "="*80)
    print("MODEL WEIGHTS CHECK")
    print("="*80)
    
    # Check for NaN or Inf values
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Inf found in {name}")
            has_inf = True
    
    if not has_nan and not has_inf:
        print("✓ No NaN or Inf values found in model weights")
    
    # Check embedding layer
    embed_weight = model.embed_tokens.weight
    lm_head_weight = model.lm_head.weight
    
    print(f"\nEmbedding layer stats:")
    print(f"  Shape: {embed_weight.shape}")
    print(f"  Mean: {embed_weight.mean().item():.6f}")
    print(f"  Std: {embed_weight.std().item():.6f}")
    print(f"  Min: {embed_weight.min().item():.6f}")
    print(f"  Max: {embed_weight.max().item():.6f}")
    
    print(f"\nLM Head layer stats:")
    print(f"  Shape: {lm_head_weight.shape}")
    print(f"  Mean: {lm_head_weight.mean().item():.6f}")
    print(f"  Std: {lm_head_weight.std().item():.6f}")
    print(f"  Min: {lm_head_weight.min().item():.6f}")
    print(f"  Max: {lm_head_weight.max().item():.6f}")
    
    return not (has_nan or has_inf)


def check_inference_logits(model, tokenizer, device):
    """Check logits distribution during inference"""
    print("\n" + "="*80)
    print("INFERENCE LOGITS CHECK")
    print("="*80)
    
    # Simple test prompt
    test_prompt = "<|bos|><|user_start|>Hello<|user_end|><|assistant_start|>"
    tokens = tokenizer.encode(test_prompt)
    
    print(f"Test prompt: '{test_prompt}'")
    print(f"Encoded tokens: {tokens[:20]}... (showing first 20)")
    
    # Convert to tensor
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model.forward(input_ids)  # (B, T, vocab_size)
        last_logits = logits[0, -1, :]  # Last token logits
    
    print(f"\nLast token logits stats:")
    print(f"  Shape: {last_logits.shape}")
    print(f"  Mean: {last_logits.mean().item():.6f}")
    print(f"  Std: {last_logits.std().item():.6f}")
    print(f"  Min: {last_logits.min().item():.6f}")
    print(f"  Max: {last_logits.max().item():.6f}")
    
    # Check for NaN/Inf
    if torch.isnan(last_logits).any():
        print("❌ NaN found in logits!")
        return False
    if torch.isinf(last_logits).any():
        print("❌ Inf found in logits!")
        return False
    
    # Check if logits are too uniform (bad sign)
    probs = torch.softmax(last_logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum()
    max_prob = probs.max().item()
    
    print(f"\nProbability distribution:")
    print(f"  Entropy: {entropy.item():.4f} (higher = more uniform, lower = more peaked)")
    print(f"  Max probability: {max_prob:.6f}")
    print(f"  Top-5 probabilities: {torch.topk(probs, 5).values.tolist()}")
    
    # Check top tokens
    top_k = 10
    top_probs, top_indices = torch.topk(probs, top_k)
    print(f"\nTop-{top_k} predicted tokens:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        print(f"  {i+1}. Token {idx.item()}: '{token_str}' (prob: {prob.item():.6f})")
    
    # If entropy is very high (>10), logits might be too uniform
    if entropy.item() > 10:
        print(f"\n⚠️  WARNING: High entropy ({entropy.item():.4f}) suggests logits are too uniform")
        print("   This could indicate:")
        print("   - Model weights not properly trained")
        print("   - Incorrect model initialization")
        print("   - Missing or incorrect softcap application")
    
    return True


def check_config(model):
    """Check model configuration"""
    print("\n" + "="*80)
    print("MODEL CONFIG CHECK")
    print("="*80)
    
    config = model.config
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of heads: {config.n_head}")
    print(f"KV heads: {config.n_kv_head}")
    print(f"Sequence length: {config.sequence_len}")
    
    # Check softcap
    if hasattr(config, 'softcap'):
        print(f"Softcap: {config.softcap}")
        if config.softcap is None:
            print("⚠️  WARNING: Softcap is None, but model code expects a value")
    else:
        print("❌ ERROR: Config missing 'softcap' parameter")
        return False
    
    return True


def test_generation(model, tokenizer, device, engine):
    """Test actual generation"""
    print("\n" + "="*80)
    print("GENERATION TEST")
    print("="*80)
    
    prompt = "<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>"
    tokens = tokenizer.encode(prompt)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating with temperature=0.7, top_k=50...")
    
    generated_tokens = []
    try:
        for token_column, token_masks in engine.generate(tokens, num_samples=1, max_tokens=20, temperature=0.7, top_k=50):
            token = token_column[0]
            generated_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        print()
        
        full_text = tokenizer.decode(generated_tokens)
        print(f"\nGenerated text: '{full_text}'")
        
        # Check for repeated tokens (bad sign)
        if len(generated_tokens) > 1:
            unique_tokens = len(set(generated_tokens))
            repetition_rate = 1 - (unique_tokens / len(generated_tokens))
            print(f"Token repetition rate: {repetition_rate:.2%}")
            
            if repetition_rate > 0.5:
                print("⚠️  WARNING: High token repetition suggests model is stuck")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("NANOCHAT INFERENCE DIAGNOSTICS")
    print("="*80)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose inference issues')
    parser.add_argument('-i', '--source', type=str, default="base", help="Model source: base|mid|sft|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    args = parser.parse_args()
    
    # Initialize
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    
    print(f"\nLoading model from source: {args.source}")
    if args.model_tag:
        print(f"Model tag: {args.model_tag}")
    if args.step:
        print(f"Step: {args.step}")
    
    try:
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
        print(f"✓ Model loaded successfully")
        if meta:
            print(f"  Step: {meta.get('step', 'N/A')}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run diagnostics
    results = {}
    
    results['config'] = check_config(model)
    results['tokenizer'] = check_tokenizer(model, tokenizer)
    results['weights'] = check_model_weights(model)
    
    with autocast_ctx:
        results['logits'] = check_inference_logits(model, tokenizer, device)
    
    engine = Engine(model, tokenizer)
    with autocast_ctx:
        results['generation'] = test_generation(model, tokenizer, device, engine)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check.upper():20s}: {status}")
    
    if not all_passed:
        print("\n⚠️  SOME CHECKS FAILED - This likely explains the gibberish output")
        print("\nRecommendations:")
        if not results['tokenizer']:
            print("  1. Rebuild tokenizer from original training data")
        if not results['weights']:
            print("  2. Check model weights for corruption or incorrect loading")
        if not results['logits']:
            print("  3. Model may not be properly trained - consider retraining")
        if not results['config']:
            print("  4. Fix configuration file (add missing parameters)")
    else:
        print("\n✓ All checks passed - if you're still seeing gibberish, the issue may be:")
        print("  1. Model was not fully trained")
        print("  2. Data mismatch during training")
        print("  3. Inference-time tokenization differences")


if __name__ == "__main__":
    main()

