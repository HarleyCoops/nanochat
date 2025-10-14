"""
Export nanochat checkpoints to a Hugging Face Hub-compatible directory.

Example:
    python scripts/export_to_huggingface.py --source sft --output-dir ./hf_model
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from configuration_nanochat import NanoChatConfig
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init


def convert_nanochat_to_hf(model, tokenizer, meta: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt_config = model.config
    hf_config = NanoChatConfig(
        vocab_size=gpt_config.vocab_size,
        sequence_len=gpt_config.sequence_len,
        n_layer=gpt_config.n_layer,
        n_head=gpt_config.n_head,
        n_kv_head=gpt_config.n_kv_head,
        n_embd=gpt_config.n_embd,
        bos_token_id=tokenizer.get_bos_token_id(),
        eos_token_id=tokenizer.get_bos_token_id(),
    ).to_dict()
    hf_config["architectures"] = ["NanoChatForCausalLM"]
    hf_config["torch_dtype"] = "bfloat16"

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(hf_config, indent=2), encoding="utf-8")
    print(f"[+] Wrote config to {config_path}")

    state_dict = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(("cos", "sin"))
    }
    model_path = output_dir / "pytorch_model.bin"
    torch.save(state_dict, model_path)
    print(f"[+] Saved model weights to {model_path}")

    save_tokenizer(tokenizer, output_dir)
    copy_support_code(output_dir)
    create_generation_config(output_dir)
    create_readme(hf_config, meta, output_dir)

    print(f"[+] Export complete. Files written to {output_dir}")
    print("Next steps:")
    print("  huggingface-cli login")
    print(f"  huggingface-cli upload YOUR-USERNAME/MODEL-NAME {output_dir}")


def save_tokenizer(tokenizer, output_dir: Path) -> None:
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tokenizer_dir)

    tokenizer_config = {
        "tokenizer_class": "NanoChatTokenizer",
        "model_type": "nanochat",
        "bos_token": "<|bos|>",
        "eos_token": "<|bos|>",
        "unk_token": "<|bos|>",
        "clean_up_tokenization_spaces": False,
        "tokenizer_file": "tokenizer/tokenizer.pkl",
    }

    (output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|bos|>",
                "eos_token": "<|bos|>",
                "unk_token": "<|bos|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Serialized tokenizer to {tokenizer_dir}")

    requirements_path = output_dir / "requirements.txt"
    requirements_path.write_text("tiktoken>=0.6.0\n", encoding="utf-8")
    print(f"[+] Added minimal requirements.txt for Hub runtime")


def copy_support_code(output_dir: Path) -> None:
    project_root = Path(__file__).resolve().parent.parent
    for filename in ("configuration_nanochat.py", "modeling_nanochat.py", "tokenization_nanochat.py"):
        src = project_root / filename
        dest = output_dir / filename
        shutil.copyfile(src, dest)
        print(f"[+] Copied {filename} to export directory")

    init_path = output_dir / "__init__.py"
    init_path.write_text(
        "from .configuration_nanochat import NanoChatConfig\n"
        "from .modeling_nanochat import NanoChatForCausalLM\n"
        "from .tokenization_nanochat import NanoChatTokenizer\n",
        encoding="utf-8",
    )
    print(f"[+] Wrote __init__.py to expose remote code")


def create_generation_config(output_dir: Path) -> None:
    generation_config = {
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
        "bos_token_id": 1,
        "eos_token_id": 1,
        "pad_token_id": 1,
    }
    (output_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2), encoding="utf-8")
    print(f"[+] Created generation_config.json")


def create_readme(config: Dict[str, Any], meta: Dict[str, Any], output_dir: Path) -> None:
    params_millions = meta.get("params", config["n_layer"] * config["n_embd"] ** 2 * 4) / 1e6
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

# nanochat (custom RoPE GPT)

This repository contains a nanochat checkpoint exported for Hugging Face. The
architecture mirrors GPT-style decoders with rotary embeddings, RMSNorm, multi-query
attention, relu² MLPs, and untied embeddings.

## Model Summary

- **Parameters**: ~{params_millions:.1f}M
- **Layers**: {config["n_layer"]}
- **Hidden size**: {config["n_embd"]}
- **Attention heads**: {config["n_head"]} (KV heads: {config["n_kv_head"]})
- **Context length**: {config["sequence_len"]}
- **Tokenizer**: nanochat rustbpe (BPE, 65k vocab)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HarleyCooper/nanochat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("HarleyCooper/nanochat", trust_remote_code=True)

inputs = tokenizer("Hello nanochat!", return_tensors="pt")
output_ids = model.generate(**inputs, max_length=128)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Remember to pass `trust_remote_code=True` so Transformers picks up the custom
model and tokenizer implementations bundled in this repository.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[+] Authored README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nanochat checkpoint to Hugging Face format")
    parser.add_argument("--source", type=str, default="sft", help="Checkpoint family to load (base|mid|sft|rl)")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional model tag to load")
    parser.add_argument("--step", type=int, default=None, help="Optional training step to load")
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory")
    args = parser.parse_args()

    print("=" * 80)
    print("nanochat → Hugging Face exporter")
    print("=" * 80)

    _, _, _, _, device = compute_init()

    print(f"[+] Loading checkpoint from '{args.source}' (step={args.step})...")
    model, tokenizer, meta = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    model.eval()
    print("[+] Model ready, beginning conversion...")

    convert_nanochat_to_hf(model, tokenizer, meta, Path(args.output_dir))


if __name__ == "__main__":
    main()
