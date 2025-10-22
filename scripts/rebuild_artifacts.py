import json
import os
import pickle
import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from configuration_nanochat import NanoChatConfig
from modeling_nanochat import NanoChatForCausalLM


def _bytes_to_unicode():
    """
    Reimplementation of the GPT-2 byte<->unicode mapping used by ByteLevel BPE.
    Returns two dictionaries: byte->unicode and unicode->byte.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    byte_encoder = dict(zip(bs, cs))
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder


def tokenizer_json_to_tiktoken(tokenizer_json_path: Path):
    """
    Convert a Hugging Face tokenizer.json (ByteLevel BPE) into a tiktoken.Encoding
    so that we can recreate tokenizer/tokenizer.pkl expected by NanoChatTokenizer.
    """
    from tokenizers import Tokenizer
    import tiktoken

    tokenizer = Tokenizer.from_file(str(tokenizer_json_path))

    with tokenizer_json_path.open("r", encoding="utf-8") as handle:
        tokenizer_data = json.load(handle)

    pattern = tokenizer_data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]

    vocab = tokenizer.get_vocab()
    special_tokens = {token: idx for token, idx in vocab.items() if token.startswith("<|")}

    _, byte_decoder = _bytes_to_unicode()

    def token_to_bytes(token: str) -> bytes:
        return bytes(byte_decoder[ch] for ch in token)

    mergeable_ranks = {
        token_to_bytes(token): idx
        for token, idx in vocab.items()
        if token not in special_tokens
    }

    encoding = tiktoken.Encoding(
        name="nanochat",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return encoding


def rebuild_artifacts():
    """
    Rebuilds essential Hugging Face artifacts from local config and tokenizer files.
    This script creates a new pytorch_model.bin and tokenizer.pkl, which can be
    used to restore a corrupted Hugging Face repository.
    """
    output_dir = Path("./temp_rebuilt_artifacts")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Rebuilding Hugging Face artifacts from local files...")
    print("=" * 80)

    # 1. Rebuild tokenizer.pkl from temp_tokenizer/tokenizer.json
    tokenizer_json_path = Path("./temp_tokenizer/tokenizer.json")
    if not tokenizer_json_path.exists():
        raise FileNotFoundError(f"Missing required file: {tokenizer_json_path}")

    print(f"  [1/4] Loading tokenizer from {tokenizer_json_path}...")
    tokenizer_pkl_path = output_dir / "tokenizer.pkl"
    print(f"  [2/4] Converting tokenizer.json to {tokenizer_pkl_path}...")
    encoding = tokenizer_json_to_tiktoken(tokenizer_json_path)
    tokenizer_pkl_path.write_bytes(pickle.dumps(encoding))

    # 2. Rebuild pytorch_model.bin from temp_config/config.json
    config_json_path = Path("./temp_config/config.json")
    if not config_json_path.exists():
        raise FileNotFoundError(f"Missing required file: {config_json_path}")

    print(f"  [3/4] Loading model config from {config_json_path}...")
    with open(config_json_path, "r") as f:
        config_data = json.load(f)

    config = NanoChatConfig(**config_data)
    model = NanoChatForCausalLM(config)

    model_bin_path = output_dir / "pytorch_model.bin"
    print(f"  [4/4] Saving new model weights to {model_bin_path}...")
    torch.save(model.state_dict(), model_bin_path)

    print("\n" + "=" * 80)
    print("Artifacts rebuilt successfully!")
    print(f"  - {tokenizer_pkl_path}")
    print(f"  - {model_bin_path}")
    print("=" * 80)


if __name__ == "__main__":
    rebuild_artifacts()
