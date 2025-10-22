import json
import sys
from pathlib import Path

import torch

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration_nanochat import NanoChatConfig
from modeling_nanochat import NanoChatForCausalLM


def rebuild_model():
    """
    Rebuilds the pytorch_model.bin from the local config.json file.
    """
    output_dir = Path("./temp_rebuilt_artifacts")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Rebuilding pytorch_model.bin from local config...")
    print("=" * 80)

    config_json_path = Path("./temp_config/config.json")
    if not config_json_path.exists():
        raise FileNotFoundError(f"Missing required file: {config_json_path}")

    print(f"  [1/2] Loading model config from {config_json_path}...")
    with open(config_json_path, "r") as f:
        config_data = json.load(f)

    config = NanoChatConfig(**config_data)
    model = NanoChatForCausalLM(config)

    model_bin_path = output_dir / "pytorch_model.bin"
    print(f"  [2/2] Saving new model weights to {model_bin_path}...")
    torch.save(model.state_dict(), model_bin_path)

    print("\n" + "=" * 80)
    print("Model rebuilt successfully!")
    print(f"  - {model_bin_path}")
    print("=" * 80)


if __name__ == "__main__":
    rebuild_model()


