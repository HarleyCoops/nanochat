"""
Generate tokenizer.pkl for HuggingFace deployment.
This creates the pickle file needed by the custom NanoChatTokenizer.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanochat.tokenizer import get_tokenizer


def main():
    print("=" * 80)
    print("Generating tokenizer.pkl for HuggingFace deployment")
    print("=" * 80)
    
    # Get the nanochat tokenizer
    print("\n[+] Loading nanochat tokenizer...")
    tokenizer = get_tokenizer()
    
    # Create output directory
    output_dir = ROOT / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the tokenizer
    output_file = output_dir / "tokenizer.pkl"
    print(f"[+] Saving tokenizer to {output_file}...")
    tokenizer.save(output_dir)
    
    # Verify the file was created
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # Convert to MB
        print(f"[+] Success! Created {output_file} ({file_size:.2f} MB)")
        print("\n" + "=" * 80)
        print("Next steps:")
        print("=" * 80)
        print("1. Upload tokenization_nanochat.py to your HF model repo:")
        print("   huggingface-cli upload HarleyCooper/nanochat561 tokenization_nanochat.py --repo-type model")
        print("\n2. Upload the tokenizer directory:")
        print("   huggingface-cli upload HarleyCooper/nanochat561 tokenizer --repo-type model")
        print("\n3. The HF Space deployment should now work!")
    else:
        print("[!] Error: Failed to create tokenizer.pkl")
        sys.exit(1)


if __name__ == "__main__":
    main()
