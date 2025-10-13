"""
Publish model card to HuggingFace Hub

This script uploads the MODEL_CARD.md file to HuggingFace as README.md
for the nanochat model repository.

Usage:
    python scripts/publish_model_card.py --username harleycooper --repo nanochat
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install with: pip install huggingface-hub")
    exit(1)


def publish_model_card(username: str, repo_name: str, model_card_path: str = "MODEL_CARD.md"):
    """
    Publish model card to HuggingFace Hub.
    
    Args:
        username: HuggingFace username
        repo_name: Repository name (e.g., 'nanochat')
        model_card_path: Path to the model card file
    """
    # Initialize HF API
    api = HfApi()
    
    # Full repo ID
    repo_id = f"{username}/{repo_name}"
    
    print(f"Publishing model card to: https://huggingface.co/{repo_id}")
    
    # Check if model card exists
    model_card_path = Path(model_card_path)
    if not model_card_path.exists():
        print(f"Error: Model card not found at {model_card_path}")
        exit(1)
    
    try:
        # Create repository if it doesn't exist
        print(f"Creating/accessing repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"Repository ready: {repo_id}")
        
        # Upload model card as README.md
        print("Uploading model card as README.md...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add comprehensive model card documentation"
        )
        print("Model card uploaded successfully!")
        
        print(f"\n{'='*60}")
        print(f"Model card published at:")
        print(f"https://huggingface.co/{repo_id}")
        print(f"{'='*60}")
        
        print("\nNext steps:")
        print("1. Training is in progress - model weights will be added when complete")
        print("2. After training, run: python scripts/export_to_huggingface.py --source sft --output-dir ./hf_model")
        print(f"3. Then upload weights: huggingface-cli upload {repo_id} ./hf_model")
        
    except Exception as e:
        print(f"Error uploading model card: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print(f"3. Verify you have permissions for: {repo_id}")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Publish model card to HuggingFace Hub"
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="HuggingFace username"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="nanochat",
        help="Repository name (default: nanochat)"
    )
    parser.add_argument(
        "--model-card",
        type=str,
        default="MODEL_CARD.md",
        help="Path to model card file (default: MODEL_CARD.md)"
    )
    
    args = parser.parse_args()
    
    publish_model_card(
        username=args.username,
        repo_name=args.repo,
        model_card_path=args.model_card
    )


if __name__ == "__main__":
    main()
