#!/usr/bin/env python3
"""
Automated HuggingFace Space Deployment Script for NanoChat

This script automates the deployment of NanoChat to HuggingFace Spaces,
including setting up the Space, uploading all necessary files, and configuring
the model for inference.

Usage:
    python scripts/deploy_hf_space.py --space-name my-nanochat-demo
    python scripts/deploy_hf_space.py --space-name my-nanochat --org my-org --private
    python scripts/deploy_hf_space.py --space-name my-nanochat --hardware cpu-upgrade
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi, create_repo, upload_folder, whoami
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


class HFSpaceDeployer:
    """Handles deployment of NanoChat to HuggingFace Spaces."""

    def __init__(
        self,
        space_name: str,
        model_id: str = "HarleyCooper/nanochat561",
        organization: Optional[str] = None,
        private: bool = False,
        hardware: str = "cpu-basic",
    ):
        self.space_name = space_name
        self.model_id = model_id
        self.organization = organization
        self.private = private
        self.hardware = hardware
        self.api = HfApi()

        # Determine repo_id
        if organization:
            self.repo_id = f"{organization}/{space_name}"
        else:
            try:
                user_info = whoami()
                username = user_info['name']
                self.repo_id = f"{username}/{space_name}"
            except Exception as e:
                print(f"Error: Could not determine username. Are you logged in?")
                print(f"Run: huggingface-cli login")
                sys.exit(1)

        # Get project root
        self.project_root = Path(__file__).parent.parent
        self.space_dir = self.project_root / "deploy" / "hf_space"

    def verify_login(self):
        """Verify that user is logged into HuggingFace."""
        print("Verifying HuggingFace login...")
        try:
            user_info = whoami()
            print(f"✓ Logged in as: {user_info['name']}")
            return True
        except Exception:
            print("✗ Not logged into HuggingFace")
            print("\nPlease login with: huggingface-cli login")
            print("Get your token from: https://huggingface.co/settings/tokens")
            return False

    def verify_files(self):
        """Verify that all necessary files exist."""
        print("\nVerifying deployment files...")

        required_files = [
            "app.py",
            "requirements.txt",
            "README.md",
            "configuration_nanochat.py",
            "modeling_nanochat.py",
        ]

        missing_files = []
        for file in required_files:
            file_path = self.space_dir / file
            if not file_path.exists():
                missing_files.append(file)
            else:
                print(f"✓ Found {file}")

        if missing_files:
            print(f"\n✗ Missing required files: {', '.join(missing_files)}")
            print(f"Expected location: {self.space_dir}")
            return False

        print("✓ All required files present")
        return True

    def create_space(self):
        """Create the HuggingFace Space."""
        print(f"\nCreating HuggingFace Space: {self.repo_id}")

        try:
            # Check if space already exists
            try:
                self.api.repo_info(repo_id=self.repo_id, repo_type="space")
                print(f"⚠ Space {self.repo_id} already exists")
                response = input("Do you want to update it? (y/n): ")
                if response.lower() != 'y':
                    print("Deployment cancelled")
                    return False
                print("Will update existing Space...")
                return True
            except RepositoryNotFoundError:
                # Space doesn't exist, create it
                pass

            # Create the Space
            create_repo(
                repo_id=self.repo_id,
                repo_type="space",
                space_sdk="gradio",
                private=self.private,
                exist_ok=True
            )
            print(f"✓ Created Space: {self.repo_id}")
            return True

        except Exception as e:
            print(f"✗ Error creating Space: {e}")
            return False

    def configure_space_metadata(self):
        """Update the README.md with proper Space metadata."""
        print("\nConfiguring Space metadata...")

        readme_path = self.space_dir / "README.md"

        # Read existing README
        with open(readme_path, 'r') as f:
            content = f.read()

        # Check if it already has frontmatter
        if content.startswith('---'):
            # Extract existing frontmatter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Keep body, update frontmatter
                body = parts[2]
            else:
                body = content
        else:
            body = content

        # Create updated frontmatter
        hardware_map = {
            "cpu-basic": "cpu-basic",
            "cpu-upgrade": "cpu-upgrade",
            "t4-small": "t4-small",
            "t4-medium": "t4-medium",
            "a10g-small": "a10g-small",
            "a10g-large": "a10g-large",
        }

        suggested_hardware = hardware_map.get(self.hardware, "cpu-basic")

        frontmatter = f"""---
title: NanoChat 561M
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
suggested_hardware: {suggested_hardware}
models:
  - {self.model_id}
---
"""

        # Write updated README
        with open(readme_path, 'w') as f:
            f.write(frontmatter + body)

        print(f"✓ Updated README.md with Space metadata")
        print(f"  Hardware: {suggested_hardware}")
        print(f"  Model: {self.model_id}")

    def upload_files(self):
        """Upload all files to the Space."""
        print(f"\nUploading files to Space...")

        try:
            upload_folder(
                folder_path=str(self.space_dir),
                repo_id=self.repo_id,
                repo_type="space",
                commit_message="Deploy NanoChat inference Space",
            )
            print(f"✓ Successfully uploaded files to {self.repo_id}")
            return True

        except Exception as e:
            print(f"✗ Error uploading files: {e}")
            return False

    def set_hardware(self):
        """Set the hardware tier for the Space."""
        if self.hardware == "cpu-basic":
            print("\nUsing default hardware (cpu-basic)")
            print("You can change this in the Space Settings later")
            return True

        print(f"\nSetting hardware to: {self.hardware}")
        print("Note: This requires a paid plan on HuggingFace")

        try:
            # Note: Hardware settings are typically done via the web UI or require special permissions
            # We'll just inform the user here
            print("⚠ Hardware tier must be set manually in Space Settings:")
            print(f"   https://huggingface.co/spaces/{self.repo_id}/settings")
            return True

        except Exception as e:
            print(f"⚠ Could not set hardware automatically: {e}")
            print(f"Please set it manually at: https://huggingface.co/spaces/{self.repo_id}/settings")
            return True

    def deploy(self):
        """Run the complete deployment process."""
        print("=" * 70)
        print("NanoChat HuggingFace Space Deployment")
        print("=" * 70)

        # Step 1: Verify login
        if not self.verify_login():
            return False

        # Step 2: Verify files
        if not self.verify_files():
            return False

        # Step 3: Configure metadata
        self.configure_space_metadata()

        # Step 4: Create Space
        if not self.create_space():
            return False

        # Step 5: Upload files
        if not self.upload_files():
            return False

        # Step 6: Set hardware
        self.set_hardware()

        # Success!
        print("\n" + "=" * 70)
        print("✓ Deployment Complete!")
        print("=" * 70)
        print(f"\nYour Space is available at:")
        print(f"  https://huggingface.co/spaces/{self.repo_id}")
        print(f"\nThe Space is building now. This may take 5-10 minutes.")
        print(f"Check build status at:")
        print(f"  https://huggingface.co/spaces/{self.repo_id}/logs")
        print(f"\nOnce built, you can chat with your model at:")
        print(f"  https://huggingface.co/spaces/{self.repo_id}")

        if self.hardware != "cpu-basic":
            print(f"\n⚠ Remember to set hardware to '{self.hardware}' in Space Settings:")
            print(f"  https://huggingface.co/spaces/{self.repo_id}/settings")

        print("\n" + "=" * 70)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Deploy NanoChat to HuggingFace Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deployment
  python scripts/deploy_hf_space.py --space-name my-nanochat-demo

  # Deploy to an organization
  python scripts/deploy_hf_space.py --space-name nanochat-demo --org my-org

  # Private space with GPU
  python scripts/deploy_hf_space.py --space-name my-nanochat --private --hardware t4-small

  # Use a different model
  python scripts/deploy_hf_space.py --space-name my-nanochat --model-id username/my-nanochat-model

Hardware options: cpu-basic, cpu-upgrade, t4-small, t4-medium, a10g-small, a10g-large
        """
    )

    parser.add_argument(
        "--space-name",
        type=str,
        required=True,
        help="Name for your HuggingFace Space (e.g., 'nanochat-demo')"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="HarleyCooper/nanochat561",
        help="HuggingFace model ID to use (default: HarleyCooper/nanochat561)"
    )

    parser.add_argument(
        "--org",
        "--organization",
        type=str,
        default=None,
        help="Deploy to an organization instead of your personal account"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Space private (requires paid plan)"
    )

    parser.add_argument(
        "--hardware",
        type=str,
        default="cpu-basic",
        choices=["cpu-basic", "cpu-upgrade", "t4-small", "t4-medium", "a10g-small", "a10g-large"],
        help="Hardware tier for the Space (default: cpu-basic)"
    )

    args = parser.parse_args()

    # Create deployer and run
    deployer = HFSpaceDeployer(
        space_name=args.space_name,
        model_id=args.model_id,
        organization=args.org,
        private=args.private,
        hardware=args.hardware,
    )

    success = deployer.deploy()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
