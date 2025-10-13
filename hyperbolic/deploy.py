"""
Main deployment script for launching Nanochat training on Hyperbolic Labs

Usage:
    python hyperbolic/deploy.py --model-size medium --auto-launch
    python hyperbolic/deploy.py --depth 20 --max-seq-len 2048 --list-machines
    python hyperbolic/deploy.py --help
"""

import argparse
import json
import sys
import os
from pathlib import Path

from api_client import HyperbolicClient, GPUInstance
from deployment_config import (
    ModelConfig,
    calculate_gpu_requirements,
    get_deployment_command,
    SMALL_MODEL,
    MEDIUM_MODEL,
    LARGE_MODEL,
    XLARGE_MODEL
)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def print_machine_info(machine: GPUInstance):
    """Print formatted machine information"""
    available_gpus = machine.gpus_total - machine.gpus_reserved
    print(f"\n  Machine ID: {machine.id}")
    print(f"  Cluster: {machine.cluster_name}")
    print(f"  GPU Type: {machine.gpu_type}")
    print(f"  Available GPUs: {available_gpus}/{machine.gpus_total}")
    print(f"  RAM: {machine.ram_gb} GB")
    print(f"  Storage: {machine.storage_gb} GB")
    print(f"  Price: ${machine.pricing:.2f}/hour")
    print(f"  Status: {machine.status}")


def list_available_machines(client: HyperbolicClient, requirements):
    """List all available machines that meet requirements"""
    print_section("Available Machines on Hyperbolic Labs")
    
    machines = client.list_available_machines()
    
    # Filter by requirements
    suitable = []
    for machine in machines:
        if machine.reserved or machine.status != "node_ready":
            continue
        available_gpus = machine.gpus_total - machine.gpus_reserved
        if available_gpus >= requirements.min_gpus:
            suitable.append(machine)
    
    if not suitable:
        print("\n  No suitable machines available at this time.")
        print(f"  Required: {requirements.min_gpus}+ GPUs with {requirements.min_vram_gb}+ GB VRAM")
        return
    
    # Sort by preference and price
    def sort_key(m):
        pref_score = 999
        for i, pref in enumerate(requirements.gpu_type_preference):
            if pref.upper() in m.gpu_type.upper():
                pref_score = i
                break
        return (pref_score, m.pricing)
    
    suitable.sort(key=sort_key)
    
    print(f"\n  Found {len(suitable)} suitable machine(s):\n")
    for i, machine in enumerate(suitable[:10], 1):  # Show top 10
        print(f"{i}.")
        print_machine_info(machine)
    
    if len(suitable) > 10:
        print(f"\n  ... and {len(suitable) - 10} more")


def create_deployment_package():
    """Create a deployment package with setup instructions"""
    print_section("Creating Deployment Package")
    
    setup_script = """#!/bin/bash
# Nanochat Training Setup Script for Hyperbolic Labs

set -e

echo "Setting up Nanochat training environment..."

# Install system dependencies
apt-get update
apt-get install -y git python3-pip

# Clone repository (if not already present)
if [ ! -d "nanochat" ]; then
    git clone https://github.com/HarleyCoops/nanochat.git
    cd nanochat
else
    cd nanochat
    git pull
fi

# Install Python dependencies with uv (faster than pip)
pip install uv
uv pip install -e .

# Download tokenized data (if needed)
# You may need to set up your data directory structure here

echo "Setup complete! Ready to start training."
echo ""
echo "To start training, run:"
echo "  torchrun --nproc_per_node=<NUM_GPUS> scripts/base_train.py"
"""
    
    # Create hyperbolic deployment directory
    deploy_dir = Path("hyperbolic_deployment")
    deploy_dir.mkdir(exist_ok=True)
    
    setup_path = deploy_dir / "setup.sh"
    setup_path.write_text(setup_script)
    setup_path.chmod(0o755)
    
    print(f"\n  Created setup script: {setup_path}")
    print(f"\n  This script will be used to set up the environment on Hyperbolic Labs.")


def launch_training(
    client: HyperbolicClient,
    config: ModelConfig,
    requirements,
    max_price_per_hour: float = None,
    gpu_count: int = None
):
    """Launch training on Hyperbolic Labs"""
    print_section("Launching Training on Hyperbolic Labs")
    
    # Determine GPU count
    if gpu_count is None:
        gpu_count = requirements.recommended_gpus
    
    print(f"\n  Searching for machine with {gpu_count} GPUs...")
    
    # Find best machine
    machine = client.find_best_machine(
        min_gpus=gpu_count,
        gpu_type_preference=requirements.gpu_type_preference,
        max_price_per_hour=max_price_per_hour
    )
    
    if not machine:
        print("\n  ERROR: No suitable machine found!")
        print(f"  Required: {gpu_count} GPUs")
        if max_price_per_hour:
            print(f"  Max price: ${max_price_per_hour}/hour")
        print("\n  Try:")
        print("    - Reducing GPU count")
        print("    - Increasing max price")
        print("    - Using --list-machines to see available options")
        return False
    
    print("\n  Found suitable machine:")
    print_machine_info(machine)
    
    # Calculate cost estimate
    estimated_cost_per_hour = machine.pricing
    print(f"\n  Estimated cost: ${estimated_cost_per_hour:.2f}/hour")
    
    # Get user confirmation
    print("\n  This will create an instance and start billing.")
    response = input("  Continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n  Deployment cancelled.")
        return False
    
    # Create instance
    print("\n  Creating instance...")
    try:
        result = client.create_instance(
            cluster_name=machine.cluster_name,
            node_name=machine.node_name,
            gpu_count=gpu_count
        )
        
        print("\n  ✓ Instance created successfully!")
        print(f"\n  Instance ID: {result.get('instance_id', 'N/A')}")
        
        # Generate deployment command
        deploy_cmd = get_deployment_command(config, gpu_count)
        
        print_section("Next Steps")
        print("\n  1. SSH into your instance (check Hyperbolic dashboard for connection details)")
        print("\n  2. Run the setup script:")
        print("     bash setup.sh")
        print("\n  3. Start training:")
        print(f"     {deploy_cmd}")
        print("\n  4. Monitor training:")
        print("     - Check logs in the terminal")
        print("     - View metrics in WandB (if configured)")
        print("\n  5. When finished, terminate the instance:")
        print("     python hyperbolic/deploy.py --terminate <instance_id>")
        
        print_section("Important Reminders")
        print("\n  - Instance is now billing at ${:.2f}/hour".format(estimated_cost_per_hour))
        print("  - Remember to terminate when done!")
        print("  - Checkpoints are saved to 'base_checkpoints/' directory")
        print("  - Download checkpoints before terminating")
        
        return True
        
    except Exception as e:
        print(f"\n  ERROR: Failed to create instance: {e}")
        return False


def terminate_instance(client: HyperbolicClient, instance_id: str):
    """Terminate a running instance"""
    print_section("Terminating Instance")
    
    print(f"\n  Instance ID: {instance_id}")
    response = input("  Are you sure you want to terminate this instance? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n  Termination cancelled.")
        return
    
    try:
        result = client.terminate_instance(instance_id)
        print("\n  ✓ Instance terminated successfully!")
        print("\n  Remember to download any checkpoints or results you need.")
    except Exception as e:
        print(f"\n  ERROR: Failed to terminate instance: {e}")


def list_user_instances(client: HyperbolicClient):
    """List user's active instances"""
    print_section("Your Active Instances")
    
    try:
        instances = client.list_instances()
        
        if not instances:
            print("\n  No active instances.")
            return
        
        for i, instance in enumerate(instances, 1):
            print(f"\n  {i}. Instance ID: {instance.get('id', 'N/A')}")
            print(f"     Status: {instance.get('status', 'N/A')}")
            print(f"     Created: {instance.get('created_at', 'N/A')}")
            
    except Exception as e:
        print(f"\n  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Nanochat training to Hyperbolic Labs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available machines
  python hyperbolic/deploy.py --list-machines
  
  # Deploy medium model (recommended)
  python hyperbolic/deploy.py --model-size medium --auto-launch
  
  # Custom configuration
  python hyperbolic/deploy.py --depth 32 --gpu-count 4 --auto-launch
  
  # Check your instances
  python hyperbolic/deploy.py --list-instances
  
  # Terminate an instance
  python hyperbolic/deploy.py --terminate <instance_id>

Environment Variables:
  HYPERBOLIC_API_KEY  Your Hyperbolic Labs API key (required)
        """
    )
    
    # Model configuration
    parser.add_argument('--model-size', choices=['small', 'medium', 'large', 'xlarge'],
                       help='Predefined model size (overrides custom params)')
    parser.add_argument('--depth', type=int, help='Model depth (number of layers)')
    parser.add_argument('--max-seq-len', type=int, help='Maximum sequence length')
    parser.add_argument('--device-batch-size', type=int, help='Batch size per device')
    
    # Deployment options
    parser.add_argument('--gpu-count', type=int, help='Number of GPUs to request')
    parser.add_argument('--max-price', type=float, help='Maximum price per hour in USD')
    parser.add_argument('--auto-launch', action='store_true', help='Automatically launch without manual selection')
    
    # Actions
    parser.add_argument('--list-machines', action='store_true', help='List available machines')
    parser.add_argument('--list-instances', action='store_true', help='List your active instances')
    parser.add_argument('--terminate', type=str, metavar='INSTANCE_ID', help='Terminate an instance')
    parser.add_argument('--check-balance', action='store_true', help='Check credit balance')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.environ.get('HYPERBOLIC_API_KEY'):
        print("ERROR: HYPERBOLIC_API_KEY environment variable not set!")
        print("\nTo get your API key:")
        print("  1. Go to https://hyperbolic.ai")
        print("  2. Sign up or log in")
        print("  3. Navigate to your API settings")
        print("  4. Copy your API key")
        print("\nThen set it:")
        print("  export HYPERBOLIC_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize client
    try:
        client = HyperbolicClient()
    except Exception as e:
        print(f"ERROR: Failed to initialize Hyperbolic client: {e}")
        sys.exit(1)
    
    # Handle actions
    if args.check_balance:
        try:
            balance = client.get_credit_balance()
            print(f"\nCredit Balance: ${balance.get('balance', 0):.2f}")
        except Exception as e:
            print(f"ERROR: {e}")
        return
    
    if args.list_instances:
        list_user_instances(client)
        return
    
    if args.terminate:
        terminate_instance(client, args.terminate)
        return
    
    # Determine model configuration
    if args.model_size:
        config_map = {
            'small': SMALL_MODEL,
            'medium': MEDIUM_MODEL,
            'large': LARGE_MODEL,
            'xlarge': XLARGE_MODEL
        }
        config = config_map[args.model_size]
        print(f"\nUsing {args.model_size} model configuration:")
    else:
        # Use custom or default configuration
        config = ModelConfig(
            depth=args.depth or 20,
            max_seq_len=args.max_seq_len or 2048,
            device_batch_size=args.device_batch_size or 32
        )
        print("\nUsing custom model configuration:")
    
    print(f"  Depth: {config.depth}")
    print(f"  Model dim: {config.model_dim}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Parameters: ~{config.num_params / 1e6:.1f}M")
    
    # Calculate requirements
    requirements = calculate_gpu_requirements(config)
    print(f"\n{requirements}")
    
    # Handle list machines
    if args.list_machines:
        list_available_machines(client, requirements)
        return
    
    # Create deployment package
    create_deployment_package()
    
    # Launch if requested
    if args.auto_launch:
        success = launch_training(
            client,
            config,
            requirements,
            max_price_per_hour=args.max_price,
            gpu_count=args.gpu_count
        )
        sys.exit(0 if success else 1)
    else:
        print("\n" + "=" * 80)
        print("  Ready to deploy!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review the GPU requirements above")
        print("  2. Check available machines: --list-machines")
        print("  3. Launch deployment: --auto-launch")


if __name__ == "__main__":
    main()
