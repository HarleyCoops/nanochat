"""
Prime Intellect CLI tool for deploying nanochat561 training

Usage:
    python primeintellect/deploy.py --config 561m
    python primeintellect/deploy.py --nodes 1 --setup-only
    python primeintellect/deploy.py --help
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primeintellect import (
    PrimeIntellectDeployer,
    ClusterConfig,
    get_561m_model_config
)
from primeintellect.api_client import PrimeIntellectClient


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def deploy_561m_model(use_api: bool = True):
    """Deploy 561M parameter nanochat561 model training"""
    print_section("Deploying Nanochat561 Model Training on Prime Intellect")
    
    deployer = PrimeIntellectDeployer()
    config = get_561m_model_config()
    
    print(f"\nConfiguration:")
    print(f"  Model: nanochat561 (561M parameters, depth={config.model_depth})")
    print(f"  Nodes: {config.nodes_needed} (8xH100 per node)")
    print(f"  Total GPUs: {config.total_gpus}")
    print(f"  Estimated training time: ~10 hours")
    print(f"  Estimated cost: ${config.nodes_needed * 8 * 2.50 * 10:.2f} for 10 hours")
    
    if use_api:
        # Use API for deployment
        try:
            client = PrimeIntellectClient()
            
            print_section("Checking Cluster Availability")
            print("\nSearching for available H100 clusters...")
            
            cluster = client.find_best_cluster(
                gpu_type="H100_80GB",
                gpu_count=config.total_gpus
            )
            
            if not cluster:
                print("\n⚠️  No available clusters found. Falling back to manual deployment.")
                use_api = False
            else:
                print(f"\n✓ Found available cluster:")
                print(f"  Provider: {cluster.provider}")
                print(f"  Region: {cluster.region}, {cluster.country}")
                print(f"  GPUs: {cluster.gpu_count}x{cluster.gpu_type}")
                print(f"  Price: ${cluster.price_per_hour:.2f}/hour")
                print(f"  Estimated cost: ${cluster.price_per_hour * config.total_gpus * 10:.2f} for 10 hours")
                
                print_section("Creating Pod via API")
                pod_name = f"nanochat561-{int(time.time())}"
                
                print(f"\nCreating pod: {pod_name}")
                pod = client.create_pod(
                    name=pod_name,
                    cloud_id=cluster.cloud_id,
                    gpu_type="H100_80GB",
                    gpu_count=config.total_gpus,
                    disk_size=200,
                    image="ubuntu_22_cuda_12"
                )
                
                print(f"\n✓ Pod created successfully!")
                print(f"  Pod ID: {pod.id}")
                print(f"  Status: {pod.status}")
                print(f"  Price: ${pod.price_per_hour:.2f}/hour")
                
                print_section("Waiting for Pod to be Ready")
                print("\nWaiting for pod provisioning (this may take 5-10 minutes)...")
                print("You can check status with: python primeintellect/deploy.py --list-pods")
                
                try:
                    pod = client.wait_for_pod_ready(pod.id, timeout=600)
                    print(f"\n✓ Pod is ready!")
                    print(f"  IP Address: {pod.ip}")
                    print(f"  SSH: {pod.ssh_connection}")
                    
                    # Generate setup script
                    script_path = deployer.generate_setup_script(config)
                    print(f"\n✓ Setup script generated: {script_path}")
                    
                    print_section("Next Steps")
                    print(f"\n1. SSH into your pod:")
                    print(f"   {pod.ssh_connection}")
                    
                    print(f"\n2. Upload and run setup script:")
                    print(f"   scp {script_path} root@{pod.ip}:/root/")
                    print(f"   ssh root@{pod.ip} 'bash setup_cluster.sh'")
                    
                    print(f"\n3. Start nanochat561 training:")
                    training_cmd = deployer.generate_training_script(config, "localhost", node_rank=0)
                    print(f"   {training_cmd}")
                    
                    print(f"\n4. Monitor training:")
                    print(f"   - Check logs in terminal")
                    print(f"   - View metrics in WandB (if configured)")
                    print(f"   - Checkpoints saved to base_checkpoints/")
                    
                    print(f"\n5. When finished, terminate pod:")
                    print(f"   python primeintellect/deploy.py --terminate {pod.id}")
                    
                    return
                    
                except TimeoutError:
                    print(f"\n⚠️  Pod provisioning is taking longer than expected.")
                    print(f"   Pod ID: {pod.id}")
                    print(f"   Check status: python primeintellect/deploy.py --status {pod.id}")
                    print(f"   Or visit: https://primeintellect.ai")
                    return
                except Exception as e:
                    print(f"\n⚠️  Error waiting for pod: {e}")
                    print(f"   Pod ID: {pod.id}")
                    print(f"   Check status manually: python primeintellect/deploy.py --status {pod.id}")
                    return
                    
        except ValueError as e:
            print(f"\n⚠️  API key not configured: {e}")
            print("   Falling back to manual deployment instructions.")
            use_api = False
        except Exception as e:
            print(f"\n⚠️  API error: {e}")
            print("   Falling back to manual deployment instructions.")
            use_api = False
    
    if not use_api:
        # Fallback to manual deployment instructions
        instructions = deployer.deploy_cluster(config)
        
        print_section("Manual Deployment Steps")
        for step in instructions["steps"]:
            print(f"  {step}")
        
        # Generate setup script
        script_path = deployer.generate_setup_script(config)
        print(f"\n✓ Setup script generated: {script_path}")
        
        print_section("Next Steps After Cluster Deployment")
        print("\n1. Wait for email confirmation with public IP addresses")
        print(f"\n2. SSH into master node (first IP address)")
        print("   ssh root@<MASTER_IP>")
        
        print("\n3. Run setup script on master node:")
        print(f"   bash {script_path}")
        
        if config.nodes_needed > 1:
            print(f"\n4. Run setup script on worker nodes:")
            print(f"   ssh root@<WORKER_IP_1> 'bash -s' < {script_path}")
            print(f"   ssh root@<WORKER_IP_2> 'bash -s' < {script_path}")
            print("   (repeat for each worker node)")
            
            print("\n5. Set environment variables on master node:")
            print("   export MASTER_ADDR=<MASTER_IP>")
            print("   export MASTER_PORT=29500")
            
            print("\n6. Start nanochat561 training on master node:")
            master_cmd = deployer.generate_training_script(config, "<MASTER_IP>", node_rank=0)
            print(f"   {master_cmd}")
            
            print("\n7. Start training on each worker node:")
            for i in range(1, config.nodes_needed):
                worker_cmd = deployer.generate_training_script(config, "<MASTER_IP>", node_rank=i)
                print(f"   Worker {i}: {worker_cmd}")
        else:
            print("\n4. Start nanochat561 training (single node):")
            training_cmd = deployer.generate_training_script(config, "localhost", node_rank=0)
            print(f"   {training_cmd}")
        
        print("\n5. Monitor training progress:")
        print("   - Check logs in terminal")
        print("   - View metrics in WandB (if configured)")
        print("   - Checkpoints saved to base_checkpoints/")
        
        print("\n6. When finished, terminate cluster in Prime Intellect dashboard")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Nanochat561 training to Prime Intellect multi-node clusters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy 561M model (1 node, 8 GPUs)
  python primeintellect/deploy.py --config 561m
  
  # Generate setup script only
  python primeintellect/deploy.py --nodes 1 --setup-only
  
  # Custom configuration
  python primeintellect/deploy.py --nodes 2 --depth 20 --gpus 16

Notes:
  - Prime Intellect clusters are deployed via web UI at https://primeintellect.ai
  - Each node has 8xH100 GPUs
  - You'll receive one public IP per node after deployment
  - SSH access is provided for each node
        """
    )
    
    parser.add_argument(
        '--config',
        choices=['561m'],
        help='Predefined model configuration'
    )
    parser.add_argument(
        '--nodes',
        type=int,
        help='Number of nodes (each node has 8 GPUs)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        help='Total number of GPUs needed'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=20,
        help='Model depth (default: 20 for 561M model)'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    parser.add_argument(
        '--device-batch-size',
        type=int,
        default=32,
        help='Batch size per device (default: 32)'
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only generate setup script, do not show deployment instructions'
    )
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='Use manual deployment instead of API'
    )
    parser.add_argument(
        '--list-pods',
        action='store_true',
        help='List all active pods'
    )
    parser.add_argument(
        '--status',
        type=str,
        metavar='POD_ID',
        help='Get status of a specific pod'
    )
    parser.add_argument(
        '--terminate',
        type=str,
        metavar='POD_ID',
        help='Terminate a specific pod'
    )
    
    args = parser.parse_args()
    
    # Handle pod management commands
    if args.list_pods:
        try:
            client = PrimeIntellectClient()
            pods = client.get_pods()
            
            print_section("Active Pods")
            if not pods:
                print("\nNo active pods found.")
            else:
                for pod in pods:
                    print(f"\n  Pod: {pod.name}")
                    print(f"    ID: {pod.id}")
                    print(f"    Status: {pod.status}")
                    print(f"    GPUs: {pod.gpu_count}x{pod.gpu_name}")
                    print(f"    Price: ${pod.price_per_hour:.2f}/hour")
                    if pod.ip:
                        print(f"    IP: {pod.ip}")
                    if pod.ssh_connection:
                        print(f"    SSH: {pod.ssh_connection}")
        except ValueError as e:
            print(f"ERROR: {e}")
            print("\nSet PRIME_INTELLECT_API_KEY environment variable to use API features.")
        except Exception as e:
            print(f"ERROR: {e}")
        return
    
    if args.status:
        try:
            client = PrimeIntellectClient()
            pod = client.get_pod(args.status)
            
            print_section(f"Pod Status: {pod.name}")
            print(f"\n  ID: {pod.id}")
            print(f"  Status: {pod.status}")
            print(f"  GPUs: {pod.gpu_count}x{pod.gpu_name}")
            print(f"  Price: ${pod.price_per_hour:.2f}/hour")
            if pod.ip:
                print(f"  IP: {pod.ip}")
            if pod.ssh_connection:
                print(f"  SSH: {pod.ssh_connection}")
        except ValueError as e:
            print(f"ERROR: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
        return
    
    if args.terminate:
        try:
            client = PrimeIntellectClient()
            print(f"\nTerminating pod: {args.terminate}")
            response = input("Are you sure? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                result = client.delete_pod(args.terminate)
                print(f"\n✓ Pod terminated successfully")
            else:
                print("Termination cancelled.")
        except ValueError as e:
            print(f"ERROR: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
        return
    
    # Determine configuration
    if args.config == '561m':
        deploy_561m_model(use_api=not args.no_api)
    elif args.setup_only:
        # Just generate setup script
        deployer = PrimeIntellectDeployer()
        if args.nodes or args.gpus:
            if args.nodes:
                total_gpus = args.nodes * 8
            else:
                total_gpus = args.gpus
                args.nodes = (total_gpus + 7) // 8
            
            config = ClusterConfig(
                num_nodes=args.nodes,
                total_gpus=total_gpus,
                model_depth=args.depth,
                max_seq_len=args.max_seq_len,
                device_batch_size=args.device_batch_size,
            )
        else:
            config = get_561m_model_config()
        
        script_path = deployer.generate_setup_script(config)
        print(f"✓ Setup script generated: {script_path}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

