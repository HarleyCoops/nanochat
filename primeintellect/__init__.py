"""
Prime Intellect Multi-Node Cluster Deployment for Nanochat561

This module provides utilities for deploying nanochat561 training to Prime Intellect
multi-node clusters. Prime Intellect provides clusters with 8xH100 per node.

Reference: https://docs.primeintellect.ai/tutorials-multi-node-cluster/deploy-multi-node
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ClusterConfig:
    """Configuration for Prime Intellect cluster"""
    num_nodes: int = 1  # Each node has 8xH100 GPUs
    total_gpus: int = 8  # Total GPUs needed (8 = 1 node, 16 = 2 nodes, etc.)
    model_depth: int = 20  # Model depth (561M params)
    max_seq_len: int = 2048
    device_batch_size: int = 32
    total_batch_size: int = 524288
    
    @property
    def gpus_per_node(self) -> int:
        """Prime Intellect nodes have 8 GPUs each"""
        return 8
    
    @property
    def nodes_needed(self) -> int:
        """Calculate number of nodes needed"""
        return (self.total_gpus + self.gpus_per_node - 1) // self.gpus_per_node


class PrimeIntellectDeployer:
    """Deployer for Prime Intellect multi-node clusters"""
    
    def __init__(self):
        """Initialize the deployer"""
        self.cli_available = self._check_cli_available()
    
    def _check_cli_available(self) -> bool:
        """Check if Prime Intellect CLI is available"""
        try:
            result = subprocess.run(
                ["prime", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def deploy_cluster(self, config: ClusterConfig) -> Dict:
        """
        Deploy a multi-node cluster on Prime Intellect
        
        Note: Prime Intellect clusters are typically deployed via the web UI.
        This function provides instructions for manual deployment.
        
        Args:
            config: Cluster configuration
            
        Returns:
            Dictionary with deployment instructions
        """
        nodes_needed = config.nodes_needed
        total_gpus = config.total_gpus
        
        instructions = {
            "method": "web_ui",
            "steps": [
                f"1. Go to https://primeintellect.ai",
                f"2. Navigate to 'Multi-Node Cluster' tab",
                f"3. Select configuration: {nodes_needed} node(s) = {total_gpus} H100 GPUs",
                f"4. Click 'Deploy Cluster'",
                f"5. Wait for email confirmation (cluster deployment typically takes 5-10 minutes)",
                f"6. Once deployed, you'll receive {nodes_needed} public IP address(es) - one per node",
            ],
            "nodes_needed": nodes_needed,
            "total_gpus": total_gpus,
            "gpus_per_node": config.gpus_per_node,
            "estimated_cost_per_hour": nodes_needed * 8 * 2.50,  # Rough estimate: $2.50/GPU/hour
            "estimated_total_cost": nodes_needed * 8 * 2.50 * 10,  # 10 hours
        }
        
        return instructions
    
    def generate_setup_script(self, config: ClusterConfig, output_path: Optional[str] = None) -> str:
        """
        Generate setup script for Prime Intellect cluster nodes
        
        Args:
            config: Cluster configuration
            output_path: Optional path to save script
            
        Returns:
            Path to generated script
        """
        nodes_needed = config.nodes_needed
        
        script_content = f"""#!/bin/bash
# Nanochat561 Training Setup Script for Prime Intellect Multi-Node Cluster
# This script should be run on each node (or via SSH on master node)

set -e

echo "=========================================="
echo "Nanochat561 Training Setup for Prime Intellect"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Nodes: {nodes_needed}"
echo "  GPUs per node: {config.gpus_per_node}"
echo "  Total GPUs: {config.total_gpus}"
echo "  Model depth: {config.model_depth}"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git python3-pip openssh-client openssh-server

# Clone repository (if not already present)
if [ ! -d "nanochat561" ]; then
    echo "Cloning repository..."
    git clone https://github.com/HarleyCoops/nanochat561.git
    cd nanochat561
else
    echo "Repository exists, updating..."
    cd nanochat561
    git pull
fi

# Install Python dependencies using uv (faster than pip)
echo "Installing Python dependencies..."
pip install uv
uv pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p base_data
mkdir -p tokenizer
mkdir -p base_checkpoints

# Verify GPU availability
echo ""
echo "Verifying GPU setup..."
nvidia-smi

# Display training command
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Multi-node training command:"
if [ {nodes_needed} -eq 1 ]; then
    echo "  torchrun --nproc_per_node={config.total_gpus} scripts/base_train.py \\"
else
    echo "  # On master node, run:"
    echo "  torchrun --nnodes={nodes_needed} --nproc_per_node={config.gpus_per_node} \\"
    echo "    --node_rank=0 --master_addr=\\$MASTER_ADDR --master_port=29500 \\"
    echo "    scripts/base_train.py depth={config.model_depth} \\"
    echo "    max_seq_len={config.max_seq_len} device_batch_size={config.device_batch_size} \\"
    echo "    total_batch_size={config.total_batch_size}"
    echo ""
    echo "  # On worker nodes, run (with appropriate node_rank):"
    echo "  torchrun --nnodes={nodes_needed} --nproc_per_node={config.gpus_per_node} \\"
    echo "    --node_rank=<NODE_RANK> --master_addr=\\$MASTER_ADDR --master_port=29500 \\"
    echo "    scripts/base_train.py depth={config.model_depth} \\"
    echo "    max_seq_len={config.max_seq_len} device_batch_size={config.device_batch_size} \\"
    echo "    total_batch_size={config.total_batch_size}"
fi
echo ""
"""
        
        if output_path:
            script_path = Path(output_path)
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            return str(script_path)
        else:
            # Return to primeintellect directory
            script_dir = Path("primeintellect")
            script_dir.mkdir(exist_ok=True)
            script_path = script_dir / "setup_cluster.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            return str(script_path)
    
    def generate_training_script(self, config: ClusterConfig, master_addr: str, node_rank: int = 0) -> str:
        """
        Generate training script for a specific node
        
        Args:
            config: Cluster configuration
            master_addr: Master node IP address
            node_rank: Rank of this node (0 for master)
            
        Returns:
            Training command string
        """
        nodes_needed = config.nodes_needed
        
        if nodes_needed == 1:
            # Single node training
            cmd = f"torchrun --nproc_per_node={config.total_gpus} scripts/base_train.py"
        else:
            # Multi-node training
            cmd = (
                f"torchrun --nnodes={nodes_needed} --nproc_per_node={config.gpus_per_node} "
                f"--node_rank={node_rank} --master_addr={master_addr} --master_port=29500 "
                f"scripts/base_train.py"
            )
        
        # Add model configuration
        cmd += (
            f" depth={config.model_depth}"
            f" max_seq_len={config.max_seq_len}"
            f" device_batch_size={config.device_batch_size}"
            f" total_batch_size={config.total_batch_size}"
        )
        
        return cmd


def get_561m_model_config() -> ClusterConfig:
    """
    Get configuration for 561M parameter model (depth=20)
    This matches the original training configuration.
    
    Returns:
        ClusterConfig for 561M model
    """
    return ClusterConfig(
        num_nodes=1,  # 1 node = 8 GPUs
        total_gpus=8,  # 8 H100s for ~10 hours
        model_depth=20,
        max_seq_len=2048,
        device_batch_size=32,
        total_batch_size=524288,
    )


if __name__ == "__main__":
    # Example usage
    deployer = PrimeIntellectDeployer()
    config = get_561m_model_config()
    
    print("Prime Intellect Deployment Configuration")
    print("=" * 60)
    print(f"Nodes needed: {config.nodes_needed}")
    print(f"GPUs per node: {config.gpus_per_node}")
    print(f"Total GPUs: {config.total_gpus}")
    print(f"Model: 561M parameters (depth={config.model_depth})")
    print(f"Estimated cost: ${config.nodes_needed * 8 * 2.50 * 10:.2f} for 10 hours")
    print()
    
    # Generate setup script
    script_path = deployer.generate_setup_script(config)
    print(f"Setup script generated: {script_path}")
    
    # Get deployment instructions
    instructions = deployer.deploy_cluster(config)
    print("\nDeployment Instructions:")
    for step in instructions["steps"]:
        print(f"  {step}")

