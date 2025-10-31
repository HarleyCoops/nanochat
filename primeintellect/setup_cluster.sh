#!/bin/bash
# Nanochat Training Setup Script for Prime Intellect Multi-Node Cluster
# This script should be run on each node (or via SSH on master node)

set -e

echo "=========================================="
echo "Nanochat Training Setup for Prime Intellect"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Nodes: 1"
echo "  GPUs per node: 8"
echo "  Total GPUs: 8"
echo "  Model depth: 20"
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
if [ 1 -eq 1 ]; then
    echo "  torchrun --nproc_per_node=8 scripts/base_train.py \"
else
    echo "  # On master node, run:"
    echo "  torchrun --nnodes=1 --nproc_per_node=8 \"
    echo "    --node_rank=0 --master_addr=\$MASTER_ADDR --master_port=29500 \"
    echo "    scripts/base_train.py depth=20 \"
    echo "    max_seq_len=2048 device_batch_size=32 \"
    echo "    total_batch_size=524288"
    echo ""
    echo "  # On worker nodes, run (with appropriate node_rank):"
    echo "  torchrun --nnodes=1 --nproc_per_node=8 \"
    echo "    --node_rank=<NODE_RANK> --master_addr=\$MASTER_ADDR --master_port=29500 \"
    echo "    scripts/base_train.py depth=20 \"
    echo "    max_seq_len=2048 device_batch_size=32 \"
    echo "    total_batch_size=524288"
fi
echo ""
