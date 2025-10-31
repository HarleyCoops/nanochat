# Prime Intellect Multi-Node Cluster Deployment for Nanochat561

This package enables you to deploy Nanochat561 model training to Prime Intellect multi-node clusters with H100 GPUs.

## Overview

Prime Intellect provides multi-node clusters with:
- **8xH100 GPUs per node**
- **Scalable**: Deploy 16-64+ H100 GPUs (2-8 nodes)
- **High Performance**: Optimized for distributed training workloads
- **SSH Access**: Direct SSH access to each node

Reference: [Prime Intellect Multi-Node Documentation](https://docs.primeintellect.ai/tutorials-multi-node-cluster/deploy-multi-node)

## Quick Start for Nanochat561 (561M Model)

### Option 1: API Deployment (Recommended)

1. **Set API Key**:
   ```bash
   export PRIME_INTELLECT_API_KEY='your-api-key-here'
   ```

2. **Deploy via API**:
   ```bash
   python primeintellect/deploy.py --config 561m
   ```

   This will:
   - Check cluster availability
   - Create pod automatically
   - Wait for pod to be ready
   - Provide SSH connection details

### Option 2: Manual Deployment via Web UI

1. Go to [https://primeintellect.ai](https://primeintellect.ai)
2. Navigate to **Multi-Node Cluster** tab
3. Select **1 node** (8xH100 GPUs) for 561M model
4. Click **Deploy Cluster**
5. Wait for email confirmation (typically 5-10 minutes)
6. You'll receive **1 public IP address** (one per node)

Then run:
```bash
python primeintellect/deploy.py --config 561m --no-api
```

This will:
- Show deployment instructions
- Generate `primeintellect/setup_cluster.sh` script

### Step 3: SSH into Node and Setup

```bash
# SSH into your node (use IP from API or email)
ssh root@<YOUR_NODE_IP>

# Upload and run setup script
scp primeintellect/setup_cluster.sh root@<YOUR_NODE_IP>:/root/
ssh root@<YOUR_NODE_IP> 'bash setup_cluster.sh'
```

### Step 4: Start Nanochat561 Training

```bash
# Single node (8 GPUs)
torchrun --nproc_per_node=8 scripts/base_train.py \
  depth=20 max_seq_len=2048 device_batch_size=32 total_batch_size=524288
```

## Configuration

### Nanochat561 Model (Default)

- **Model**: nanochat561
- **Model depth**: 20 layers
- **Parameters**: ~561M
- **GPUs needed**: 8 H100s (1 node)
- **Training time**: ~10 hours
- **Estimated cost**: ~$200 for 10 hours

### Multi-Node Configuration

For larger models or faster training:

```bash
# 2 nodes (16 GPUs)
python primeintellect/deploy.py --nodes 2

# Custom configuration
python primeintellect/deploy.py --nodes 2 --depth 32 --gpus 16
```

## Usage Guide

### Command-Line Options

```bash
python primeintellect/deploy.py [OPTIONS]
```

#### Predefined Configurations

- `--config 561m`: Deploy 561M model (1 node, 8 GPUs)

#### Custom Configuration

- `--nodes N`: Number of nodes (each node = 8 GPUs)
- `--gpus N`: Total number of GPUs needed
- `--depth N`: Model depth (default: 20)
- `--max-seq-len N`: Maximum sequence length (default: 2048)
- `--device-batch-size N`: Batch size per device (default: 32)
- `--setup-only`: Only generate setup script

### Example Workflows

#### Deploy Nanochat561 Model

```bash
# Deploy via API (requires PRIME_INTELLECT_API_KEY)
python primeintellect/deploy.py --config 561m

# Manual deployment (web UI)
python primeintellect/deploy.py --config 561m --no-api

# Just generate setup script
python primeintellect/deploy.py --config 561m --setup-only

# List active pods
python primeintellect/deploy.py --list-pods

# Check pod status
python primeintellect/deploy.py --status <POD_ID>

# Terminate pod
python primeintellect/deploy.py --terminate <POD_ID>
```

#### Custom Multi-Node Setup

```bash
# 2 nodes (16 GPUs) for faster training
python primeintellect/deploy.py --nodes 2 --depth 20
```

## Multi-Node Training

For multi-node training (2+ nodes), you'll need to:

1. **Set master address** on all nodes:
   ```bash
   export MASTER_ADDR=<MASTER_NODE_IP>
   export MASTER_PORT=29500
   ```

2. **Start training on master node** (node_rank=0):
   ```bash
   torchrun --nnodes=2 --nproc_per_node=8 \
     --node_rank=0 --master_addr=$MASTER_ADDR --master_port=29500 \
     scripts/base_train.py depth=20 ...
   ```

3. **Start training on worker nodes** (node_rank=1, 2, ...):
   ```bash
   torchrun --nnodes=2 --nproc_per_node=8 \
     --node_rank=1 --master_addr=$MASTER_ADDR --master_port=29500 \
     scripts/base_train.py depth=20 ...
   ```

The deploy script will generate these commands automatically.

## Cost Estimation

Prime Intellect pricing (approximate):
- **H100 GPU**: ~$2.50/hour per GPU
- **1 node (8 GPUs)**: ~$20/hour
- **10 hours training**: ~$200

### 561M Model Training Cost

- **GPUs**: 8 H100s
- **Duration**: ~10 hours
- **Total cost**: ~$200

## Troubleshooting

### Cluster Not Deploying

- Check your Prime Intellect account balance
- Verify you're selecting the correct configuration
- Wait a few minutes - deployment can take 5-10 minutes

### SSH Connection Issues

- Verify the IP address from email
- Check firewall/security group settings
- Ensure your local network allows outbound SSH

### Multi-Node Communication Issues

- Ensure all nodes can reach each other
- Verify `MASTER_ADDR` and `MASTER_PORT` are set correctly
- Check firewall rules allow communication on port 29500

### Training Fails to Start

- Verify all nodes have completed setup
- Check that all nodes have the same code version
- Ensure data is accessible (shared storage or copied to each node)
- Check GPU availability: `nvidia-smi`

## Best Practices

1. **Test Locally First**: Ensure training script works locally before deploying
2. **Use Version Control**: Push code changes before deploying
3. **Monitor Actively**: Check training progress in first 30 minutes
4. **Save Checkpoints**: Configure frequent checkpointing
5. **Set Up WandB**: Use Weights & Biases for remote monitoring
6. **Download Results**: Always download checkpoints before terminating cluster
7. **Terminate Promptly**: Don't leave clusters running when not training

## Comparison with Other Providers

| Feature | Prime Intellect | Hyperbolic/Lambda Labs |
|---------|----------------|------------------------|
| GPUs per node | 8xH100 | Variable (1-8) |
| Multi-node | Yes (2-8 nodes) | Yes |
| Deployment | Web UI | API/CLI |
| SSH Access | Yes | Yes |
| Best for | Large-scale training | Flexible configurations |

## Reference

- [Prime Intellect Multi-Node Docs](https://docs.primeintellect.ai/tutorials-multi-node-cluster/deploy-multi-node)
- [Prime Intellect Platform](https://primeintellect.ai)
- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

---

**Note**: Prime Intellect deployment is currently done via web UI. This package provides helper scripts and instructions for setting up training after cluster deployment.

