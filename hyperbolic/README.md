# Hyperbolic Labs Deployment for Nanochat

This package enables you to deploy Nanochat model training to Hyperbolic Labs GPU marketplace, allowing you to train models that are too large for your local machine.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Model Configurations](#model-configurations)
- [GPU Requirements](#gpu-requirements)
- [Cost Estimation](#cost-estimation)
- [Troubleshooting](#troubleshooting)

## Overview

The Hyperbolic Labs deployment system provides:

- **Automatic GPU Selection**: Intelligently selects the best GPU type and count based on your model configuration
- **Cost Optimization**: Finds the most cost-effective machines that meet your requirements
- **Easy Deployment**: Simple CLI interface to launch training instances
- **Instance Management**: Monitor and terminate instances easily

### Key Features

- 🚀 One-command deployment to cloud GPUs
- 💰 Automatic cost estimation and budgeting
- 🎯 Smart GPU selection based on model size
- 📊 Real-time instance monitoring
- ⚡ Support for H100, A100, and other high-performance GPUs

## Prerequisites

### 1. Hyperbolic Labs Account

You need a Hyperbolic Labs account with API access:

1. Go to [https://hyperbolic.ai](https://hyperbolic.ai)
2. Sign up or log in to your account
3. Navigate to your API settings/dashboard
4. Generate and copy your API key
5. Add credits to your account for GPU usage

### 2. Environment Setup

```bash
# Set your Hyperbolic API key
export HYPERBOLIC_API_KEY='your-api-key-here'

# For Windows PowerShell:
$env:HYPERBOLIC_API_KEY='your-api-key-here'

# For Windows CMD:
set HYPERBOLIC_API_KEY=your-api-key-here
```

### 3. Install Dependencies

```bash
# Install required packages
pip install requests

# Or add to your requirements.txt:
# requests>=2.31.0
```

## Installation

The deployment package is already part of your nanochat repository. No additional installation required!

## Quick Start

### 1. Check Your Balance

```bash
python hyperbolic/deploy.py --check-balance
```

### 2. View Available Machines

```bash
# See what GPUs are available that meet your requirements
python hyperbolic/deploy.py --model-size medium --list-machines
```

### 3. Deploy Training

```bash
# Deploy with recommended settings (Medium model, 4 GPUs)
python hyperbolic/deploy.py --model-size medium --auto-launch

# Deploy with custom configuration
python hyperbolic/deploy.py --depth 32 --gpu-count 4 --auto-launch
```

### 4. Monitor Your Instances

```bash
# List your active instances
python hyperbolic/deploy.py --list-instances
```

### 5. Terminate When Done

```bash
# Terminate an instance (replace with your instance ID)
python hyperbolic/deploy.py --terminate <instance-id>
```

## Usage Guide

### Command-Line Options

```bash
python hyperbolic/deploy.py [OPTIONS]
```

#### Model Configuration

- `--model-size {small,medium,large,xlarge}`: Use predefined model size
- `--depth N`: Custom model depth (number of layers)
- `--max-seq-len N`: Maximum sequence length (default: 2048)
- `--device-batch-size N`: Batch size per device (default: 32)

#### Deployment Options

- `--gpu-count N`: Number of GPUs to request
- `--max-price FLOAT`: Maximum price per hour in USD
- `--auto-launch`: Automatically launch without manual confirmation

#### Actions

- `--list-machines`: List available machines
- `--list-instances`: List your active instances
- `--terminate ID`: Terminate a specific instance
- `--check-balance`: Check your credit balance

### Example Workflows

#### Deploy Small Model for Testing

```bash
# Small model on 1-2 GPUs (~$3-6/hour)
python hyperbolic/deploy.py --model-size small --gpu-count 2 --auto-launch
```

#### Deploy Medium Model (Recommended)

```bash
# Medium model on 4 GPUs (~$12-20/hour)
python hyperbolic/deploy.py --model-size medium --auto-launch
```

#### Deploy Large Model

```bash
# Large model on 8 GPUs (~$24-40/hour)
python hyperbolic/deploy.py --model-size large --gpu-count 8 --auto-launch
```

#### Budget-Constrained Deployment

```bash
# Find machines under $5/hour
python hyperbolic/deploy.py --model-size medium --max-price 5.0 --list-machines

# Deploy with budget constraint
python hyperbolic/deploy.py --model-size medium --max-price 5.0 --gpu-count 2 --auto-launch
```

## Model Configurations

### Predefined Configurations

| Size | Depth | Params | Min VRAM/GPU | Recommended GPUs | Est. Cost/Hour* |
|------|-------|--------|--------------|------------------|-----------------|
| Small | 12 | ~60M | 24 GB | 2 | $3-6 |
| Medium | 20 | ~160M | 40 GB | 4 | $12-20 |
| Large | 32 | ~410M | 80 GB | 4-8 | $20-40 |
| XLarge | 48 | ~920M | 80 GB | 8 | $40-80 |

*Based on H100 pricing (~$1.50-3.20/hr per GPU on Hyperbolic Labs)

### Custom Configuration

You can specify custom model parameters:

```bash
python hyperbolic/deploy.py \
  --depth 24 \
  --max-seq-len 4096 \
  --device-batch-size 16 \
  --gpu-count 4 \
  --auto-launch
```

## GPU Requirements

### Automatic Calculation

The system automatically calculates GPU requirements based on:

- **Model Parameters**: Number of parameters in your model
- **Optimizer States**: Memory for AdamW and Muon optimizers
- **Activations**: Gradient checkpointing and activation memory
- **Batch Size**: Per-device and total batch size

### Memory Breakdown

For a typical training run:

- **Model Weights** (bf16): ~2 bytes per parameter
- **Optimizer States** (fp32): ~8 bytes per parameter
- **Gradients** (bf16): ~2 bytes per parameter
- **Activations**: ~6x model size (varies with batch size)
- **Safety Margin**: +20% buffer

### GPU Type Recommendations

| Model Size | Recommended GPU | Alternatives |
|------------|----------------|--------------|
| < 100M params | A100 40GB | L40, RTX 6000 Ada |
| 100-300M params | A100 80GB | H100 |
| 300-1B params | H100 80GB | 2x A100 80GB |
| 1B+ params | Multi-GPU H100 | Multi-GPU A100 |

## Cost Estimation

### Hyperbolic Labs Pricing (Approximate)

- **H100 SXM**: $3.20/hour per GPU
- **H100 PCIe**: $2.50/hour per GPU
- **A100 80GB**: $1.50/hour per GPU
- **A100 40GB**: $1.20/hour per GPU

### Training Duration Estimates

For Chinchilla-optimal training (20 tokens per parameter):

| Model Size | Training Tokens | H100 8-GPU Time | Estimated Cost |
|------------|----------------|-----------------|----------------|
| 60M | 1.2B | ~1 hour | $25 |
| 160M | 3.2B | ~2 hours | $50 |
| 410M | 8.2B | ~5 hours | $125 |
| 920M | 18.4B | ~10 hours | $250 |

*These are rough estimates. Actual times depend on data loading, evaluation frequency, etc.*

### Cost Optimization Tips

1. **Start Small**: Test with small model first to verify setup
2. **Use Spot Instances**: If available, spot instances are cheaper
3. **Batch Evaluations**: Reduce eval frequency to save time
4. **Monitor Actively**: Check training curves early, terminate if issues
5. **Download Checkpoints**: Save checkpoints periodically, don't lose work

## Troubleshooting

### Common Issues

#### 1. API Key Not Set

```
ERROR: HYPERBOLIC_API_KEY environment variable not set!
```

**Solution**: Set your API key:
```bash
export HYPERBOLIC_API_KEY='your-key-here'
```

#### 2. No Suitable Machines Available

```
ERROR: No suitable machine found!
```

**Solutions**:
- Reduce GPU count: `--gpu-count 2`
- Increase max price: `--max-price 10.0`
- Check available machines: `--list-machines`
- Try different time of day (peak vs off-peak)

#### 3. Instance Creation Failed

**Possible causes**:
- Insufficient credits
- Machine became unavailable
- API rate limiting

**Solutions**:
- Check balance: `--check-balance`
- Try different machine type
- Wait a few minutes and retry

#### 4. Out of Memory During Training

**Solutions**:
- Reduce `device_batch_size`: `--device-batch-size 16`
- Use more GPUs to distribute memory
- Reduce sequence length if possible

### Getting Help

1. Check Hyperbolic Labs documentation: [https://docs.hyperbolic.xyz](https://docs.hyperbolic.xyz)
2. Review the nanochat repository issues
3. Check your instance logs on Hyperbolic dashboard

## Advanced Usage

### Using the API Programmatically

```python
from hyperbolic import (
    HyperbolicClient,
    ModelConfig,
    calculate_gpu_requirements,
    MEDIUM_MODEL
)

# Initialize client
client = HyperbolicClient()

# Calculate requirements for your model
config = MEDIUM_MODEL
requirements = calculate_gpu_requirements(config)
print(requirements)

# Find best machine
machine = client.find_best_machine(
    min_gpus=4,
    gpu_type_preference=["H100", "A100"],
    max_price_per_hour=20.0
)

if machine:
    print(f"Found: {machine.gpu_type} at ${machine.pricing:.2f}/hour")
    
    # Create instance
    result = client.create_instance(
        cluster_name=machine.cluster_name,
        node_name=machine.node_name,
        gpu_count=4
    )
    print(f"Instance ID: {result['instance_id']}")
```

### SSH Access

Once your instance is created:

1. Go to Hyperbolic Labs dashboard
2. Find your instance
3. Copy SSH command
4. SSH into the instance
5. Run the setup script from `hyperbolic_deployment/setup.sh`
6. Start training

## Best Practices

1. **Test Locally First**: Ensure your training script works locally before deploying
2. **Use Version Control**: Push code changes before deploying
3. **Monitor Actively**: Check training progress in first 30 minutes
4. **Save Checkpoints**: Configure frequent checkpointing
5. **Set Up WandB**: Use Weights & Biases for remote monitoring
6. **Download Results**: Always download checkpoints before terminating
7. **Terminate Promptly**: Don't leave instances running when not training

## Security Notes

- Never commit your API key to version control
- Use environment variables for sensitive data
- Regularly rotate your API keys
- Monitor your usage and billing

## Support

For issues specific to:
- **Hyperbolic Labs API**: Contact Hyperbolic Labs support
- **Nanochat Training**: Check nanochat repository issues
- **Deployment Scripts**: Open an issue in this repository

---

**Happy Training! 🚀**
