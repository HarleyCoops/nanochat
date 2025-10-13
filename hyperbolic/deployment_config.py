"""
Deployment Configuration for Nanochat on Hyperbolic Labs

This module calculates GPU requirements based on model configuration
and provides deployment parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    depth: int = 20
    max_seq_len: int = 2048
    device_batch_size: int = 32
    total_batch_size: int = 524288
    
    @property
    def num_layers(self) -> int:
        return self.depth
    
    @property
    def model_dim(self) -> int:
        return self.depth * 64
    
    @property
    def num_heads(self) -> int:
        return max(1, (self.model_dim + 127) // 128)
    
    @property
    def num_params(self) -> int:
        """Estimate number of parameters"""
        d = self.model_dim
        l = self.num_layers
        vocab = 50304
        
        # Embedding + unembedding
        params = vocab * d + vocab * d
        
        # Each layer has:
        # - Attention: Q, K, V projections + output projection
        # - MLP: 2 linear layers (d -> 4d and 4d -> d)
        attn_params = d * d + d * d + d * d + d * d  # Q, K, V, output
        mlp_params = d * 4 * d + 4 * d * d  # fc + proj
        layer_params = attn_params + mlp_params
        
        params += l * layer_params
        return params


@dataclass
class GPURequirements:
    """GPU requirements for training"""
    min_vram_gb: int
    recommended_vram_gb: int
    min_gpus: int
    recommended_gpus: int
    gpu_type_preference: list[str]
    estimated_memory_per_gpu_gb: float
    
    def __str__(self) -> str:
        return (
            f"GPU Requirements:\n"
            f"  Min VRAM per GPU: {self.min_vram_gb} GB\n"
            f"  Recommended VRAM per GPU: {self.recommended_vram_gb} GB\n"
            f"  Min GPUs: {self.min_gpus}\n"
            f"  Recommended GPUs: {self.recommended_gpus}\n"
            f"  Preferred GPU types: {', '.join(self.gpu_type_preference)}\n"
            f"  Estimated memory per GPU: {self.estimated_memory_per_gpu_gb:.1f} GB"
        )


def calculate_gpu_requirements(config: ModelConfig) -> GPURequirements:
    """
    Calculate GPU requirements based on model configuration
    
    Args:
        config: Model configuration
    
    Returns:
        GPURequirements object with calculated requirements
    """
    num_params = config.num_params
    
    # Memory estimation
    # Model parameters in bfloat16: 2 bytes per parameter
    model_memory_gb = (num_params * 2) / (1024 ** 3)
    
    # Optimizer states (AdamW has 2 states per param, Muon has 1)
    # Most params use Muon (1 state), embeddings use AdamW (2 states)
    # Conservative estimate: assume 2 states for all
    optimizer_memory_gb = (num_params * 2 * 4) / (1024 ** 3)  # fp32 states
    
    # Activations memory (depends on batch size and sequence length)
    # Rough estimate: 4-8x model size for activations during training
    activation_multiplier = 6
    activation_memory_gb = model_memory_gb * activation_multiplier
    
    # Gradient memory (same size as model in bf16)
    gradient_memory_gb = model_memory_gb
    
    # Total memory per GPU (assuming data parallel training)
    total_memory_per_gpu = (
        model_memory_gb +
        optimizer_memory_gb +
        activation_memory_gb +
        gradient_memory_gb
    )
    
    # Add 20% safety margin
    total_memory_per_gpu *= 1.2
    
    # Determine minimum GPU type based on memory requirements
    if total_memory_per_gpu <= 24:
        min_vram = 24
        recommended_vram = 40
        min_gpus = 1
        recommended_gpus = 4
        gpu_preference = ["A100", "L40", "RTX 6000 Ada"]
    elif total_memory_per_gpu <= 40:
        min_vram = 40
        recommended_vram = 80
        min_gpus = 1
        recommended_gpus = 4
        gpu_preference = ["A100", "H100"]
    elif total_memory_per_gpu <= 80:
        min_vram = 80
        recommended_vram = 80
        min_gpus = 1
        recommended_gpus = 8
        gpu_preference = ["H100", "A100"]
    else:
        # Need multiple GPUs
        min_vram = 80
        recommended_vram = 80
        min_gpus = max(2, int((total_memory_per_gpu + 79) // 80))
        recommended_gpus = min_gpus * 2
        gpu_preference = ["H100", "A100"]
    
    return GPURequirements(
        min_vram_gb=min_vram,
        recommended_vram_gb=recommended_vram,
        min_gpus=min_gpus,
        recommended_gpus=recommended_gpus,
        gpu_type_preference=gpu_preference,
        estimated_memory_per_gpu_gb=total_memory_per_gpu
    )


def get_deployment_command(
    config: ModelConfig,
    num_gpus: int,
    checkpoint_dir: Optional[str] = None
) -> str:
    """
    Generate the deployment command for running training on Hyperbolic Labs
    
    Args:
        config: Model configuration
        num_gpus: Number of GPUs to use
        checkpoint_dir: Optional checkpoint directory to resume from
    
    Returns:
        Command string to execute training
    """
    if num_gpus == 1:
        cmd = "python scripts/base_train.py"
    else:
        cmd = f"torchrun --nproc_per_node={num_gpus} scripts/base_train.py"
    
    # Add configuration parameters
    cmd += f" depth={config.depth}"
    cmd += f" max_seq_len={config.max_seq_len}"
    cmd += f" device_batch_size={config.device_batch_size}"
    cmd += f" total_batch_size={config.total_batch_size}"
    
    return cmd


# Default configurations for different model sizes
SMALL_MODEL = ModelConfig(depth=12, max_seq_len=1024, device_batch_size=64)
MEDIUM_MODEL = ModelConfig(depth=20, max_seq_len=2048, device_batch_size=32)
LARGE_MODEL = ModelConfig(depth=32, max_seq_len=2048, device_batch_size=16)
XLARGE_MODEL = ModelConfig(depth=48, max_seq_len=2048, device_batch_size=8)
