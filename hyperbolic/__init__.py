"""
Hyperbolic Labs Deployment Package for Nanochat

This package provides tools to deploy Nanochat training to Hyperbolic Labs GPU marketplace.
"""

from .api_client import HyperbolicClient, GPUInstance
from .deployment_config import (
    ModelConfig,
    GPURequirements,
    calculate_gpu_requirements,
    get_deployment_command,
    SMALL_MODEL,
    MEDIUM_MODEL,
    LARGE_MODEL,
    XLARGE_MODEL
)

__version__ = "0.1.0"
__all__ = [
    "HyperbolicClient",
    "GPUInstance",
    "ModelConfig",
    "GPURequirements",
    "calculate_gpu_requirements",
    "get_deployment_command",
    "SMALL_MODEL",
    "MEDIUM_MODEL",
    "LARGE_MODEL",
    "XLARGE_MODEL",
]
