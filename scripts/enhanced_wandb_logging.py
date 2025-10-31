"""
Enhanced W&B Logging Integration for NanoChat Training

This module integrates enhanced logging into the training pipeline with support for:
- Multi-dataset tracking (TaskMixture)
- Dataset-specific metrics
- Training stage identification
- Enhanced system metrics

Integration Points:
1. base_train.py - Pretraining (FineWeb-EDU)
2. mid_train.py - Midtraining (SmolTalk + MMLU + GSM8K)
3. chat_sft.py - SFT (ARC + GSM8K + SmolTalk)
"""

import wandb
import torch
import time
import psutil
import os
from typing import Dict, Optional, Any, List
from collections import defaultdict


def log_gpu_metrics(device: torch.device, prefix: str = "gpu") -> Dict[str, float]:
    """Log GPU metrics for enhanced visualization."""
    if not torch.cuda.is_available():
        return {}
    
    metrics = {}
    allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
    
    metrics[f"{prefix}/memory_allocated_gb"] = allocated
    metrics[f"{prefix}/memory_reserved_gb"] = reserved
    metrics[f"{prefix}/memory_max_allocated_gb"] = max_allocated
    
    return metrics


def log_system_metrics(prefix: str = "system") -> Dict[str, float]:
    """Log system metrics (CPU, RAM, etc.)."""
    metrics = {}
    
    # CPU metrics
    metrics[f"{prefix}/cpu_percent"] = psutil.cpu_percent(interval=0.1)
    metrics[f"{prefix}/cpu_count"] = psutil.cpu_count()
    
    # Memory metrics
    mem = psutil.virtual_memory()
    metrics[f"{prefix}/memory_total_gb"] = mem.total / (1024**3)
    metrics[f"{prefix}/memory_available_gb"] = mem.available / (1024**3)
    metrics[f"{prefix}/memory_percent"] = mem.percent
    metrics[f"{prefix}/memory_used_gb"] = mem.used / (1024**3)
    
    return metrics


def log_dataset_metrics(
    dataset_name: str,
    dataset_index: Optional[int] = None,
    dataset_size: Optional[int] = None,
    examples_seen: Optional[int] = None,
    prefix: str = "dataset"
) -> Dict[str, Any]:
    """
    Log dataset-specific metrics.
    
    Args:
        dataset_name: Name of the dataset (e.g., "SmolTalk", "MMLU", "GSM8K")
        dataset_index: Index in TaskMixture (if applicable)
        dataset_size: Total size of dataset
        examples_seen: Number of examples seen so far
    """
    metrics = {}
    
    metrics[f"{prefix}/name"] = dataset_name
    if dataset_index is not None:
        metrics[f"{prefix}/index"] = dataset_index
    if dataset_size is not None:
        metrics[f"{prefix}/size"] = dataset_size
        metrics[f"{prefix}/coverage_percent"] = (examples_seen / dataset_size * 100) if examples_seen else 0.0
    if examples_seen is not None:
        metrics[f"{prefix}/examples_seen"] = examples_seen
    
    return metrics


def log_task_mixture_metrics(
    task_names: List[str],
    task_sizes: List[int],
    examples_seen_per_task: Optional[Dict[str, int]] = None,
    prefix: str = "task_mixture"
) -> Dict[str, Any]:
    """
    Log metrics for TaskMixture training.
    
    Args:
        task_names: List of task names in the mixture
        task_sizes: List of sizes for each task
        examples_seen_per_task: Dictionary mapping task names to examples seen
    """
    metrics = {}
    
    total_size = sum(task_sizes)
    metrics[f"{prefix}/num_tasks"] = len(task_names)
    metrics[f"{prefix}/total_examples"] = total_size
    metrics[f"{prefix}/task_names"] = ",".join(task_names)
    
    # Log per-task metrics
    if examples_seen_per_task:
        for task_name, seen in examples_seen_per_task.items():
            task_idx = task_names.index(task_name) if task_name in task_names else -1
            if task_idx >= 0:
                task_size = task_sizes[task_idx]
                metrics[f"{prefix}/{task_name.lower()}/examples_seen"] = seen
                metrics[f"{prefix}/{task_name.lower()}/coverage_percent"] = (seen / task_size * 100) if task_size > 0 else 0.0
    
    # Calculate mixture ratios
    for i, (name, size) in enumerate(zip(task_names, task_sizes)):
        ratio = size / total_size if total_size > 0 else 0.0
        metrics[f"{prefix}/{name.lower()}/ratio"] = ratio
    
    return metrics


def enhance_wandb_logging(
    wandb_run: wandb.sdk.wandb_run.Run,
    step: int,
    device: torch.device,
    dt: float,
    tokens_per_sec: float,
    mfu: float,
    flops_so_far: float,
    total_training_time: float,
    training_stage: str = "pretraining",
    dataset_name: Optional[str] = None,
    **additional_metrics: Dict[str, Any]
) -> None:
    """
    Enhanced logging function that logs all metrics needed for 3D visualizations.
    
    Args:
        wandb_run: W&B run object
        step: Current training step
        device: PyTorch device
        dt: Step time in seconds
        tokens_per_sec: Tokens processed per second
        mfu: Model FLOPs utilization percentage
        flops_so_far: Total FLOPs so far
        total_training_time: Total training time in seconds
        training_stage: Stage of training (pretraining, midtraining, sft, rl)
        dataset_name: Name of current dataset (if applicable)
        **additional_metrics: Any additional metrics to log
    """
    log_dict = {}
    
    # Training stage metadata
    log_dict["training_stage"] = training_stage
    
    # Dataset metadata
    if dataset_name:
        log_dict["dataset/current"] = dataset_name
    
    # GPU metrics
    gpu_metrics = log_gpu_metrics(device)
    log_dict.update(gpu_metrics)
    
    # System metrics
    system_metrics = log_system_metrics()
    log_dict.update(system_metrics)
    
    # Efficiency metrics
    log_dict["efficiency/step"] = step
    log_dict["efficiency/step_time_sec"] = dt
    log_dict["efficiency/tokens_per_sec"] = tokens_per_sec
    log_dict["efficiency/mfu_percent"] = mfu
    log_dict["efficiency/total_flops"] = flops_so_far
    log_dict["efficiency/total_training_time_sec"] = total_training_time
    log_dict["efficiency/flops_per_sec"] = flops_so_far / total_training_time if total_training_time > 0 else 0
    
    # Additional metrics
    log_dict.update(additional_metrics)
    
    # Log to W&B
    wandb_run.log(log_dict)


def log_evaluation_metrics_3d(
    wandb_run: wandb.sdk.wandb_run.Run,
    step: int,
    metrics: Dict[str, float],
    training_stage: str = "pretraining"
) -> None:
    """Log evaluation metrics in a format optimized for 3D visualization."""
    log_dict = {
        "step": step,
        "training_stage": training_stage,
    }
    
    # Log raw metrics
    log_dict.update(metrics)
    
    # Calculate normalized metrics for 3D visualization
    if "mmlu_acc" in metrics:
        log_dict["evaluation/mmlu_normalized"] = metrics["mmlu_acc"] * 100
    if "arc_easy_acc" in metrics:
        log_dict["evaluation/arc_easy_normalized"] = metrics["arc_easy_acc"] * 100
    if "gsm8k_acc" in metrics:
        log_dict["evaluation/gsm8k_normalized"] = metrics["gsm8k_acc"] * 100
    if "humaneval_acc" in metrics:
        log_dict["evaluation/humaneval_normalized"] = metrics["humaneval_acc"] * 100
    
    # Calculate composite metrics
    accuracy_metrics = [
        metrics.get("mmlu_acc", 0),
        metrics.get("arc_easy_acc", 0),
        metrics.get("gsm8k_acc", 0),
        metrics.get("humaneval_acc", 0),
    ]
    if any(accuracy_metrics):
        valid_metrics = [m for m in accuracy_metrics if m > 0]
        log_dict["evaluation/mean_accuracy"] = sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0
    
    wandb_run.log(log_dict)


class DatasetTracker:
    """
    Track dataset usage during training for TaskMixture.
    
    Usage:
        tracker = DatasetTracker(task_names=["SmolTalk", "MMLU", "GSM8K"])
        # In training loop:
        current_dataset = get_current_dataset()  # Your logic
        tracker.log_step(wandb_run, step, current_dataset)
    """
    
    def __init__(self, task_names: List[str], task_sizes: List[int]):
        self.task_names = task_names
        self.task_sizes = task_sizes
        self.examples_seen = defaultdict(int)
        self.step_last_logged = defaultdict(int)
    
    def log_step(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
        step: int,
        current_dataset_name: Optional[str] = None,
        log_every: int = 100
    ) -> None:
        """Log dataset metrics at regular intervals."""
        if step % log_every != 0:
            return
        
        # Log TaskMixture metrics
        mixture_metrics = log_task_mixture_metrics(
            self.task_names,
            self.task_sizes,
            examples_seen_per_task=self.examples_seen if self.examples_seen else None
        )
        
        # Log current dataset
        if current_dataset_name:
            dataset_metrics = log_dataset_metrics(
                dataset_name=current_dataset_name,
                dataset_size=self.task_sizes[self.task_names.index(current_dataset_name)] if current_dataset_name in self.task_names else None
            )
            mixture_metrics.update(dataset_metrics)
        
        wandb_run.log(mixture_metrics)
    
    def increment(self, dataset_name: str, count: int = 1):
        """Increment examples seen for a dataset."""
        self.examples_seen[dataset_name] += count
