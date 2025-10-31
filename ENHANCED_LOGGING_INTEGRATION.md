# Enhanced W&B Logging Integration Guide

This document shows how to integrate enhanced logging into the existing training pipeline, with special attention to **multi-dataset training**.

## Overview

Your training pipeline uses multiple datasets in different stages:

### 1. **Pretraining** (`base_train.py`)
- **Dataset**: FineWeb-EDU (single dataset, multiple shards)
- **Total**: ~24GB compressed, 240 shards

### 2. **Midtraining** (`mid_train.py`)
- **TaskMixture** with 3 datasets:
  - **SmolTalk**: 460K conversations
  - **MMLU** (auxiliary_train): 100K multiple-choice problems
  - **GSM8K**: 8K math problems
- **Total**: 568K rows

### 3. **SFT** (`chat_sft.py`)
- **TaskMixture** with 4 datasets:
  - **ARC-Easy**: 2.3K rows
  - **ARC-Challenge**: 1.1K rows
  - **GSM8K**: 8K rows
  - **SmolTalk**: 10K rows (subset)
- **Total**: 21.4K rows

## Integration Points

### Base Training (`base_train.py`)

**Location**: Around line 294-304 (training loop logging)

```python
# Add import at top
from scripts.enhanced_wandb_logging import enhance_wandb_logging

# Replace existing logging block (around line 294-304):
if step % 100 == 0:
    enhance_wandb_logging(
        wandb_run=wandb_run,
        step=step,
        device=device,
        dt=dt,
        tokens_per_sec=tok_per_sec,
        mfu=mfu,
        flops_so_far=flops_so_far,
        total_training_time=total_training_time,
        training_stage="pretraining",
        dataset_name="FineWeb-EDU",
        train_loss=debiased_smooth_loss,
        train_lrm=lrm,
    )
```

### Mid Training (`mid_train.py`)

**Location**: Around line 257-267 (training loop logging)

```python
# Add imports at top
from scripts.enhanced_wandb_logging import enhance_wandb_logging, DatasetTracker

# After dataset initialization (around line 95-100):
task_names = ["SmolTalk", "MMLU", "GSM8K"]
task_sizes = [
    len(SmolTalk(split="train")),
    len(MMLU(subset="auxiliary_train", split="train")),
    len(GSM8K(subset="main", split="train")),
]
dataset_tracker = DatasetTracker(task_names, task_sizes)

# Replace existing logging block (around line 257-267):
if step % 10 == 0:
    # Track which dataset we're currently on (you'll need to extract this from the generator)
    # For now, log mixture metrics
    dataset_tracker.log_step(wandb_run, step, log_every=10)
    
    enhance_wandb_logging(
        wandb_run=wandb_run,
        step=step,
        device=device,
        dt=dt,
        tokens_per_sec=tok_per_sec,
        mfu=mfu,
        flops_so_far=flops_so_far,
        total_training_time=total_training_time,
        training_stage="midtraining",
        train_loss=debiased_smooth_loss,
        train_lrm=lrm,
    )
```

**Enhanced Dataset Tracking** (requires modifying `mid_data_generator`):

```python
# Modify mid_data_generator to track dataset source
def mid_data_generator(split):
    global last_step, approx_progress, current_dataset
    # ... existing code ...
    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            # Extract which task this came from
            task_idx, local_idx = dataset.index_map[cursor % len(dataset.index_map)]
            current_dataset = task_names[task_idx]  # Track current dataset
            dataset_tracker.increment(current_dataset)
            # ... rest of existing code ...
```

### SFT Training (`chat_sft.py`)

**Location**: Around line 237-243 (training loop logging)

```python
# Add imports at top
from scripts.enhanced_wandb_logging import enhance_wandb_logging, DatasetTracker, log_evaluation_metrics_3d

# After dataset initialization (around line 81-87):
task_names = ["ARC-Easy", "ARC-Challenge", "GSM8K", "SmolTalk"]
task_sizes = [
    len(ARC(subset="ARC-Easy", split="train")),
    len(ARC(subset="ARC-Challenge", split="train")),
    len(GSM8K(subset="main", split="train")),
    len(SmolTalk(split="train", stop=10_000)),
]
dataset_tracker = DatasetTracker(task_names, task_sizes)

# Replace existing logging block (around line 237-243):
wandb_run.log({
    "step": step,
    "lrm": lrm,
    "train_loss": train_loss_item,
    "num_tokens": num_tokens_item,
})

# Add enhanced logging
enhance_wandb_logging(
    wandb_run=wandb_run,
    step=step,
    device=device,
    dt=0.0,  # Not directly available, estimate or track separately
    tokens_per_sec=0.0,  # Calculate if needed
    mfu=0.0,  # Not calculated in SFT, set to 0
    flops_so_far=0.0,  # Not tracked in SFT
    total_training_time=0.0,  # Track separately if needed
    training_stage="sft",
    train_loss=train_loss_item,
    num_tokens=num_tokens_item,
    lrm=lrm,
)

# Track dataset usage
dataset_tracker.log_step(wandb_run, step, log_every=50)

# Replace evaluation logging (around line 201-204):
if last_step or (step > 0 and step % eval_metrics_every == 0):
    # ... existing evaluation code ...
    log_evaluation_metrics_3d(
        wandb_run=wandb_run,
        step=step,
        metrics=metrics,
        training_stage="sft",
    )
```

## Enhanced Dataset Tracking

To track which dataset is being used in TaskMixture, you can modify the data generators:

### For `mid_train.py`:

```python
# Add global variable
current_dataset_name = None

# Modify mid_data_generator
def mid_data_generator(split):
    global last_step, approx_progress, current_dataset_name
    # ... existing code ...
    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            # Determine which task this conversation belongs to
            # TaskMixture.index_map contains (task_idx, local_idx) pairs
            if hasattr(dataset, 'index_map'):
                task_idx, local_idx = dataset.index_map[cursor % len(dataset.index_map)]
                current_dataset_name = task_names[task_idx]
                dataset_tracker.increment(current_dataset_name)
            # ... rest of code ...
```

### For `chat_sft.py`:

```python
# Modify sft_data_generator to track dataset source
def sft_data_generator(dataset, batch_size):
    # ... existing code ...
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            # Determine which task this belongs to
            if hasattr(dataset, 'index_map'):
                task_idx, local_idx = dataset.index_map[i % len(dataset.index_map)]
                current_dataset_name = task_names[task_idx]
                dataset_tracker.increment(current_dataset_name)
            # ... rest of code ...
```

## What Gets Logged

With enhanced logging, you'll get:

1. **Training Stage**: `pretraining`, `midtraining`, `sft`, or `rl`
2. **Dataset Metrics**:
   - Current dataset name
   - Dataset coverage percentage
   - Examples seen per dataset
   - Task mixture ratios
3. **GPU Metrics**:
   - Memory allocated/reserved
   - Peak memory usage
4. **System Metrics**:
   - CPU usage
   - RAM usage
   - Disk usage
5. **Efficiency Metrics**:
   - MFU percentage
   - Tokens per second
   - FLOPs per second
   - Training time
6. **Evaluation Metrics** (for 3D visualization):
   - Normalized accuracies
   - Mean accuracy across tasks

## Benefits for 3D Visualization

These enhanced metrics enable:

1. **Dataset-Specific Visualizations**: See how performance differs across datasets
2. **Multi-Dataset Trajectories**: Track training progress per dataset
3. **Resource Utilization**: 3D plots showing GPU/system usage vs performance
4. **Stage Comparison**: Compare metrics across pretraining → midtraining → SFT

## Quick Integration Checklist

- [ ] Add imports to training scripts
- [ ] Replace existing `wandb_run.log()` calls with `enhance_wandb_logging()`
- [ ] Add `DatasetTracker` for multi-dataset stages (midtraining, SFT)
- [ ] Update data generators to track dataset source
- [ ] Use `log_evaluation_metrics_3d()` for evaluation logging
- [ ] Test with a short run to verify metrics are logged correctly

## Example Integration

See `scripts/enhanced_wandb_logging.py` for complete implementation and helper functions.

The enhanced logging is **optional** - your existing logging will still work. The enhanced version adds:
- More detailed metrics
- Dataset tracking
- Better support for 3D visualizations
- Stage identification

