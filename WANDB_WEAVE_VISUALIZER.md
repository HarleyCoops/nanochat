# W&B Weave 3D Training Visualizer for NanoChat

This module creates stunning 3D visualizations of your NanoChat training pipeline, showcasing the new 3D chart capabilities in **W&B Server v0.75.0** ([release notes](https://github.com/wandb/server/releases/tag/0.75.0)).

## Features

- **3D Training Trajectory**: Visualize training progress in 3D space (Step × Loss × Accuracy)
- **Hyperparameter Space Exploration**: 3D visualization with semantic coloring by performance
- **Performance Landscape**: See training efficiency metrics in 3D (MFU × Throughput × Accuracy)
- **Multi-Metric Comparison**: Compare multiple evaluation metrics simultaneously
- **Colorful Stage Visualization**: Distinct colors for each training stage (pretraining, midtraining, SFT, RL)

## New W&B Server v0.75.0 Features Showcased

- **3D Charts**: Full support for 3D scatter plots and trajectories
- **Semantic Coloring**: Color plots based on config properties (hyperparameters, performance metrics)
- **Full-Fidelity Rendering**: Detailed system metrics visualization
- **Interactive Visualizations**: Rotate, zoom, and explore your training data

## Installation

```bash
pip install wandb weave plotly numpy pandas psutil
```

## Quick Start

### 1. Run Training with W&B Enabled

Make sure your training runs log to W&B (set `run != "dummy"`):

```bash
# Example: Base training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --run "my-pretraining-run"

# Example: SFT training  
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run "my-sft-run"
```

### 2. Generate 3D Visualizations

```bash
python scripts/weave_training_visualizer.py --project nanochat
```

This will create a Weave app with all 3D visualizations!

### 3. Enhanced Logging (Optional)

For even richer visualizations, integrate enhanced logging into your training scripts:

```python
from scripts.enhanced_wandb_logging import enhance_wandb_logging

# In your training loop:
enhance_wandb_logging(
    wandb_run=wandb_run,
    step=step,
    device=device,
    dt=dt,
    tokens_per_sec=tok_per_sec,
    mfu=mfu,
    flops_so_far=flops_so_far,
    total_training_time=total_training_time,
    training_stage="pretraining",  # or "midtraining", "sft", "rl"
    train_loss=loss.item(),
)
```

## Visualizations

### 1. 3D Training Trajectory
- **X-axis**: Training step
- **Y-axis**: Loss
- **Z-axis**: Accuracy
- **Color**: Training stage (pretraining=red, midtraining=teal, SFT=blue, RL=green)

### 2. Hyperparameter Space
- **X-axis**: Model depth
- **Y-axis**: Model dimension  
- **Z-axis**: Batch size
- **Color**: Best accuracy (semantic coloring)
- **Size**: Learning rate

### 3. Performance Landscape
- **X-axis**: Model FLOPs Utilization (MFU %)
- **Y-axis**: Tokens per second
- **Z-axis**: Accuracy
- **Color**: Training stage

### 4. Multi-Metric 3D Plot
- **X-axis**: MMLU Accuracy
- **Y-axis**: ARC-Easy Accuracy
- **Z-axis**: GSM8K Accuracy
- **Size**: HumanEval Accuracy
- **Color**: Training stage

## Usage Examples

### Basic Usage

```python
from scripts.weave_training_visualizer import create_complete_visualization_app

# Create visualizations for your project
app = create_complete_visualization_app("nanochat")
```

### Custom Project

```bash
python scripts/weave_training_visualizer.py --project my-custom-project
```

### Integration with Training Scripts

Add to your training scripts:

```python
# In base_train.py, mid_train.py, chat_sft.py, etc.
from scripts.enhanced_wandb_logging import enhance_wandb_logging, log_evaluation_metrics_3d

# Enhanced logging in training loop
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
)

# Enhanced evaluation logging
metrics = {
    "mmlu_acc": mmlu_accuracy,
    "arc_easy_acc": arc_easy_accuracy,
    "gsm8k_acc": gsm8k_accuracy,
    "humaneval_acc": humaneval_accuracy,
}
log_evaluation_metrics_3d(
    wandb_run=wandb_run,
    step=step,
    metrics=metrics,
    training_stage="sft",
)
```

## Advanced Features

### Semantic Coloring by Hyperparameters

The visualizations automatically use semantic coloring based on config properties. This is a new feature in W&B Server v0.75.0 that allows you to visualize config-driven effects:

- **Hyperparameter Space Plot**: Colors points by best accuracy achieved
- **Training Trajectory**: Colors by training stage
- **Performance Landscape**: Colors by efficiency metrics

### Multi-Stage Pipeline Visualization

The visualizations automatically detect and visualize the full training pipeline:
1. **Pretraining** (base model)
2. **Midtraining** (instruction adaptation)
3. **SFT** (supervised fine-tuning)
4. **RL** (reinforcement learning, optional)

Each stage is color-coded for easy identification.

## Troubleshooting

### No Data Shown

- Ensure you have runs logged to W&B (`run != "dummy"`)
- Check that metrics are being logged (loss, accuracy, etc.)
- Verify your project name matches

### Missing Metrics

- Use `enhanced_wandb_logging.py` to add more metrics
- Ensure evaluation metrics are logged during training
- Check that your training scripts log to W&B

### Weave Not Working

- Update to latest W&B: `pip install --upgrade wandb weave`
- Ensure W&B Server is updated to v0.75.0+
- Check Weave initialization: `weave.init("project-name")`

## References

- [W&B Server v0.75.0 Release Notes](https://github.com/wandb/server/releases/tag/0.75.0)
- [W&B Weave Documentation](https://weave-docs.wandb.ai/)
- [W&B Python SDK](https://docs.wandb.ai/guides/track)

## Next Steps

1. **Run Training**: Train your models with W&B enabled
2. **Generate Visualizations**: Run the Weave app script
3. **Explore**: Rotate and zoom the 3D plots to discover insights
4. **Share**: Share your visualizations with your team!

## Contributing

To add more visualizations:
1. Add data preparation function in `weave_training_visualizer.py`
2. Create visualization function with `@weave.op()` decorator
3. Add to `create_complete_visualization_app()`

Happy visualizing!

