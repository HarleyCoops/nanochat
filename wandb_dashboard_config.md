# Weights & Biases Dashboard Configuration for NanoChat Training

This document provides the configuration for creating a dashboard to track the key training metrics from your nanochat model.

## Dashboard Overview

The dashboard tracks the following metrics during training:

1. **MMLU Accuracy** - Multitask language understanding across 57 subjects
2. **ARC-Easy Accuracy** - Elementary science questions 
3. **GSM8K Accuracy** - Grade-school mathematics reasoning
4. **HumanEval Accuracy** - Python code generation benchmark
5. **Validation Loss** - Training validation loss over time

## How to Create the Dashboard

### Option 1: Manual Creation (Recommended)

1. Go to your Weights & Biases project: `https://wandb.ai/your-username/nanochat-mid`
2. Click on **'Dashboards'** in the left sidebar
3. Click **'Create Dashboard'**
4. Add the following panels:

#### Panel 1: MMLU Accuracy
- **Type**: Line Chart
- **X-axis**: `step`
- **Y-axis**: `mmlu_acc`
- **Y-axis range**: 0 to 1
- **Title**: "MMLU Accuracy"

#### Panel 2: ARC-Easy Accuracy
- **Type**: Line Chart
- **X-axis**: `step`
- **Y-axis**: `arc_easy_acc`
- **Y-axis range**: 0 to 1
- **Title**: "ARC-Easy Accuracy"

#### Panel 3: GSM8K Accuracy
- **Type**: Line Chart
- **X-axis**: `step`
- **Y-axis**: `gsm8k_acc`
- **Y-axis range**: 0 to 1
- **Title**: "GSM8K Accuracy"

#### Panel 4: HumanEval Accuracy
- **Type**: Line Chart
- **X-axis**: `step`
- **Y-axis**: `humaneval_acc`
- **Y-axis range**: 0 to 1
- **Title**: "HumanEval Accuracy"

#### Panel 5: Validation Loss
- **Type**: Line Chart
- **X-axis**: `step`
- **Y-axis**: `val_loss`
- **Y-axis scale**: Logarithmic
- **Title**: "Validation Loss"

5. Save your dashboard with a name like **"NanoChat Training Metrics"**

### Option 2: Using the Script

Run the provided script to get instructions:

```powershell
py scripts/create_wandb_dashboard.py --manual
```

## Current Integration

Your training scripts are already set up to log these metrics to Weights & Biases:

- **`scripts/chat_sft.py`** - Logs all four accuracy metrics every `eval_metrics_every` steps (default: 200)
- **`scripts/base_train.py`** - Logs core metrics and validation loss
- **`scripts/mid_train.py`** - Logs training metrics and validation loss

## Metric Details

### MMLU (Massive Multitask Language Understanding)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Baseline**: 0.25 (25% - random chance for 4-choice questions)
- **Current Performance**: ~31.51% (from MODEL_CARD.md)

### ARC-Easy (AI2 Reasoning Challenge - Easy)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Baseline**: 0.25 (25% - random chance for 4-choice questions)
- **Current Performance**: ~38.76% (from MODEL_CARD.md)

### GSM8K (Grade School Math 8K)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Baseline**: 0.0 (0% - open-ended math problems)
- **Current Performance**: ~4.55% (from MODEL_CARD.md)

### HumanEval (Python Code Generation)
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Baseline**: 0.0 (0% - open-ended code generation)
- **Current Performance**: ~8.54% (from MODEL_CARD.md)

## Dashboard Layout Recommendation

Arrange the panels in a 2x3 grid:
```
[MMLU Accuracy]    [ARC-Easy Accuracy]
[GSM8K Accuracy]   [HumanEval Accuracy]
[Validation Loss - Full Width]
```

## Automatic Updates

Once created, the dashboard will automatically update as new training runs log metrics. No additional configuration is needed - just run your training scripts as usual and the metrics will appear in real-time.

## Troubleshooting

If metrics don't appear:
1. Ensure your training script is logging to the correct W&B project (`nanochat-mid`)
2. Check that the run name is not set to "dummy" (which disables W&B logging)
3. Verify that `eval_metrics_every` is set to a reasonable value (e.g., 200 steps)
4. Make sure you have the required dependencies installed (`wandb` package)

## Example Training Command

To run training with metrics logging:

```powershell
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft --run "my-experiment-name"
```

Replace `"my-experiment-name"` with your desired run name (not "dummy").
