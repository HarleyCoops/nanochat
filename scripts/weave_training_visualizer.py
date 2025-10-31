"""
W&B Weave App: 3D Training Visualizer for NanoChat

This Weave app creates stunning 3D visualizations of the training pipeline,
showcasing the new 3D chart capabilities in W&B Server v0.75.0.

Features:
- 3D training trajectory visualization
- Hyperparameter space exploration with semantic coloring
- Performance landscape 3D surfaces
- Multi-stage training comparison
- Colorful, interactive visualizations

Usage:
    python scripts/weave_training_visualizer.py --project nanochat
    # Then visit the generated Weave app URL

Requirements:
    pip install wandb weave plotly numpy pandas
"""

import wandb
import weave
import numpy as np
from typing import Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Initialize Weave
weave.init("nanochat-training-visualizer")


def fetch_training_runs(project_name: str = "nanochat") -> List[Dict]:
    """Fetch all training runs from W&B project."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    runs_data = []
    for run in runs:
        history = run.history()
        config = dict(run.config)
        summary = dict(run.summary)
        
        runs_data.append({
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "config": config,
            "summary": summary,
            "history": history.to_dict() if not history.empty else {},
        })
    
    return runs_data


def extract_training_stage(run_name: str) -> str:
    """Extract training stage from run name."""
    name_lower = run_name.lower()
    if "base" in name_lower or "pretrain" in name_lower:
        return "pretraining"
    elif "mid" in name_lower:
        return "midtraining"
    elif "sft" in name_lower:
        return "sft"
    elif "rl" in name_lower or "reinforcement" in name_lower:
        return "rl"
    else:
        return "unknown"


def prepare_3d_trajectory_data(runs_data: List[Dict]) -> pd.DataFrame:
    """Prepare data for 3D trajectory visualization."""
    rows = []
    
    for run in runs_data:
        history = run.get("history", {})
        if not history:
            continue
        
        stage = extract_training_stage(run["run_name"])
        config = run.get("config", {})
        
        # Extract metrics from history
        steps = history.get("step", [])
        train_loss = history.get("train/loss", [])
        val_loss = history.get("val_loss", [])
        val_bpb = history.get("val/bpb", [])
        
        # Extract accuracy metrics
        mmlu_acc = history.get("mmlu_acc", [])
        arc_easy_acc = history.get("arc_easy_acc", [])
        gsm8k_acc = history.get("gsm8k_acc", [])
        humaneval_acc = history.get("humaneval_acc", [])
        core_metric = history.get("core_metric", [])
        
        # Convert to lists if needed
        if not isinstance(steps, list):
            steps = steps.tolist() if hasattr(steps, 'tolist') else []
        
        # Create pairs
        max_len = max(len(steps), len(train_loss) if train_loss else 0, 
                     len(val_loss) if val_loss else 0, len(val_bpb) if val_bpb else 0)
        
        for i in range(max_len):
            step = steps[i] if i < len(steps) else None
            
            # Get loss (prefer train loss, fallback to val loss or bpb)
            loss = None
            if train_loss and i < len(train_loss):
                loss = train_loss[i]
            elif val_loss and i < len(val_loss):
                loss = val_loss[i]
            elif val_bpb and i < len(val_bpb):
                loss = val_bpb[i]
            
            # Find closest accuracy
            accuracy = None
            accuracy_type = None
            
            # Get accuracy at closest step
            acc_metrics = [
                ("mmlu", mmlu_acc, steps),
                ("arc_easy", arc_easy_acc, steps),
                ("gsm8k", gsm8k_acc, steps),
                ("humaneval", humaneval_acc, steps),
                ("core", core_metric, steps),
            ]
            
            for acc_type, acc_values, acc_steps in acc_metrics:
                if acc_values and step:
                    if isinstance(acc_values, list) and len(acc_values) > 0:
                        # Find closest step
                        if isinstance(acc_steps, list) and len(acc_steps) == len(acc_values):
                            closest_idx = min(range(len(acc_steps)), 
                                             key=lambda j: abs(acc_steps[j] - step) if j < len(acc_steps) else float('inf'))
                            if closest_idx < len(acc_values) and acc_values[closest_idx] is not None:
                                accuracy = acc_values[closest_idx]
                                accuracy_type = acc_type
                                break
            
            if step is not None and loss is not None:
                rows.append({
                    "step": float(step),
                    "loss": float(loss),
                    "accuracy": float(accuracy) if accuracy is not None else 0.0,
                    "accuracy_type": accuracy_type if accuracy_type else "none",
                    "stage": stage,
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "depth": config.get("depth", 20),
                    "model_dim": config.get("model_dim", config.get("depth", 20) * 64),
                    "batch_size": config.get("device_batch_size", 32),
                })
    
    return pd.DataFrame(rows)


def prepare_hyperparameter_space_data(runs_data: List[Dict]) -> pd.DataFrame:
    """Prepare data for 3D hyperparameter space visualization."""
    rows = []
    
    for run in runs_data:
        config = run.get("config", {})
        summary = run.get("summary", {})
        
        depth = config.get("depth", 20)
        model_dim = config.get("model_dim", depth * 64)
        batch_size = config.get("device_batch_size", 32)
        learning_rate = config.get("learning_rate", config.get("embedding_lr", 6e-4))
        max_seq_len = config.get("max_seq_len", 2048)
        
        # Extract performance metrics
        best_loss = summary.get("train/loss", summary.get("val_loss", summary.get("val/bpb", None)))
        best_mmlu = summary.get("mmlu_acc", None)
        best_arc_easy = summary.get("arc_easy_acc", None)
        best_gsm8k = summary.get("gsm8k_acc", None)
        best_humaneval = summary.get("humaneval_acc", None)
        best_core = summary.get("core_metric", None)
        
        accuracies = [v for v in [best_mmlu, best_arc_easy, best_gsm8k, best_humaneval, best_core] if v is not None]
        best_accuracy = max(accuracies) if accuracies else None
        
        stage = extract_training_stage(run["run_name"])
        
        rows.append({
            "depth": float(depth),
            "model_dim": float(model_dim),
            "batch_size": float(batch_size),
            "learning_rate": float(learning_rate),
            "max_seq_len": float(max_seq_len),
            "best_loss": float(best_loss) if best_loss is not None else 0.0,
            "best_accuracy": float(best_accuracy) if best_accuracy is not None else 0.0,
            "best_mmlu": float(best_mmlu) if best_mmlu is not None else 0.0,
            "best_arc_easy": float(best_arc_easy) if best_arc_easy is not None else 0.0,
            "best_gsm8k": float(best_gsm8k) if best_gsm8k is not None else 0.0,
            "best_humaneval": float(best_humaneval) if best_humaneval is not None else 0.0,
            "best_core": float(best_core) if best_core is not None else 0.0,
            "stage": stage,
            "run_id": run["run_id"],
            "run_name": run["run_name"],
        })
    
    return pd.DataFrame(rows)


def prepare_performance_landscape_data(runs_data: List[Dict]) -> pd.DataFrame:
    """Prepare data for 3D performance landscape visualization."""
    rows = []
    
    for run in runs_data:
        history = run.get("history", {})
        if not history:
            continue
        
        config = run.get("config", {})
        stage = extract_training_stage(run["run_name"])
        
        steps = history.get("step", [])
        mfu = history.get("train/mfu", [])
        tok_per_sec = history.get("train/tok_per_sec", [])
        total_time = history.get("total_training_time", [])
        flops = history.get("total_training_flops", [])
        
        mmlu_acc = history.get("mmlu_acc", [])
        arc_easy_acc = history.get("arc_easy_acc", [])
        
        max_len = max(len(steps) if steps else 0, len(mfu) if mfu else 0,
                     len(tok_per_sec) if tok_per_sec else 0)
        
        for i in range(max_len):
            step = steps[i] if i < len(steps) else None
            mfu_val = mfu[i] if i < len(mfu) else None
            tok_per_sec_val = tok_per_sec[i] if i < len(tok_per_sec) else None
            time_val = total_time[i] if i < len(total_time) else None
            
            # Get accuracy
            accuracy = None
            if mmlu_acc and i < len(mmlu_acc):
                accuracy = mmlu_acc[i]
            elif arc_easy_acc and i < len(arc_easy_acc):
                accuracy = arc_easy_acc[i]
            
            if step is not None and mfu_val is not None and tok_per_sec_val is not None:
                rows.append({
                    "step": float(step),
                    "mfu": float(mfu_val),
                    "tokens_per_sec": float(tok_per_sec_val),
                    "training_time": float(time_val) if time_val is not None else 0.0,
                    "accuracy": float(accuracy) if accuracy is not None else 0.0,
                    "flops": float(flops[i]) if flops and i < len(flops) else 0.0,
                    "stage": stage,
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "depth": float(config.get("depth", 20)),
                })
    
    return pd.DataFrame(rows)


@weave.op()
def create_3d_trajectory_plot(project_name: str = "nanochat") -> weave.ops.Op:
    """Create a 3D trajectory plot showing training progress."""
    runs_data = fetch_training_runs(project_name)
    df = prepare_3d_trajectory_data(runs_data)
    
    if df.empty:
        return weave.ops.make_list([])
    
    df = df.dropna(subset=["step", "loss", "accuracy"])
    if df.empty:
        return weave.ops.make_list([])
    
    # Create color mapping for stages
    stage_colors = {
        "pretraining": "#FF6B6B",  # Red
        "midtraining": "#4ECDC4",  # Teal
        "sft": "#45B7D1",  # Blue
        "rl": "#96CEB4",  # Green
        "unknown": "#FFEAA7",  # Yellow
    }
    
    fig = go.Figure()
    
    # Plot each stage separately for better color control
    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]
        fig.add_trace(go.Scatter3d(
            x=stage_df["step"].tolist(),
            y=stage_df["loss"].tolist(),
            z=stage_df["accuracy"].tolist(),
            mode='markers+lines',
            name=stage,
            marker=dict(
                size=8,
                color=stage_colors.get(stage, "#95A5A6"),
                opacity=0.7,
            ),
            line=dict(
                color=stage_colors.get(stage, "#95A5A6"),
                width=2,
            ),
            text=stage_df["run_name"].tolist(),
            hovertemplate='<b>%{text}</b><br>' +
                         'Stage: ' + stage + '<br>' +
                         'Step: %{x}<br>' +
                         'Loss: %{y:.4f}<br>' +
                         'Accuracy: %{z:.4f}<extra></extra>',
        ))
    
    fig.update_layout(
        title="3D Training Trajectory - Step vs Loss vs Accuracy",
        scene=dict(
            xaxis_title="Training Step",
            yaxis_title="Loss",
            zaxis_title="Accuracy",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        height=800,
    )
    
    return weave.ops.plotly_plot(fig)


@weave.op()
def create_hyperparameter_space_plot(project_name: str = "nanochat") -> weave.ops.Op:
    """Create a 3D hyperparameter space visualization with semantic coloring."""
    runs_data = fetch_training_runs(project_name)
    df = prepare_hyperparameter_space_data(runs_data)
    
    if df.empty:
        return weave.ops.make_list([])
    
    df = df.dropna(subset=["depth", "model_dim", "batch_size", "best_accuracy"])
    if df.empty:
        return weave.ops.make_list([])
    
    fig = go.Figure(data=go.Scatter3d(
        x=df["depth"].tolist(),
        y=df["model_dim"].tolist(),
        z=df["batch_size"].tolist(),
        mode='markers',
        marker=dict(
            size=10,
            color=df["best_accuracy"].tolist(),  # Semantic coloring by accuracy
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Best Accuracy"),
            opacity=0.8,
        ),
        text=df["run_name"].tolist(),
        hovertemplate='<b>%{text}</b><br>' +
                      'Depth: %{x}<br>' +
                      'Model Dim: %{y}<br>' +
                      'Batch Size: %{z}<br>' +
                      'Best Accuracy: %{marker.color:.4f}<extra></extra>',
    ))
    
    fig.update_layout(
        title="3D Hyperparameter Space (Semantic Coloring by Accuracy)",
        scene=dict(
            xaxis_title="Model Depth",
            yaxis_title="Model Dimension",
            zaxis_title="Batch Size",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=800,
    )
    
    return weave.ops.plotly_plot(fig)


@weave.op()
def create_performance_landscape_plot(project_name: str = "nanochat") -> weave.ops.Op:
    """Create a 3D performance landscape visualization."""
    runs_data = fetch_training_runs(project_name)
    df = prepare_performance_landscape_data(runs_data)
    
    if df.empty:
        return weave.ops.make_list([])
    
    df = df.dropna(subset=["mfu", "tokens_per_sec", "accuracy"])
    if df.empty:
        return weave.ops.make_list([])
    
    # Create color mapping for stages
    stage_colors = {
        "pretraining": "#FF6B6B",
        "midtraining": "#4ECDC4",
        "sft": "#45B7D1",
        "rl": "#96CEB4",
        "unknown": "#FFEAA7",
    }
    
    fig = go.Figure()
    
    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]
        fig.add_trace(go.Scatter3d(
            x=stage_df["mfu"].tolist(),
            y=stage_df["tokens_per_sec"].tolist(),
            z=stage_df["accuracy"].tolist(),
            mode='markers',
            name=stage,
            marker=dict(
                size=8,
                color=stage_colors.get(stage, "#95A5A6"),
                opacity=0.7,
            ),
            text=stage_df["run_name"].tolist(),
            hovertemplate='<b>%{text}</b><br>' +
                         'MFU: %{x:.2f}%<br>' +
                         'Tokens/sec: %{y:,.0f}<br>' +
                         'Accuracy: %{z:.4f}<extra></extra>',
        ))
    
    fig.update_layout(
        title="3D Performance Landscape - MFU vs Throughput vs Accuracy",
        scene=dict(
            xaxis_title="Model FLOPs Utilization (%)",
            yaxis_title="Tokens per Second",
            zaxis_title="Accuracy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=800,
    )
    
    return weave.ops.plotly_plot(fig)


@weave.op()
def create_multi_metric_3d_plot(project_name: str = "nanochat") -> weave.ops.Op:
    """Create a 3D plot comparing multiple evaluation metrics."""
    runs_data = fetch_training_runs(project_name)
    df = prepare_hyperparameter_space_data(runs_data)
    
    if df.empty:
        return weave.ops.make_list([])
    
    df = df.dropna(subset=["best_mmlu", "best_arc_easy", "best_gsm8k"])
    if df.empty:
        return weave.ops.make_list([])
    
    # Create color mapping for stages
    stage_colors = {
        "pretraining": "#FF6B6B",
        "midtraining": "#4ECDC4",
        "sft": "#45B7D1",
        "rl": "#96CEB4",
        "unknown": "#FFEAA7",
    }
    
    fig = go.Figure()
    
    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]
        fig.add_trace(go.Scatter3d(
            x=stage_df["best_mmlu"].tolist(),
            y=stage_df["best_arc_easy"].tolist(),
            z=stage_df["best_gsm8k"].tolist(),
            mode='markers',
            name=stage,
            marker=dict(
                size=stage_df["best_humaneval"].tolist() * 100,  # Size by HumanEval
                color=stage_colors.get(stage, "#95A5A6"),
                opacity=0.7,
            ),
            text=stage_df["run_name"].tolist(),
            hovertemplate='<b>%{text}</b><br>' +
                         'MMLU: %{x:.4f}<br>' +
                         'ARC-Easy: %{y:.4f}<br>' +
                         'GSM8K: %{z:.4f}<br>' +
                         'HumanEval: %{marker.size:.4f}<extra></extra>',
        ))
    
    fig.update_layout(
        title="3D Multi-Metric Comparison - MMLU vs ARC-Easy vs GSM8K",
        scene=dict(
            xaxis_title="MMLU Accuracy",
            yaxis_title="ARC-Easy Accuracy",
            zaxis_title="GSM8K Accuracy",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        height=800,
    )
    
    return weave.ops.plotly_plot(fig)


@weave.op()
def create_complete_visualization_app(project_name: str = "nanochat") -> weave.ops.Dict:
    """Create the complete Weave app with all visualizations."""
    plots = weave.ops.make_dict({
        "trajectory_3d": create_3d_trajectory_plot(project_name),
        "hyperparameter_space": create_hyperparameter_space_plot(project_name),
        "performance_landscape": create_performance_landscape_plot(project_name),
        "multi_metric_3d": create_multi_metric_3d_plot(project_name),
    })
    
    return plots


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create W&B Weave app with 3D visualizations")
    parser.add_argument("--project", type=str, default="nanochat",
                       help="W&B project name (default: nanochat)")
    parser.add_argument("--entity", type=str, default=None,
                       help="W&B entity name (default: your default entity)")
    
    args = parser.parse_args()
    
    print("🚀 Creating W&B Weave app with 3D visualizations...")
    print(f"📊 Project: {args.project}")
    print("\n✨ Features showcased:")
    print("   - 3D charts (new in W&B Server v0.75.0)")
    print("   - Semantic coloring by config properties")
    print("   - Interactive visualizations")
    print("   - Multi-stage training pipeline views")
    print("\n📈 Visualizations being created:")
    print("   - 3D Training Trajectory")
    print("   - Hyperparameter Space (with semantic coloring)")
    print("   - Performance Landscape")
    print("   - Multi-Metric 3D Plot")
    print("\n🔄 Processing runs...")
    
    # Create the app
    try:
        app = create_complete_visualization_app(args.project)
        print("\n✅ Weave app created successfully!")
        print("\n🌐 Access your visualizations through the Weave interface")
        print("\n💡 Tip: Use the enhanced_wandb_logging module to add more metrics")
        print("   for even richer visualizations!")
    except Exception as e:
        print(f"\n❌ Error creating app: {e}")
        print("\n💡 Make sure you have runs logged to W&B first!")
        print("   Run your training scripts with W&B enabled (run != 'dummy')")
