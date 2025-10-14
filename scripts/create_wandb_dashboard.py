"""
Create a Weights & Biases dashboard for tracking training metrics.

This script creates a dashboard that displays the key metrics from the
nanochat training pipeline: MMLU, ARC-Easy, GSM8K, and HumanEval accuracies.

Usage:
    python scripts/create_wandb_dashboard.py
"""

import wandb
import argparse
from typing import Dict, Any

def create_nanochat_dashboard(project_name: str = "nanochat-mid") -> str:
    """
    Create a W&B dashboard for nanochat training metrics.
    
    Args:
        project_name: The W&B project name to create the dashboard for
        
    Returns:
        The dashboard URL
    """
    
    # Initialize wandb to get access to the API
    api = wandb.Api()
    
    # Dashboard configuration
    dashboard_config = {
        "version": 1,
        "name": "NanoChat Training Metrics",
        "description": "Dashboard for tracking MMLU, ARC-Easy, GSM8K, and HumanEval accuracies during nanochat training",
        "project": project_name,
        "entity": None,  # Will use the default entity
        "spec": {
            "layout": {
                "panels": [
                    # MMLU Accuracy Panel
                    {
                        "id": "mmlu_panel",
                        "type": "line",
                        "title": "MMLU Accuracy",
                        "content": {
                            "query": {
                                "fields": ["step", "mmlu_acc"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": "mmlu_acc",
                                "title": "MMLU Accuracy Over Time",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Accuracy", "min": 0, "max": 1}
                            }
                        },
                        "position": {"x": 0, "y": 0, "w": 6, "h": 4}
                    },
                    
                    # ARC-Easy Accuracy Panel
                    {
                        "id": "arc_easy_panel",
                        "type": "line",
                        "title": "ARC-Easy Accuracy",
                        "content": {
                            "query": {
                                "fields": ["step", "arc_easy_acc"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": "arc_easy_acc",
                                "title": "ARC-Easy Accuracy Over Time",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Accuracy", "min": 0, "max": 1}
                            }
                        },
                        "position": {"x": 6, "y": 0, "w": 6, "h": 4}
                    },
                    
                    # GSM8K Accuracy Panel
                    {
                        "id": "gsm8k_panel",
                        "type": "line",
                        "title": "GSM8K Accuracy",
                        "content": {
                            "query": {
                                "fields": ["step", "gsm8k_acc"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": "gsm8k_acc",
                                "title": "GSM8K Accuracy Over Time",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Accuracy", "min": 0, "max": 1}
                            }
                        },
                        "position": {"x": 0, "y": 4, "w": 6, "h": 4}
                    },
                    
                    # HumanEval Accuracy Panel
                    {
                        "id": "humaneval_panel",
                        "type": "line",
                        "title": "HumanEval Accuracy",
                        "content": {
                            "query": {
                                "fields": ["step", "humaneval_acc"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": "humaneval_acc",
                                "title": "HumanEval Accuracy Over Time",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Accuracy", "min": 0, "max": 1}
                            }
                        },
                        "position": {"x": 6, "y": 4, "w": 6, "h": 4}
                    },
                    
                    # Validation Loss Panel (if available)
                    {
                        "id": "val_loss_panel",
                        "type": "line",
                        "title": "Validation Loss",
                        "content": {
                            "query": {
                                "fields": ["step", "val_loss"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": "val_loss",
                                "title": "Validation Loss Over Time",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Loss", "scale": "log"}
                            }
                        },
                        "position": {"x": 0, "y": 8, "w": 12, "h": 4}
                    },
                    
                    # Training Metrics Summary Panel
                    {
                        "id": "training_metrics_panel",
                        "type": "line",
                        "title": "Training Metrics",
                        "content": {
                            "query": {
                                "fields": ["step", "train/loss", "train/lrm", "train/mfu"],
                                "filters": {
                                    "config.project_name": project_name
                                }
                            },
                            "chart": {
                                "type": "line",
                                "x": "step",
                                "y": ["train/loss", "train/lrm", "train/mfu"],
                                "title": "Training Loss, Learning Rate Multiplier, and MFU",
                                "xAxis": {"title": "Training Step"},
                                "yAxis": {"title": "Value"}
                            }
                        },
                        "position": {"x": 0, "y": 12, "w": 12, "h": 4}
                    }
                ]
            }
        }
    }
    
    try:
        # Create the dashboard using the W&B API
        dashboard = api.create_dashboard(
            project=project_name,
            name=dashboard_config["name"],
            description=dashboard_config["description"]
        )
        
        # Update the dashboard with the panel configuration
        dashboard.spec = dashboard_config["spec"]
        dashboard.update()
        
        dashboard_url = f"https://wandb.ai/{api.default_entity}/{project_name}/dashboards/{dashboard.id}"
        
        print(f"Dashboard created successfully!")
        print(f"Dashboard URL: {dashboard_url}")
        print(f"Dashboard Name: {dashboard_config['name']}")
        print(f"Project: {project_name}")
        
        return dashboard_url
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        raise


def create_simple_dashboard_config():
    """
    Create a simpler dashboard configuration that can be used with the W&B UI.
    This returns the configuration that can be copied into the W&B dashboard builder.
    """
    
    config = {
        "name": "NanoChat Training Metrics",
        "description": "Dashboard for tracking MMLU, ARC-Easy, GSM8K, and HumanEval accuracies during nanochat training",
        "panels": [
            {
                "title": "MMLU Accuracy",
                "type": "line",
                "query": "step, mmlu_acc",
                "chart_type": "line",
                "x_axis": "step",
                "y_axis": "mmlu_acc",
                "y_min": 0,
                "y_max": 1
            },
            {
                "title": "ARC-Easy Accuracy", 
                "type": "line",
                "query": "step, arc_easy_acc",
                "chart_type": "line",
                "x_axis": "step",
                "y_axis": "arc_easy_acc",
                "y_min": 0,
                "y_max": 1
            },
            {
                "title": "GSM8K Accuracy",
                "type": "line", 
                "query": "step, gsm8k_acc",
                "chart_type": "line",
                "x_axis": "step",
                "y_axis": "gsm8k_acc",
                "y_min": 0,
                "y_max": 1
            },
            {
                "title": "HumanEval Accuracy",
                "type": "line",
                "query": "step, humaneval_acc", 
                "chart_type": "line",
                "x_axis": "step",
                "y_axis": "humaneval_acc",
                "y_min": 0,
                "y_max": 1
            },
            {
                "title": "Validation Loss",
                "type": "line",
                "query": "step, val_loss",
                "chart_type": "line", 
                "x_axis": "step",
                "y_axis": "val_loss",
                "y_scale": "log"
            }
        ]
    }
    
    return config


def print_dashboard_instructions(project_name: str = "nanochat-mid"):
    """
    Print instructions for manually creating the dashboard in W&B UI.
    """
    
    print("\n" + "="*80)
    print("MANUAL DASHBOARD CREATION INSTRUCTIONS")
    print("="*80)
    print(f"1. Go to your W&B project: https://wandb.ai/your-username/{project_name}")
    print("2. Click on 'Dashboards' in the left sidebar")
    print("3. Click 'Create Dashboard'")
    print("4. Add the following panels:")
    print()
    
    panels = [
        ("MMLU Accuracy", "step", "mmlu_acc"),
        ("ARC-Easy Accuracy", "step", "arc_easy_acc"), 
        ("GSM8K Accuracy", "step", "gsm8k_acc"),
        ("HumanEval Accuracy", "step", "humaneval_acc"),
        ("Validation Loss", "step", "val_loss"),
    ]
    
    for i, (title, x_field, y_field) in enumerate(panels, 1):
        print(f"   {i}. Panel: '{title}'")
        print(f"      - Type: Line Chart")
        print(f"      - X-axis: {x_field}")
        print(f"      - Y-axis: {y_field}")
        if y_field in ["mmlu_acc", "arc_easy_acc", "gsm8k_acc", "humaneval_acc"]:
            print(f"      - Y-axis range: 0 to 1")
        elif y_field == "val_loss":
            print(f"      - Y-axis scale: Logarithmic")
        print()
    
    print("5. Save your dashboard with a name like 'NanoChat Training Metrics'")
    print("6. The dashboard will automatically update as new training runs are logged")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create W&B dashboard for nanochat training metrics")
    parser.add_argument("--project", type=str, default="nanochat-mid", 
                       help="W&B project name (default: nanochat-mid)")
    parser.add_argument("--manual", action="store_true", 
                       help="Print manual creation instructions instead of creating via API")
    
    args = parser.parse_args()
    
    if args.manual:
        print_dashboard_instructions(args.project)
    else:
        try:
            dashboard_url = create_nanochat_dashboard(args.project)
            print(f"\nYour dashboard is ready at: {dashboard_url}")
            print("\nTip: The dashboard will automatically update as new training runs log metrics!")
        except Exception as e:
            print(f"\nCould not create dashboard automatically: {e}")
            print("\nFalling back to manual instructions...")
            print_dashboard_instructions(args.project)
