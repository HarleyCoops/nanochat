"""
Example usage of the Hyperbolic Labs deployment API

This script demonstrates how to use the deployment tools programmatically.
"""

import os
from hyperbolic import (
    HyperbolicClient,
    ModelConfig,
    calculate_gpu_requirements,
    get_deployment_command,
    SMALL_MODEL,
    MEDIUM_MODEL,
    LARGE_MODEL,
)


def example_1_check_requirements():
    """Example 1: Check GPU requirements for different model sizes"""
    print("=" * 80)
    print("Example 1: GPU Requirements Analysis")
    print("=" * 80)
    
    for name, config in [("Small", SMALL_MODEL), ("Medium", MEDIUM_MODEL), ("Large", LARGE_MODEL)]:
        print(f"\n{name} Model Configuration:")
        print(f"  Depth: {config.depth}")
        print(f"  Model dim: {config.model_dim}")
        print(f"  Parameters: ~{config.num_params / 1e6:.1f}M")
        
        requirements = calculate_gpu_requirements(config)
        print(f"\n{requirements}")
        print()


def example_2_list_machines():
    """Example 2: List available machines"""
    print("=" * 80)
    print("Example 2: List Available Machines")
    print("=" * 80)
    
    # Check if API key is set
    if not os.environ.get('HYPERBOLIC_API_KEY'):
        print("\nSkipping: HYPERBOLIC_API_KEY not set")
        print("Set your API key with: export HYPERBOLIC_API_KEY='your-key-here'")
        return
    
    try:
        client = HyperbolicClient()
        machines = client.list_available_machines()
        
        print(f"\nFound {len(machines)} machines in total")
        
        # Show available machines
        available = [m for m in machines if not m.reserved and m.status == "node_ready"]
        print(f"Available machines: {len(available)}\n")
        
        for i, machine in enumerate(available[:5], 1):
            print(f"{i}. {machine.gpu_type}")
            print(f"   GPUs: {machine.gpus_total - machine.gpus_reserved}/{machine.gpus_total} available")
            print(f"   Price: ${machine.pricing:.2f}/hour")
            print(f"   RAM: {machine.ram_gb} GB")
            print()
        
        if len(available) > 5:
            print(f"... and {len(available) - 5} more available machines")
    
    except Exception as e:
        print(f"\nError: {e}")


def example_3_find_best_machine():
    """Example 3: Find best machine for a specific configuration"""
    print("=" * 80)
    print("Example 3: Find Best Machine")
    print("=" * 80)
    
    if not os.environ.get('HYPERBOLIC_API_KEY'):
        print("\nSkipping: HYPERBOLIC_API_KEY not set")
        return
    
    try:
        client = HyperbolicClient()
        
        # Calculate requirements for medium model
        config = MEDIUM_MODEL
        requirements = calculate_gpu_requirements(config)
        
        print(f"\nSearching for machine for Medium model:")
        print(f"  Min GPUs: {requirements.min_gpus}")
        print(f"  Preferred: {', '.join(requirements.gpu_type_preference)}")
        
        machine = client.find_best_machine(
            min_gpus=requirements.recommended_gpus,
            gpu_type_preference=requirements.gpu_type_preference,
            max_price_per_hour=20.0
        )
        
        if machine:
            print(f"\nBest match found:")
            print(f"  GPU Type: {machine.gpu_type}")
            print(f"  Available GPUs: {machine.gpus_total - machine.gpus_reserved}")
            print(f"  Price: ${machine.pricing:.2f}/hour")
            print(f"  RAM: {machine.ram_gb} GB")
            print(f"  Storage: {machine.storage_gb} GB")
            
            # Show deployment command
            cmd = get_deployment_command(config, requirements.recommended_gpus)
            print(f"\nDeployment command:")
            print(f"  {cmd}")
        else:
            print("\nNo suitable machine found with current constraints")
    
    except Exception as e:
        print(f"\nError: {e}")


def example_4_cost_estimation():
    """Example 4: Estimate training costs"""
    print("=" * 80)
    print("Example 4: Cost Estimation")
    print("=" * 80)
    
    # Estimate costs for different scenarios
    scenarios = [
        ("Small model, 2 GPUs, 2 hours", 2, 2.0, 2),
        ("Medium model, 4 GPUs, 5 hours", 4, 2.5, 5),
        ("Large model, 8 GPUs, 10 hours", 8, 3.0, 10),
    ]
    
    print("\nEstimated Training Costs:\n")
    print(f"{'Scenario':<40} {'GPUs':<5} {'$/hr':<8} {'Hours':<8} {'Total Cost':<12}")
    print("-" * 80)
    
    for name, gpus, price_per_gpu, hours in scenarios:
        total_cost = gpus * price_per_gpu * hours
        print(f"{name:<40} {gpus:<5} ${price_per_gpu:<7.2f} {hours:<8} ${total_cost:<11.2f}")
    
    print("\nNote: These are estimates. Actual costs may vary.")


def example_5_check_balance():
    """Example 5: Check account balance"""
    print("=" * 80)
    print("Example 5: Check Account Balance")
    print("=" * 80)
    
    if not os.environ.get('HYPERBOLIC_API_KEY'):
        print("\nSkipping: HYPERBOLIC_API_KEY not set")
        return
    
    try:
        client = HyperbolicClient()
        balance = client.get_credit_balance()
        
        print(f"\nAccount Balance: ${balance.get('balance', 0):.2f}")
        
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Hyperbolic Labs Deployment - Usage Examples")
    print("=" * 80)
    
    examples = [
        example_1_check_requirements,
        example_2_list_machines,
        example_3_find_best_machine,
        example_4_cost_estimation,
        example_5_check_balance,
    ]
    
    for example in examples:
        print()
        try:
            example()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user")
            break
        except Exception as e:
            print(f"\nError in example: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nTo deploy for real, use:")
    print("  python hyperbolic/deploy.py --model-size medium --auto-launch")


if __name__ == "__main__":
    main()
