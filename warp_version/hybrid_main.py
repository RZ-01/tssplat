#!/usr/bin/env python3
"""
Main entry point for Hybrid TetSplatting pipeline.
Combines original TetSplatting with Warp components for optimal performance.
"""

import argparse
import os
import sys
import torch
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import hybrid components
from hybrid_trainer import train_hybrid_tetsplat, train_mesh_fitting_hybrid
from hybrid_config import (
    create_hybrid_config,
    create_gso_hybrid_config,
    create_mesh_fitting_hybrid_config,
    save_hybrid_config,
    load_hybrid_config,
    get_example_config
)


def setup_environment():
    """Setup environment for hybrid training"""
    # Set CUDA device
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")
    
    # Initialize Warp
    try:
        import warp as wp
        wp.init()
        print("Warp initialized successfully")
    except ImportError:
        print("Warning: Warp not available, falling back to original components")
    
    # Check for required dependencies
    try:
        import nvdiffrast.torch as dr
        print("nvdiffrast available")
    except ImportError:
        print("Warning: nvdiffrast not available")
    
    try:
        import pypgo
        print("pypgo available")
    except ImportError:
        print("Warning: pypgo not available")


def train_gso_hybrid(args):
    """Train on GSO dataset with hybrid pipeline"""
    print("Starting GSO hybrid training...")
    
    # Create configuration
    config = create_gso_hybrid_config(
        data_path=args.data_path,
        output_path=args.output_path,
        use_warp_rasterization=args.use_warp,
        total_iterations=args.iterations
    )
    
    # Save configuration
    config_path = os.path.join(args.output_path, "config.yaml")
    save_hybrid_config(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Start training
    try:
        results = train_hybrid_tetsplat(config)
        print("Training completed successfully!")
        print(f"Best loss: {results['best_loss']:.6f}")
        print(f"Best iteration: {results['best_iter']}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        if args.fallback:
            print("Attempting fallback to original pipeline...")
            config.hybrid.use_warp_rasterization = False
            results = train_hybrid_tetsplat(config)
            print("Fallback training completed!")


def train_mesh_fitting_hybrid(args):
    """Train mesh fitting with hybrid pipeline"""
    print("Starting mesh fitting hybrid training...")
    
    # Create configuration
    config = create_mesh_fitting_hybrid_config(
        target_mesh_path=args.target_mesh,
        output_path=args.output_path,
        use_warp_rasterization=args.use_warp,
        total_iterations=args.iterations
    )
    
    # Save configuration
    config_path = os.path.join(args.output_path, "config.yaml")
    save_hybrid_config(config, config_path)
    print(f"Configuration saved to: {config_path}")
    
    # Start training
    try:
        train_mesh_fitting_hybrid(config)
        print("Mesh fitting completed successfully!")
        
    except Exception as e:
        print(f"Mesh fitting failed: {e}")
        if args.fallback:
            print("Attempting fallback to original pipeline...")
            config.hybrid.use_warp_rasterization = False
            train_mesh_fitting_hybrid(config)
            print("Fallback mesh fitting completed!")


def train_from_config(args):
    """Train from existing configuration file"""
    print(f"Loading configuration from: {args.config}")
    
    config = load_hybrid_config(args.config)
    
    # Override settings if specified
    if args.use_warp is not None:
        config.hybrid.use_warp_rasterization = args.use_warp
    if args.output_path:
        config.output_path = args.output_path
    
    print(f"Using Warp rasterization: {config.hybrid.use_warp_rasterization}")
    print(f"Output path: {config.output_path}")
    
    # Determine training type
    if config.get("target_mesh_path"):
        # Mesh fitting
        train_mesh_fitting_hybrid(config)
    else:
        # Regular training
        train_hybrid_tetsplat(config)


def create_example_config(args):
    """Create example configuration file"""
    config_name = args.example
    example_config = get_example_config(config_name)
    
    print(f"Creating example configuration: {config_name}")
    print(f"Description: {example_config['description']}")
    
    # Create base configuration
    base_config = {
        "expr_name": f"{config_name}_example",
        "fitting_stage": "geometry",
        "total_num_iter": 10000,
        "verbose": True
    }
    
    # Create hybrid configuration
    from hybrid_config import create_hybrid_config_from_dict
    config = create_hybrid_config_from_dict(base_config, example_config)
    
    # Save configuration
    output_path = args.output_path or f"config_{config_name}.yaml"
    save_hybrid_config(config, output_path)
    print(f"Example configuration saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hybrid TetSplatting Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # GSO training command
    gso_parser = subparsers.add_parser("gso", help="Train on GSO dataset")
    gso_parser.add_argument("--data_path", required=True, help="Path to GSO dataset")
    gso_parser.add_argument("--output_path", default="results/gso_hybrid", help="Output path")
    gso_parser.add_argument("--use_warp", action="store_true", default=True, help="Use Warp rasterization")
    gso_parser.add_argument("--iterations", type=int, default=10000, help="Number of iterations")
    gso_parser.add_argument("--fallback", action="store_true", help="Fallback to original if Warp fails")
    
    # Mesh fitting command
    mesh_parser = subparsers.add_parser("mesh", help="Mesh fitting")
    mesh_parser.add_argument("--target_mesh", required=True, help="Path to target mesh")
    mesh_parser.add_argument("--output_path", default="results/mesh_fitting_hybrid", help="Output path")
    mesh_parser.add_argument("--use_warp", action="store_true", default=True, help="Use Warp rasterization")
    mesh_parser.add_argument("--iterations", type=int, default=15000, help="Number of iterations")
    mesh_parser.add_argument("--fallback", action="store_true", help="Fallback to original if Warp fails")
    
    # Config-based training command
    config_parser = subparsers.add_parser("config", help="Train from configuration file")
    config_parser.add_argument("--config", required=True, help="Path to configuration file")
    config_parser.add_argument("--use_warp", type=bool, help="Override Warp usage")
    config_parser.add_argument("--output_path", help="Override output path")
    
    # Example config creation command
    example_parser = subparsers.add_parser("example", help="Create example configuration")
    example_parser.add_argument("--example", required=True, 
                               choices=["gso_hybrid", "mesh_fitting_hybrid", "full_warp", "original_only"],
                               help="Example configuration type")
    example_parser.add_argument("--output_path", help="Output path for configuration file")
    
    # Global arguments
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup environment
    setup_environment()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    torch.cuda.set_device(0) if torch.cuda.is_available() else None
    
    # Execute command
    try:
        if args.command == "gso":
            train_gso_hybrid(args)
        elif args.command == "mesh":
            train_mesh_fitting_hybrid(args)
        elif args.command == "config":
            train_from_config(args)
        elif args.command == "example":
            create_example_config(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
