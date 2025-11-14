#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAEs) on collected activations.

This script trains SAEs to decompose activations into sparse, interpretable features.
A single SAE is trained on all data (safe + toxic) to ensure feature space comparability.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sae.sae_manager import SAEManager
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoders on collected activations"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SAE configuration YAML file"
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        help="Directory containing activation files (overrides config)"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="Result directory for finding activation files (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paths if provided
    if args.activations_dir:
        config['data']['activations_dir'] = args.activations_dir
    if args.result_dir:
        config['data']['result_dir'] = args.result_dir
    
    # Initialize SAE manager
    sae_manager = SAEManager(config)
    
    # Train SAEs
    print("=" * 80)
    print("Training Sparse Autoencoders")
    print("=" * 80)
    print("NOTE: Training SINGLE SAE on ALL data (safe + toxic) for comparability")
    
    sae_manager.train_all()
    
    print("\n✓ SAE training complete!")
    print(f"  SAEs saved to: {sae_manager.config.get('output', {}).get('save_dir', 'results/saes')}")


if __name__ == "__main__":
    main()

