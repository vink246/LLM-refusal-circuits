#!/usr/bin/env python3
"""
Collect model activations by running inference on OR-Bench dataset.

This script loads models, runs inference, and collects activations from
specified layers for circuit analysis.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.inference import InferencePipeline
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Collect model activations by running inference on OR-Bench"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        help="Path to data configuration YAML file (if different from model config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configurations
    model_config = load_config(args.config)
    if args.data_config:
        data_config = load_config(args.data_config)
        # Merge data config into model config
        model_config['data'] = data_config.get('dataset', {})
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(model_config)
    
    # Run inference and collect activations
    print("=" * 80)
    print("Collecting Model Activations")
    print("=" * 80)
    
    pipeline.run()
    
    print("\n✓ Activation collection complete!")
    print(f"  Activations saved to: {pipeline.config.get('output', {}).get('activations_dir', 'results/activations')}")


if __name__ == "__main__":
    main()

