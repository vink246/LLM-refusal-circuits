#!/usr/bin/env python3
"""
Train Sparse Autoencoders (SAEs) on collected activations.

CRITICAL: Trains a SINGLE SAE on ALL data (safe + toxic, all categories)
to ensure feature space comparability across categories and refusal types.

Uses two-stage approach:
- Stage 1: Balanced data collection (already done)
- Stage 2: Single SAE training with category weighting (this script)
"""

import argparse
import sys
import json
from pathlib import Path
from glob import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sae.sae_manager import SAEManager
from src.utils.config import load_config
from src.utils.logging import setup_logging


def get_activation_files(result_dir: str, model_name: str, categories: list) -> list:
    """Get all activation files for a model and categories."""
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    activation_dir = Path(result_dir) / "activations"
    
    activation_files = []
    for category in categories:
        file_path = activation_dir / f"{safe_model_name}_{category}_activations.pt"
        if file_path.exists():
            activation_files.append(str(file_path))
    
    return activation_files


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
        "--result-dir",
        type=str,
        default="results",
        help="Result directory containing activations"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name (overrides config)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs='+',
        help="Categories to use (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model and categories
    model_name = args.model_name or config.get('model', {}).get('name')
    if not model_name:
        raise ValueError("Model name must be provided via --model-name or config")
    
    categories = args.categories or config.get('data', {}).get('categories', [])
    if not categories:
        # Try to infer from activation files
        activation_dir = Path(args.result_dir) / "activations"
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        pattern = str(activation_dir / f"{safe_model_name}_*_activations.pt")
        files = glob(pattern)
        categories = list(set([Path(f).stem.split('_')[-2] for f in files]))
        print(f"Inferred categories from activation files: {categories}")
    
    # Load category weights if available
    weights_file = Path(args.result_dir) / "category_weights.json"
    category_weights = None
    if weights_file.exists():
        with open(weights_file, 'r') as f:
            weights_data = json.load(f)
            category_weights = weights_data.get('category_weights')
            print(f"\nLoaded category weights from: {weights_file}")
    
    # Get activation files
    activation_files = get_activation_files(args.result_dir, model_name, categories)
    
    if not activation_files:
        print(f"Error: No activation files found for {model_name}")
        print(f"  Looked in: {Path(args.result_dir) / 'activations'}")
        print(f"  Categories: {categories}")
        sys.exit(1)
    
    print(f"\nFound {len(activation_files)} activation files")
    
    # Get layers from config
    layers = config.get('model', {}).get('activation_layers', [])
    if not layers:
        # Try to infer from first activation file
        first_activation = torch.load(activation_files[0], map_location='cpu')
        layers = list(first_activation.keys())
        print(f"Inferred layers from activation file: {layers}")
    
    # Initialize SAE manager
    sae_config = config.get('sae', {})
    training_config = config.get('training', {})
    
    sae_manager = SAEManager(
        sae_dir=config.get('output', {}).get('save_dir', 'results/saes')
    )
    
    # Train SAEs
    print("=" * 80)
    print("Stage 2: Training Sparse Autoencoders")
    print("=" * 80)
    print("CRITICAL: Training SINGLE SAE on ALL data")
    print("  - All categories (safe + toxic)")
    print("  - Ensures feature space comparability")
    print("  - Safe and toxic circuits use same features")
    
    sae_manager.train_saes_for_model(
        model_name=model_name,
        activation_files=activation_files,
        layers=layers,
        sae_hidden_dim=sae_config.get('hidden_dim', 8192),
        max_samples=training_config.get('max_samples', 100000),
        batch_size=training_config.get('batch_size', 512),
        epochs=training_config.get('epochs', 100),
        sparsity_coeff=sae_config.get('sparsity_coeff', 0.01),
        balance_safe_toxic=training_config.get('balance_safe_toxic', True),
        result_dir=args.result_dir,
        category_weights=category_weights if training_config.get('use_category_weights', True) else None
    )
    
    print("\n✓ SAE training complete!")
    print(f"  SAEs saved to: {config.get('output', {}).get('save_dir', 'results/saes')}")
    print("\n✓ Feature space comparability guaranteed:")
    print("  - Single SAE used for all categories and refusal types")
    print("  - Safe and toxic circuits are directly comparable")


if __name__ == "__main__":
    import torch
    main()

