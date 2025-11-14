#!/usr/bin/env python3
"""
Collect model activations by running inference on OR-Bench dataset.

Uses two-stage approach:
- Stage 1: Collect balanced data (all toxic samples, matched with safe)
- Stage 2: SAE training will use single SAE on all data for comparability
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import DataProcessor
from src.models.inference_pipeline import InferencePipeline
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
        # Merge data config
        if 'data' not in model_config:
            model_config['data'] = {}
        model_config['data'].update(data_config.get('dataset', {}))
    
    # Get data configuration
    data_config_section = model_config.get('data', {})
    dataset_dir = data_config_section.get('path', 'data/raw/or-bench')
    categories = data_config_section.get('categories', [])  # Empty = all
    strategy = data_config_section.get('category_balance_strategy', 'use_all')
    
    # Initialize data processor
    processor = DataProcessor(dataset_dir)
    
    # Prepare balanced dataset (Stage 1)
    print("=" * 80)
    print("Stage 1: Preparing Balanced Dataset")
    print("=" * 80)
    
    data_by_category, toxic_counts = processor.prepare_balanced_dataset(
        categories=categories if categories else None,
        strategy=strategy,
        shuffle=True
    )
    
    # Compute category weights for SAE training (Stage 2)
    category_weights = processor.compute_category_weights(toxic_counts)
    
    print("\nCategory weights for SAE training:")
    for cat, weight in sorted(category_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {weight:.4f}")
    
    # Save category weights for SAE training
    output_dir = Path(model_config.get('output', {}).get('result_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weights_file = output_dir / "category_weights.json"
    import json
    with open(weights_file, 'w') as f:
        json.dump({
            'toxic_counts': toxic_counts,
            'category_weights': category_weights
        }, f, indent=2)
    print(f"\nSaved category weights to: {weights_file}")
    
    # Stage 2: Run inference to collect activations
    print("\n" + "=" * 80)
    print("Stage 2: Collecting Activations via Inference")
    print("=" * 80)
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(model_config)
    
    # Run inference for all categories
    pipeline.run(data_by_category)
    
    # Cleanup
    pipeline.cleanup()
    
    print("\n" + "=" * 80)
    print("✓ Activation Collection Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Activations: {output_dir / 'activations'}")
    print(f"  - Refusal labels: {output_dir / 'refusal_labels'}")
    print(f"  - Model outputs: {output_dir / 'model_outputs'}")
    print(f"  - Evaluation results: {output_dir / 'evaluation_results'}")
    print(f"  - Category weights: {weights_file}")
    print("\nNext step: Train SAEs using train_saes.py")


if __name__ == "__main__":
    main()

