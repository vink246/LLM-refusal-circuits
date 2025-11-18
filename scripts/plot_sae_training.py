#!/usr/bin/env python3
"""
Plot SAE training metrics from saved training history files.

This script can be used to regenerate plots from existing training history
JSON files without retraining the SAEs.
"""

import argparse
import sys
from pathlib import Path
from glob import glob
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.training_plots import (
    plot_sae_training_history,
    plot_reconstruction_loss_only,
    plot_all_layers_comparison
)


def plot_single_history(history_file: Path, output_dir: Path):
    """Plot a single training history file."""
    # Extract model name and layer from filename
    # Expected format: {model_name}_{layer}_training_history.json
    stem = history_file.stem
    parts = stem.split('_')
    
    if len(parts) < 3:
        print(f"Warning: Could not parse filename: {history_file.name}")
        return
    
    # Find the layer part (before 'training')
    try:
        training_idx = parts.index('training')
        layer = '_'.join(parts[training_idx-1:training_idx])
        model_name = '_'.join(parts[:training_idx-1])
    except ValueError:
        # Fallback: assume last 2 parts before 'training_history' are layer
        layer = parts[-3]
        model_name = '_'.join(parts[:-3])
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print(f"\nPlotting: {model_name} - {layer}")
    
    # Generate plots
    plot_sae_training_history(history, model_name, layer, output_dir)
    plot_reconstruction_loss_only(history, model_name, layer, output_dir)


def plot_model_comparison(sae_dir: Path, model_name: str, output_dir: Path):
    """Plot comparison across all layers for a model."""
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    model_sae_dir = sae_dir / safe_model_name
    
    if not model_sae_dir.exists():
        print(f"Error: Model SAE directory not found: {model_sae_dir}")
        return
    
    # Find all training history files
    history_files = list(model_sae_dir.glob("*_training_history.json"))
    
    if not history_files:
        print(f"No training history files found in {model_sae_dir}")
        return
    
    # Load all histories
    histories = {}
    for history_file in history_files:
        stem = history_file.stem
        # Extract layer name
        layer = stem.replace('_training_history', '')
        
        # Remove model name prefix if present
        if layer.startswith(safe_model_name + '_'):
            layer = layer[len(safe_model_name)+1:]
        
        with open(history_file, 'r') as f:
            histories[layer] = json.load(f)
    
    print(f"\nGenerating comparison plots for {model_name} ({len(histories)} layers)")
    
    # Generate comparison plots
    plot_all_layers_comparison(histories, model_name, output_dir, 'reconstruction_loss')
    plot_all_layers_comparison(histories, model_name, output_dir, 'total_loss')
    plot_all_layers_comparison(histories, model_name, output_dir, 'l0_norm')


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAE training metrics from saved history files"
    )
    parser.add_argument(
        "--sae-dir",
        type=str,
        default="results/saes",
        help="Directory containing SAE training results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots (default: {sae_dir}/training_plots)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Plot only a specific model (optional)"
    )
    parser.add_argument(
        "--history-file",
        type=str,
        help="Plot a specific training history JSON file (optional)"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate layer comparison plots"
    )
    
    args = parser.parse_args()
    
    sae_dir = Path(args.sae_dir)
    
    if not sae_dir.exists():
        print(f"Error: SAE directory not found: {sae_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sae_dir / "training_plots"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SAE Training History Plotting")
    print("=" * 80)
    print(f"SAE Directory: {sae_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Plot specific history file
    if args.history_file:
        history_file = Path(args.history_file)
        if not history_file.exists():
            print(f"Error: History file not found: {history_file}")
            sys.exit(1)
        plot_single_history(history_file, output_dir)
        print("\n✓ Plotting complete!")
        return
    
    # Plot specific model
    if args.model_name:
        safe_model_name = args.model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = sae_dir / safe_model_name
        
        if not model_sae_dir.exists():
            print(f"Error: Model directory not found: {model_sae_dir}")
            sys.exit(1)
        
        # Find all history files for this model
        history_files = list(model_sae_dir.glob("*_training_history.json"))
        
        if not history_files:
            print(f"No training history files found for {args.model_name}")
            sys.exit(1)
        
        print(f"\nFound {len(history_files)} training history files")
        
        # Plot each history file
        for history_file in history_files:
            plot_single_history(history_file, output_dir)
        
        # Generate comparison plots if requested
        if args.comparison:
            plot_model_comparison(sae_dir, args.model_name, output_dir)
        
        print("\n✓ Plotting complete!")
        return
    
    # Plot all models
    print("\nPlotting all available models...")
    
    model_dirs = [d for d in sae_dir.iterdir() if d.is_dir() and d.name != "training_plots"]
    
    if not model_dirs:
        print("No model directories found")
        sys.exit(1)
    
    for model_dir in model_dirs:
        history_files = list(model_dir.glob("*_training_history.json"))
        
        if not history_files:
            continue
        
        print(f"\n{model_dir.name}: {len(history_files)} layers")
        
        for history_file in history_files:
            plot_single_history(history_file, output_dir)
        
        if args.comparison:
            model_name = model_dir.name.replace('-', '/')
            plot_model_comparison(sae_dir, model_name, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ All plotting complete!")
    print(f"  Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

