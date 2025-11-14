#!/usr/bin/env python3
"""
Discover sparse feature circuits for each refusal category.

This script uses trained SAEs to discover circuits that explain refusal behavior.
Can discover separate circuits for safe and toxic samples.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.discovery import CircuitDiscoverer
from src.sae.sae_manager import SAEManager
from src.utils.config import load_config
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Discover sparse feature circuits for each refusal category"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to circuit discovery configuration YAML file"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="results",
        help="Result directory containing activations and refusal labels"
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
    circuit_config = load_config(args.config)
    
    # Get model and categories
    model_name = args.model_name or circuit_config.get('model', {}).get('name')
    if not model_name:
        raise ValueError("Model name must be provided via --model-name or config")
    
    categories = args.categories or circuit_config.get('discovery', {}).get('categories', [])
    if not categories:
        # Try to infer from activation files
        from glob import glob
        activation_dir = Path(args.result_dir) / "activations"
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        pattern = str(activation_dir / f"{safe_model_name}_*_activations.pt")
        files = glob(pattern)
        categories = list(set([Path(f).stem.split('_')[-2] for f in files]))
        print(f"Inferred categories from activation files: {categories}")
    
    # Initialize SAE manager
    sae_dir = circuit_config.get('sae', {}).get('save_dir') or circuit_config.get('output', {}).get('save_dir', 'results/saes')
    sae_manager = SAEManager(sae_dir=sae_dir)
    
    # Initialize circuit discoverer
    discoverer = CircuitDiscoverer(circuit_config, sae_manager)
    
    # Discover circuits
    print("=" * 80)
    print("Discovering Circuits")
    print("=" * 80)
    
    if circuit_config.get('discovery', {}).get('separate_safe_toxic', False):
        print("Mode: Separate safe and toxic circuits")
    else:
        print("Mode: Combined circuits")
    
    discoverer.discover_all(args.result_dir, model_name, categories)
    
    print("\n✓ Circuit discovery complete!")
    print(f"  Circuits saved to: {circuit_config.get('output', {}).get('circuits_dir', 'results/circuits')}")


if __name__ == "__main__":
    main()

