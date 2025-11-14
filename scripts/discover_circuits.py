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
        "--sae-config",
        type=str,
        help="Path to SAE configuration YAML file (if different from circuit config)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configurations
    circuit_config = load_config(args.config)
    if args.sae_config:
        sae_config = load_config(args.sae_config)
        # Merge SAE config if needed
        circuit_config['sae'] = sae_config
    
    # Load SAEs
    sae_manager = SAEManager(circuit_config.get('sae', {}))
    saes = sae_manager.load_all()
    
    if not saes:
        print("Error: No trained SAEs found. Please train SAEs first using train_saes.py")
        sys.exit(1)
    
    # Initialize circuit discoverer
    discoverer = CircuitDiscoverer(circuit_config, saes)
    
    # Discover circuits
    print("=" * 80)
    print("Discovering Circuits")
    print("=" * 80)
    
    if circuit_config.get('discovery', {}).get('separate_safe_toxic', False):
        print("Mode: Separate safe and toxic circuits")
    else:
        print("Mode: Combined circuits")
    
    circuits = discoverer.discover_all()
    
    # Save circuits
    discoverer.save_circuits()
    
    print("\n✓ Circuit discovery complete!")
    print(f"  Circuits saved to: {circuit_config.get('output', {}).get('circuits_dir', 'results/circuits')}")


if __name__ == "__main__":
    main()

