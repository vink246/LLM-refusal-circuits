#!/usr/bin/env python3
"""
Analyze and compare circuits across categories.

This script compares discovered circuits, computes similarities, assesses modularity,
and generates visualizations (heatmaps, network diagrams).
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuits.similarity import CircuitSimilarity, equalize_circuits
from src.analysis.modularity import assess_modularity
from src.visualization.heatmaps import (
    create_similarity_heatmap,
    create_safe_toxic_heatmaps,
    create_cross_refusal_heatmap
)
from src.utils.config import load_config
from src.utils.logging import setup_logging
import json


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare circuits across categories"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to circuit discovery configuration YAML file"
    )
    parser.add_argument(
        "--circuits-dir",
        type=str,
        help="Directory containing circuit files (overrides config)"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paths if provided
    if args.circuits_dir:
        config['output']['circuits_dir'] = args.circuits_dir
    
    # Load circuits
    circuits_dir = Path(config['output'].get('circuits_dir', 'results/circuits'))
    
    print("=" * 80)
    print("Analyzing Circuits")
    print("=" * 80)
    
    # Load circuits (TODO: implement circuit loading)
    # For now, circuits should be loaded from JSON files
    print("\nLoading circuits...")
    circuits = {}  # Will be populated when circuit loading is implemented
    
    # TODO: Load circuits from JSON files in circuits_dir
    # circuits = load_circuits_from_json(circuits_dir)
    
    if not circuits:
        print("Error: No circuits found. Please discover circuits first using discover_circuits.py")
        print(f"  Expected in: {circuits_dir}")
        sys.exit(1)
    
    # Equalize circuits for fair comparison (Stage 2: Circuit Comparison)
    print("\n" + "=" * 80)
    print("Equalizing Circuits for Fair Comparison")
    print("=" * 80)
    
    equalize = config.get('analysis', {}).get('equalize_circuits', True)
    if equalize:
        circuits = equalize_circuits(circuits)
    
    # Compute similarities
    print("\nComputing circuit similarities...")
    similarity_analyzer = CircuitSimilarity(config)
    similarities = similarity_analyzer.compute_all_similarities(circuits, equalize=equalize)
    
    # Assess modularity
    print("\nAssessing modularity...")
    modularity_results = assess_modularity(similarities, config)
    
    # Generate visualizations
    if not args.no_visualizations:
        print("\nGenerating visualizations...")
        viz_dir = Path(config['output'].get('visualizations_dir', 'results/visualizations'))
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create similarity heatmaps
        similarity_analyzer.create_visualizations(circuits, similarities, viz_dir)
    
    # Save analysis results
    output_dir = Path(config['output'].get('reports_dir', 'results/reports'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    similarity_analyzer.save_results(similarities, modularity_results, output_dir)
    
    print("\n✓ Circuit analysis complete!")
    print(f"  Results saved to: {output_dir}")
    if not args.no_visualizations:
        print(f"  Visualizations saved to: {config['output'].get('visualizations_dir', 'results/visualizations')}")


if __name__ == "__main__":
    main()

