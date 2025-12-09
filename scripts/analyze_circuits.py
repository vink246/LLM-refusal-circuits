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

from src.circuits.similarity import CircuitSimilarity, equalize_circuits, compute_circuit_similarity
from src.circuits.discovery import CircuitDiscoverer
from src.visualization.heatmaps import (
    create_similarity_heatmap,
    create_safe_toxic_heatmaps,
    create_cross_refusal_heatmap
)
from src.utils.config import load_config
from src.utils.logging import setup_logging
import json
import numpy as np


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
    
    # Get model name for loading circuits
    model_name = config.get('model', {}).get('name')
    if not model_name:
        # Try to infer from circuit files
        circuit_files = list(circuits_dir.glob("*_circuit.json")) + list(circuits_dir.glob("*_safe_circuit.json"))
        if circuit_files:
            # Extract model name from first file
            filename = circuit_files[0].stem
            if '_safe_circuit' in filename or '_toxic_circuit' in filename:
                model_name = filename.rsplit('_', 2)[0]
            else:
                model_name = filename.rsplit('_', 1)[0]
            model_name = model_name.replace('-', '/')
        else:
            raise ValueError("Model name must be provided in config or circuit files must exist")
    
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    separate_safe_toxic = config.get('discovery', {}).get('separate_safe_toxic', False)
    
    # Load circuits
    print("\nLoading circuits...")
    circuits = {}
    safe_circuits = {}
    toxic_circuits = {}
    
    if separate_safe_toxic:
        # Load separate safe and toxic circuits
        safe_files = list(circuits_dir.glob(f"{safe_model_name}_*_safe_circuit.json"))
        toxic_files = list(circuits_dir.glob(f"{safe_model_name}_*_toxic_circuit.json"))
        
        for file in safe_files:
            category = file.stem.replace(f"{safe_model_name}_", "").replace("_safe_circuit", "")
            safe_circuits[category] = CircuitDiscoverer.load_circuit_from_json(file)
            print(f"  Loaded safe circuit for {category}")
        
        for file in toxic_files:
            category = file.stem.replace(f"{safe_model_name}_", "").replace("_toxic_circuit", "")
            toxic_circuits[category] = CircuitDiscoverer.load_circuit_from_json(file)
            print(f"  Loaded toxic circuit for {category}")
        
        if not safe_circuits and not toxic_circuits:
            print("Error: No circuits found. Please discover circuits first using discover_circuits.py")
            print(f"  Expected in: {circuits_dir}")
            print(f"  Pattern: {safe_model_name}_*_safe_circuit.json and {safe_model_name}_*_toxic_circuit.json")
            sys.exit(1)
    else:
        # Load combined circuits
        circuit_files = list(circuits_dir.glob(f"{safe_model_name}_*_circuit.json"))
        # Exclude safe/toxic specific files
        circuit_files = [f for f in circuit_files if '_safe_' not in f.name and '_toxic_' not in f.name]
        
        for file in circuit_files:
            category = file.stem.replace(f"{safe_model_name}_", "").replace("_circuit", "")
            circuits[category] = CircuitDiscoverer.load_circuit_from_json(file)
            print(f"  Loaded circuit for {category}")
        
        if not circuits:
            print("Error: No circuits found. Please discover circuits first using discover_circuits.py")
            print(f"  Expected in: {circuits_dir}")
            print(f"  Pattern: {safe_model_name}_*_circuit.json")
            sys.exit(1)
    
    # Equalize circuits for fair comparison (Stage 3: Circuit Comparison)
    print("\n" + "=" * 80)
    print("Equalizing Circuits for Fair Comparison")
    print("=" * 80)
    
    equalize = config.get('analysis', {}).get('equalize_circuits', True)
    
    if separate_safe_toxic:
        if equalize:
            safe_circuits = equalize_circuits(safe_circuits)
            toxic_circuits = equalize_circuits(toxic_circuits)
        
        # Generate visualizations
        if not args.no_visualizations:
            print("\nGenerating visualizations...")
            viz_dir = Path(config['output'].get('visualizations_dir', 'results/visualizations')).joinpath(config['model'].get('name', 'model').replace('/', '-').replace(' ', '_'))
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe and toxic heatmaps (category-to-category)
            create_safe_toxic_heatmaps(safe_circuits, toxic_circuits, viz_dir)
            
            # Create cross-refusal heatmap (safe vs toxic)
            create_cross_refusal_heatmap(safe_circuits, toxic_circuits, viz_dir)
        
        # Compute similarities for analysis
        print("\nComputing circuit similarities...")
        similarity_analyzer = CircuitSimilarity(config)
        
        # Safe circuit similarities
        safe_similarities = similarity_analyzer.compute_all_similarities(safe_circuits, equalize=False)
        
        # Toxic circuit similarities
        toxic_similarities = similarity_analyzer.compute_all_similarities(toxic_circuits, equalize=False)
        
        # Cross-refusal similarities
        cross_similarities = {}
        for safe_cat in safe_circuits.keys():
            cross_similarities[safe_cat] = {}
            for toxic_cat in toxic_circuits.keys():
                cross_similarities[safe_cat][toxic_cat] = compute_circuit_similarity(
                    safe_circuits[safe_cat], toxic_circuits[toxic_cat]
                )
        
        # Assess modularity
        print("\nAssessing modularity...")
        # Modularity: High similarity = monolithic, Low similarity = modular
        safe_avg_similarity = np.mean([v for row in safe_similarities.values() for v in row.values() if v < 1.0])
        toxic_avg_similarity = np.mean([v for row in toxic_similarities.values() for v in row.values() if v < 1.0])
        cross_avg_similarity = np.mean([v for row in cross_similarities.values() for v in row.values()])
        
        modularity_results = {
            'safe_avg_similarity': float(safe_avg_similarity),
            'toxic_avg_similarity': float(toxic_avg_similarity),
            'cross_avg_similarity': float(cross_avg_similarity),
            'safe_similarities': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in safe_similarities.items()},
            'toxic_similarities': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in toxic_similarities.items()},
            'cross_similarities': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in cross_similarities.items()}
        }
        
        # Save analysis results
        output_dir = Path(config['output'].get('reports_dir', 'results/reports')).joinpath(config['model'].get('name', 'model').replace('/', '-').replace(' ', '_'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'modularity_analysis.json', 'w') as f:
            json.dump(modularity_results, f, indent=2)
        
        print("\n✓ Circuit analysis complete!")
        print(f"\nModularity Assessment:")
        print(f"  Safe circuit avg similarity: {safe_avg_similarity:.3f}")
        print(f"  Toxic circuit avg similarity: {toxic_avg_similarity:.3f}")
        print(f"  Cross-refusal avg similarity: {cross_avg_similarity:.3f}")
        print(f"\n  Interpretation:")
        print(f"    - High similarity (>0.5) = Monolithic (shared circuits)")
        print(f"    - Low similarity (<0.5) = Modular (category-specific circuits)")
        print(f"  Results saved to: {output_dir}")
        if not args.no_visualizations:
            print(f"  Visualizations saved to: {config['output'].get('visualizations_dir', 'results/visualizations')+' '+config['model'].get('name', 'model').replace('/', '-').replace(' ', '_')}")
    else:
        # Combined circuits
        if equalize:
            circuits = equalize_circuits(circuits)
        
        # Compute similarities
        print("\nComputing circuit similarities...")
        similarity_analyzer = CircuitSimilarity(config)
        similarities = similarity_analyzer.compute_all_similarities(circuits, equalize=False)
        
        # Assess modularity
        print("\nAssessing modularity...")
        avg_similarity = np.mean([v for row in similarities.values() for v in row.values() if v < 1.0])
        
        modularity_results = {
            'avg_similarity': float(avg_similarity),
            'similarities': {k: {k2: float(v2) for k2, v2 in v.items()} for k, v in similarities.items()}
        }
        
        # Generate visualizations
        if not args.no_visualizations:
            print("\nGenerating visualizations...")
            viz_dir = Path(config['output'].get('visualizations_dir', 'results/visualizations')).joinpath(config['model'].get('name', 'model').replace('/', '-').replace(' ', '_'))
            viz_dir.mkdir(parents=True, exist_ok=True)

            create_similarity_heatmap(circuits, viz_dir, "Circuit Similarity Across Categories")
        
        # Save analysis results
        output_dir = Path(config['output'].get('reports_dir', 'results/reports')).joinpath(config['model'].get('name', 'model').replace('/', '-').replace(' ', '_'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'modularity_analysis.json', 'w') as f:
            json.dump(modularity_results, f, indent=2)
        
        print("\n✓ Circuit analysis complete!")
        print(f"\nModularity Assessment:")
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Interpretation:")
        print(f"    - High similarity (>0.5) = Monolithic (shared circuits)")
        print(f"    - Low similarity (<0.5) = Modular (category-specific circuits)")
        print(f"  Results saved to: {output_dir}")
        if not args.no_visualizations:
            print(f"  Visualizations saved to: {config['output'].get('visualizations_dir', 'results/visualizations')+' '+config['model'].get('name', 'model').replace('/', '-').replace(' ', '_')}")


if __name__ == "__main__":
    main()

