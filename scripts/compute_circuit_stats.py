"""
Compute statistical tests for discovered TOXIC circuits.

This script loads all saved toxic circuits for a specified model and runs statistical 
significance tests comparing them to random baselines. It does NOT compute 
faithfulness/completeness metrics.

IMPORTANT: This script ONLY processes toxic circuits (not safe or combined circuits).

Circuit Save Locations:
  - Default directory: results/circuits/
  - Configurable via: configs/circuits/*.yaml -> circuits_dir
  
Circuit File Naming Pattern (TOXIC ONLY):
  - {model_name}_{category}_toxic_circuit.json
  
  Example: meta-llama-Llama-2-7b-chat-hf_violence_toxic_circuit.json

Circuits are saved by: scripts/discover_circuits.py
  - Uses CircuitDiscoverer.save_circuits() from src/circuits/discovery.py
  - Requires: separate_safe_toxic=True in config
  - Save location: config.get('output', {}).get('circuits_dir', 'results/circuits')

Output:
  - Single JSON file: {model_name}_toxic_circuits_stats.json
  - Contains all statistical test results for all toxic circuits
  - Saved to: results/evaluation/ (or --output_dir)
"""

import torch
import sys
import os
from pathlib import Path
import json
import numpy as np
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.circuits.circuit import SparseFeatureCircuit
from src.circuits.similarity import compute_random_similarity_distribution, compute_significance


def find_toxic_circuit_file(circuit_dir: Path, safe_model_name: str, category: str) -> Path:
    """
    Find toxic circuit file for a category.
    
    Looks for: {model}_{category}_toxic_circuit.json
    """
    circuit_path = circuit_dir / f"{safe_model_name}_{category}_toxic_circuit.json"
    if circuit_path.exists():
        return circuit_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute statistical tests for discovered circuits (no faithfulness/completeness)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-2-7b-chat-hf", 
        help="Model name"
    )
    parser.add_argument(
        "--circuit_dir", 
        type=str, 
        default="results/circuits", 
        help="Directory containing circuits (default: results/circuits)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/evaluation_stats", 
        help="Output directory for stats JSON files (default: results/evaluation)"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Specific categories to process (default: all found in circuit_dir)"
    )
    args = parser.parse_args()

    # Setup directories
    circuit_dir = Path(args.circuit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not circuit_dir.exists():
        print(f"Error: Circuit directory not found: {circuit_dir}")
        print(f"  Circuits should be saved to: {circuit_dir}")
        print(f"  Expected file pattern: {{model_name}}_{{category}}_toxic_circuit.json")
        return
    
    safe_model_name = args.model.replace('/', '-').replace(' ', '_')
    
    # Find all TOXIC circuit files for this model
    if args.categories:
        categories = args.categories
        print(f"Processing specified categories: {categories}")
    else:
        # Auto-detect categories from toxic circuit files only
        toxic_circuit_files = list(circuit_dir.glob(f"{safe_model_name}_*_toxic_circuit.json"))
        
        # Extract categories from filenames
        categories = set()
        for file in toxic_circuit_files:
            # Pattern: {model}_{category}_toxic_circuit.json
            # Remove model name prefix and "_toxic_circuit" suffix
            stem = file.stem
            if stem.startswith(f"{safe_model_name}_") and stem.endswith("_toxic_circuit"):
                category = stem[len(f"{safe_model_name}_"):-len("_toxic_circuit")]
                if category:
                    categories.add(category)
        
        categories = sorted(list(categories))
        print(f"Auto-detected {len(categories)} toxic circuit files: {categories}")
    
    if not categories:
        print(f"Error: No toxic circuit files found in {circuit_dir}")
        print(f"  Looked for files matching: {safe_model_name}_*_toxic_circuit.json")
        print(f"  Make sure circuits were discovered with separate_safe_toxic=True")
        return
    
    # Hardcoded values
    # Using lower threshold (0.001) to retain more nodes and increase statistical power
    # Increased permutations (5000) for better p-value resolution and statistical power
    target_threshold = 0.001
    sae_hidden_dim = 8192  # Default for LLaMA/Mistral configs
    n_permutations = 5000  # Increased from 1000 for better resolution and power
    
    print(f"\n{'='*80}")
    print(f"Computing Statistical Tests for TOXIC Circuits")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Circuit Directory: {circuit_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"SAE Hidden Dimension: {sae_hidden_dim} (hardcoded)")
    print(f"Target Threshold: {target_threshold} (hardcoded)")
    print(f"Permutations: {n_permutations} (hardcoded)")
    print(f"Categories to process: {len(categories)}")
    print(f"{'='*80}\n")
    
    all_stats = {}
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {category}")
        print(f"{'='*50}")
        
        # Find toxic circuit file
        circuit_path = find_toxic_circuit_file(circuit_dir, safe_model_name, category)
        
        if circuit_path is None or not circuit_path.exists():
            print(f"Warning: Toxic circuit file not found for category '{category}'")
            print(f"  Expected: {safe_model_name}_{category}_toxic_circuit.json")
            continue
        
        print(f"Loading circuit: {circuit_path}")
        try:
            full_circuit = SparseFeatureCircuit.load_from_json(str(circuit_path))
        except Exception as e:
            print(f"Error loading circuit: {e}")
            continue
        
        # Get circuit info
        importances = [node_data['importance'] for node_data in full_circuit.nodes.values()]
        if not importances:
            print(f"Warning: Circuit has no nodes. Skipping category '{category}'.")
            continue
        
        importances_sorted = sorted(importances, reverse=True)
        max_importance = max(importances)
        min_importance = min(importances)
        median_importance = importances_sorted[len(importances_sorted) // 2]
        p95_importance = importances_sorted[int(0.05 * len(importances_sorted))]
        p99_importance = importances_sorted[int(0.01 * len(importances_sorted))]
        
        print(f"Circuit info: {len(importances)} nodes")
        print(f"  Importance range: [{min_importance:.6f}, {max_importance:.6f}]")
        print(f"  Median: {median_importance:.6f}, 95th percentile: {p95_importance:.6f}, 99th percentile: {p99_importance:.6f}")
        
        # Filter circuit by threshold (hardcoded to match evaluate_circuits.py)
        rep_circuit = SparseFeatureCircuit()
        nodes_above_threshold = 0
        for node_id, node_data in full_circuit.nodes.items():
            if node_data['importance'] >= target_threshold:
                rep_circuit.add_node(
                    feature_id=node_data['feature_id'],
                    layer=node_data['layer'],
                    position=node_data['position'],
                    importance=node_data['importance']
                )
                nodes_above_threshold += 1
        
        # Add edges for filtered nodes
        for (src, tgt), edge_data in full_circuit.edges.items():
            if src in rep_circuit.nodes and tgt in rep_circuit.nodes:
                rep_circuit.add_edge(src, tgt, edge_data['importance'])
        
        if len(rep_circuit.nodes) == 0:
            print(f"Warning: Circuit is empty at threshold {target_threshold}.")
            print(f"  Consider using a lower threshold. Suggested: {p99_importance:.6f} (99th percentile) or {median_importance:.6f} (median)")
            continue
        
        print(f"Filtered circuit: {len(rep_circuit.nodes)} nodes (threshold >= {target_threshold})")
        print(f"  Retained {100.0 * len(rep_circuit.nodes) / len(importances):.1f}% of nodes")
        
        # Warn if too many nodes filtered
        if len(rep_circuit.nodes) < len(importances) * 0.1:
            print(f"  ⚠️  WARNING: Threshold may be too high - only {len(rep_circuit.nodes)}/{len(importances)} nodes retained")
            print(f"     Consider lowering threshold to {p95_importance:.6f} (95th percentile) or {median_importance:.6f} (median)")
        
        # Statistical Tests
        print("Running Statistical Tests...")
        print(f"  Comparing circuit to random baseline...")
        
        # Create a single random circuit for comparison
        import random
        random_comp_circuit = SparseFeatureCircuit()
        nodes = rep_circuit.get_top_nodes(100)
        layer_counts = {}
        for node_id in nodes:
            layer = rep_circuit.nodes[node_id]['layer']
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        
        for layer, count in layer_counts.items():
            # Sample from actual SAE feature space (0 to sae_hidden_dim - 1)
            random_features = random.sample(range(sae_hidden_dim), count)
            for feat_id in random_features:
                random_comp_circuit.add_node(str(feat_id), layer, 0, 1.0)
        
        from src.circuits.similarity import compute_circuit_similarity
        observed_sim = compute_circuit_similarity(rep_circuit, random_comp_circuit)
        print(f"  Observed Similarity (Circuit vs Random): {observed_sim:.6f}")
        
        print(f"  Computing null distribution ({n_permutations} permutations)...")
        print(f"    This may take a few minutes - progress bar will show status...")
        print(f"    Null distribution: similarity(discovered_circuit, random_circuit) for many random circuits")
        print(f"    This tests: Is discovered circuit different from what we'd expect vs random?")
        
        # CRITICAL FIX: Null distribution should compare discovered_circuit to many random circuits
        # NOT random_circuit1 vs random_circuit2
        # This properly tests: "Is discovered_circuit significantly different from random?"
        null_dist = []
        for _ in tqdm(range(n_permutations), desc="Computing null distribution", leave=False):
            # Create a new random circuit with same structure as rep_circuit
            random_circuit_null = SparseFeatureCircuit()
            for layer, count in layer_counts.items():
                if count > 0:
                    random_features = random.sample(range(sae_hidden_dim), count)
                    for feat_id in random_features:
                        random_circuit_null.add_node(str(feat_id), layer, 0, 1.0)
            
            # Compare discovered circuit to this random circuit
            # This is the correct null: "What similarity do we get when comparing discovered to random?"
            sim = compute_circuit_similarity(rep_circuit, random_circuit_null)
            null_dist.append(sim)
        
        print(f"    Completed {len(null_dist)} permutations")
        null_mean = np.mean(null_dist)
        null_std = np.std(null_dist)
        null_min = np.min(null_dist)
        null_max = np.max(null_dist)
        print(f"    Null distribution: mean={null_mean:.6f}, std={null_std:.6f}, range=[{null_min:.6f}, {null_max:.6f}]")
        
        # Diagnostic: Check if observed is in the null distribution
        if abs(observed_sim - null_mean) < null_std:
            print(f"    ⚠️  WARNING: Observed ({observed_sim:.6f}) is within 1 std of null mean ({null_mean:.6f})")
            print(f"       This suggests the circuit may not be significantly different from random")
        
        # Use two-sided test: Is discovered circuit significantly DIFFERENT from random?
        stats = compute_significance(observed_sim, null_dist, two_sided=True)
        
        print("\nStatistical Results:")
        print(f"  Observed Similarity: {stats['observed']:.6f}")
        print(f"  Null Distribution Mean: {stats['random_mean']:.6f}")
        print(f"  Null Distribution Std: {stats['random_std']:.6f}")
        print(f"  Z-Score: {stats['z_score']:.4f}")
        print(f"  P-Value (Two-Sided): {stats['p_value']:.6f}")
        if stats.get('p_value_normal') is not None:
            print(f"  P-Value (Normal Approx): {stats['p_value_normal']:.6f}")
        print(f"  95% CI (Null Distribution): [{stats['ci_lower']:.6f}, {stats['ci_upper']:.6f}]")
        
        if stats['p_value'] < 0.05:
            print(f"  ✓ Significant (p < 0.05): Circuit is significantly different from random")
        else:
            print(f"  ✗ Not Significant (p ≥ 0.05): Cannot reject null hypothesis")
        
        # Add metadata
        stats['category'] = category
        stats['model_name'] = args.model
        stats['circuit_file'] = str(circuit_path)
        stats['target_threshold'] = target_threshold
        stats['n_nodes_in_circuit'] = len(rep_circuit.nodes)
        stats['n_nodes_in_full_circuit'] = len(full_circuit.nodes)
        
        all_stats[category] = stats
    
    # Save all stats to a single JSON file
    if all_stats:
        output_file = output_dir / f"{safe_model_name}_toxic_circuits_stats.json"
        
        # Create output structure with metadata and all stats
        output_data = {
            'model_name': args.model,
            'circuit_dir': str(circuit_dir),
            'sae_hidden_dim': sae_hidden_dim,
            'target_threshold': target_threshold,
            'n_permutations': n_permutations,
            'n_categories': len(all_stats),
            'categories': sorted(list(all_stats.keys())),
            'stats_by_category': all_stats
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"All stats saved to: {output_file}")
        print(f"{'='*80}")
        
        # Print summary table
        print("\nSummary Table:")
        print(f"{'Category':<20} {'Observed':<12} {'Null Mean':<12} {'P-Value':<12} {'Significant':<12}")
        print("-" * 70)
        for category in sorted(all_stats.keys()):
            stats = all_stats[category]
            sig = "✓ Yes" if stats['p_value'] < 0.05 else "✗ No"
            print(f"{category:<20} {stats['observed']:<12.6f} {stats['random_mean']:<12.6f} "
                  f"{stats['p_value']:<12.6f} {sig:<12}")
    else:
        print("\nWarning: No stats computed. No valid toxic circuits found.")
        return
    
    print(f"\n✓ Statistical tests complete!")
    print(f"  Processed {len(all_stats)} toxic circuits")
    print(f"  All results saved to: {output_file}")


if __name__ == "__main__":
    main()

