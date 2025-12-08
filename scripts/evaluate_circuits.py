
import torch
import sys
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_wrapper import load_model_with_hooks
from src.circuits.evaluation import CircuitEvaluator
from src.circuits.circuit import SparseFeatureCircuit
from src.sae.sae_manager import SAEManager

def main():
    parser = argparse.ArgumentParser(description="Evaluate circuit faithfulness and completeness")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Model name")
    parser.add_argument("--circuit_dir", type=str, default="results/circuits", help="Directory containing circuits")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Output directory for plots and results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print(f"Loading model: {args.model}")
    if args.cache_dir:
        print(f"Using cache directory: {args.cache_dir}")
        
    model_wrapper = load_model_with_hooks(args.model, [], device=args.device, cache_dir=args.cache_dir)
    sae_manager = SAEManager()
    evaluator = CircuitEvaluator(model_wrapper, sae_manager, {})
    
    # Load Data
    data_file = Path("data/raw/or-bench/or-bench-toxic.csv")
    if not data_file.exists():
        print(f"Error: Data file {data_file} not found.")
        return
        
    import pandas as pd
    df = pd.read_csv(data_file)
    categories = df['category'].unique()
    print(f"Found categories: {categories}")
    
    safe_model_name = args.model.replace('/', '-').replace(' ', '_')
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Processing category: {category}")
        print(f"{'='*50}")
        
        # Filter data
        df_cat = df[df['category'] == category]
        if len(df_cat) == 0:
             print(f"Warning: No samples found for category '{category}'. Skipping.")
             continue

        prompts = df_cat['prompt'].tolist()[:10]
        target_outputs = ["I cannot"] * len(prompts)
        print(f"Loaded {len(prompts)} samples for category '{category}'")
        
        # Calculate Mean Activations
        print("Calculating mean activations...")
        layers = [f"residuals_{i}" for i in range(21, 32)]
        model_wrapper.setup_activation_hooks(layers)
        _, activations = model_wrapper.run_with_activations(prompts)
        mean_activations = evaluator.get_mean_ablation_values(activations)
        
        # Load Circuit
        # Try with _toxic suffix first, then without
        circuit_path = Path(args.circuit_dir) / f"{safe_model_name}_{category}_toxic_circuit.json"
        if not circuit_path.exists():
            circuit_path = Path(args.circuit_dir) / f"{safe_model_name}_{category}_circuit.json"
            
        if not circuit_path.exists():
            print(f"Warning: Circuit file not found for category '{category}' (checked {circuit_path}). Skipping.")
            continue

        print(f"Loading circuit: {circuit_path}")
        full_circuit = SparseFeatureCircuit.load_from_json(circuit_path)
        
        # Get importance distribution to set meaningful thresholds
        importances = [node_data['importance'] for node_data in full_circuit.nodes.values()]
        if not importances:
            print(f"Warning: Circuit has no nodes. Skipping category '{category}'.")
            continue
            
        importances_sorted = sorted(importances, reverse=True)
        max_importance = max(importances)
        min_importance = min(importances)
        median_importance = importances_sorted[len(importances_sorted) // 2]
        
        print(f"Circuit importance stats: min={min_importance:.6f}, max={max_importance:.6f}, median={median_importance:.6f}, total_nodes={len(importances)}")
        
        # Use percentiles to ensure we get different numbers of nodes
        # Calculate thresholds that correspond to different percentiles
        percentiles = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
        thresholds = []
        for p in percentiles:
            idx = int((100 - p) / 100.0 * len(importances_sorted))
            idx = min(idx, len(importances_sorted) - 1)
            thresholds.append(importances_sorted[idx])
        
        # Also add some absolute thresholds as fallback (scaled to actual range)
        if max_importance > 0:
            absolute_thresholds = [max_importance * f for f in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]]
            thresholds.extend(absolute_thresholds)
        
        # Remove duplicates and sort
        thresholds = sorted(set(thresholds), reverse=True)
        # Keep at most 15 thresholds to avoid too many evaluations
        if len(thresholds) > 15:
            step = len(thresholds) // 15
            thresholds = thresholds[::step]
        
        print(f"Using {len(thresholds)} thresholds: {[f'{t:.6f}' for t in thresholds[:5]]}...{[f'{t:.6f}' for t in thresholds[-5:]]}")
        
        results = {
            "thresholds": thresholds,
            "faithfulness": [],
            "completeness": [],
            "n_nodes": []
        }
        
        print("Evaluating across thresholds...")
        seen_node_counts = set()
        for threshold in tqdm(thresholds, desc=f"Thresholds ({category})"):
            filtered_circuit = SparseFeatureCircuit()
            for node_id, node_data in full_circuit.nodes.items():
                if node_data['importance'] >= threshold:
                    filtered_circuit.add_node(
                        feature_id=node_data['feature_id'],
                        layer=node_data['layer'],
                        position=node_data['position'],
                        importance=node_data['importance']
                    )
            
            for (src, tgt), edge_data in full_circuit.edges.items():
                if src in filtered_circuit.nodes and tgt in filtered_circuit.nodes:
                    filtered_circuit.add_edge(src, tgt, edge_data['importance'])
                    
            n_nodes = len(filtered_circuit.nodes)
            results["n_nodes"].append(n_nodes)
            seen_node_counts.add(n_nodes)
            
            if n_nodes == 0:
                results["faithfulness"].append(0.0)
                results["completeness"].append(0.0)
                continue
                
            metrics = evaluator.evaluate_circuit(filtered_circuit, prompts, target_outputs, mean_activations)
            results["faithfulness"].append(metrics["faithfulness"])
            results["completeness"].append(metrics["completeness"])
        
        print(f"Unique node counts across thresholds: {sorted(seen_node_counts)}")
            
        # Plotting - sort by number of nodes for proper curve
        print("Plotting results...")
        
        # Create tuples and sort by n_nodes
        plot_data = list(zip(results["n_nodes"], results["faithfulness"], results["completeness"], results["thresholds"]))
        plot_data.sort(key=lambda x: x[0])  # Sort by n_nodes
        
        n_nodes_sorted = [x[0] for x in plot_data]
        faithfulness_sorted = [x[1] for x in plot_data]
        completeness_sorted = [x[2] for x in plot_data]
        thresholds_sorted = [x[3] for x in plot_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(n_nodes_sorted, faithfulness_sorted, label="Faithfulness", marker='o', linestyle='-', linewidth=2, markersize=6)
        plt.plot(n_nodes_sorted, completeness_sorted, label="Completeness", marker='x', linestyle='-', linewidth=2, markersize=6)
        plt.xlabel("Number of Nodes", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title(f"Faithfulness & Completeness vs Circuit Size ({category})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = output_dir / f"{safe_model_name}_{category}_metrics.png"
        plt.savefig(plot_path)
        plt.close() # Close plot to free memory
        print(f"Saved plot to {plot_path}")
        
        # Save results (sorted by node count)
        results_sorted = {
            "thresholds": thresholds_sorted,
            "faithfulness": faithfulness_sorted,
            "completeness": completeness_sorted,
            "n_nodes": n_nodes_sorted,
            "original_results": results  # Keep original for reference
        }
        results_path = output_dir / f"{safe_model_name}_{category}_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results_sorted, f, indent=2)
        print(f"Saved results to {results_path}")

        # Statistical Tests
        print("Running Statistical Tests...")
        from src.circuits.similarity import compute_random_similarity_distribution, compute_significance
        
        target_threshold = 0.05
        rep_circuit = SparseFeatureCircuit()
        for node_id, node_data in full_circuit.nodes.items():
            if node_data['importance'] >= target_threshold:
                rep_circuit.add_node(
                    feature_id=node_data['feature_id'],
                    layer=node_data['layer'],
                    position=node_data['position'],
                    importance=node_data['importance']
                )
                
        if len(rep_circuit.nodes) > 0:
            print(f"Comparing circuit (threshold={target_threshold}) with random baseline...")
            random_comp_circuit = SparseFeatureCircuit()
            import random
            nodes = rep_circuit.get_top_nodes(100)
            layer_counts = {}
            for node_id in nodes:
                layer = rep_circuit.nodes[node_id]['layer']
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            
            for layer, count in layer_counts.items():
                random_features = random.sample(range(32768), count)
                for feat_id in random_features:
                    random_comp_circuit.add_node(str(feat_id), layer, 0, 1.0)
                    
            from src.circuits.similarity import compute_circuit_similarity
            observed_sim = compute_circuit_similarity(rep_circuit, random_comp_circuit)
            print(f"Observed Similarity (Circuit vs Random): {observed_sim:.4f}")
            
            print("Computing null distribution (100 permutations)...")
            null_dist = compute_random_similarity_distribution(rep_circuit, random_comp_circuit, n_permutations=100)
            
            stats = compute_significance(observed_sim, null_dist)
            print("Statistical Results:")
            print(f"  Mean (Random): {stats['random_mean']:.4f}")
            print(f"  Std (Random): {stats['random_std']:.4f}")
            print(f"  Z-Score: {stats['z_score']:.4f}")
            print(f"  P-Value: {stats['p_value']:.4f}")
            print(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
            
            stats_path = output_dir / f"{safe_model_name}_{category}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved stats to {stats_path}")
        else:
            print("Circuit is empty at threshold 0.05, skipping stats.")

if __name__ == "__main__":
    main()