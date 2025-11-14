"""
Circuit similarity computation with equalization for fair comparison.
"""

from typing import Dict, List, Tuple, Optional
from .circuit import SparseFeatureCircuit
import numpy as np


def equalize_circuits(
    circuits: Dict[str, SparseFeatureCircuit],
    equalize_to: Optional[int] = None
) -> Dict[str, SparseFeatureCircuit]:
    """
    Equalize circuits to ensure fair comparison across categories.
    
    Uses top-N features from each circuit where N is the minimum circuit size.
    This ensures fair comparison when categories have different sample sizes.
    
    Args:
        circuits: Dictionary mapping category to circuit
        equalize_to: Number of features to use (None = use minimum)
        
    Returns:
        Dictionary of equalized circuits
    """
    if not circuits:
        return circuits
    
    # Find minimum circuit size
    circuit_sizes = {cat: len(circuit.nodes) for cat, circuit in circuits.items()}
    
    if equalize_to is None:
        equalize_to = min(circuit_sizes.values())
    
    print(f"\nEqualizing circuits to {equalize_to} features (minimum: {min(circuit_sizes.values())})")
    print(f"Circuit sizes before equalization:")
    for cat, size in sorted(circuit_sizes.items()):
        print(f"  {cat}: {size} features")
    
    # Create equalized circuits
    equalized_circuits = {}
    
    for category, circuit in circuits.items():
        # Get top-N nodes by importance
        top_nodes = circuit.get_top_nodes(equalize_to)
        
        # Create new circuit with only top nodes
        equalized_circuit = SparseFeatureCircuit()
        
        for node_id in top_nodes:
            if node_id in circuit.nodes:
                node_data = circuit.nodes[node_id]
                importance = circuit.node_importances.get(node_id, 0.0)
                
                equalized_circuit.add_node(
                    feature_id=node_data['feature_id'],
                    layer=node_data['layer'],
                    position=node_data['position'],
                    importance=importance,
                    feature_type=node_data.get('type', 'sae_feature')
                )
        
        # Add edges between top nodes
        for (src, tgt), edge_data in circuit.edges.items():
            if src in top_nodes and tgt in top_nodes:
                importance = circuit.edge_importances.get((src, tgt), 0.0)
                equalized_circuit.add_edge(src, tgt, importance)
        
        equalized_circuits[category] = equalized_circuit
    
    print(f"\nEqualized circuit sizes:")
    for cat in sorted(equalized_circuits.keys()):
        print(f"  {cat}: {len(equalized_circuits[cat].nodes)} features")
    
    return equalized_circuits


def compute_circuit_similarity(
    circuit1: SparseFeatureCircuit,
    circuit2: SparseFeatureCircuit
) -> float:
    """
    Compute similarity between two circuits using multiple metrics.
    
    Args:
        circuit1: First circuit
        circuit2: Second circuit
        
    Returns:
        Combined similarity score (0-1)
    """
    # Get top nodes (use top 100 for comparison)
    nodes1 = set(circuit1.get_top_nodes(100))
    nodes2 = set(circuit2.get_top_nodes(100))
    
    # Jaccard similarity on nodes
    intersection = nodes1.intersection(nodes2)
    union = nodes1.union(nodes2)
    
    if len(union) == 0:
        node_similarity = 0.0
    else:
        node_similarity = len(intersection) / len(union)
    
    # Edge similarity
    edges1 = set(circuit1.get_top_edges(100))
    edges2 = set(circuit2.get_top_edges(100))
    
    edge_intersection = edges1.intersection(edges2)
    edge_union = edges1.union(edges2)
    
    if len(edge_union) == 0:
        edge_similarity = 0.0
    else:
        edge_similarity = len(edge_intersection) / len(edge_union)
    
    # Importance correlation (for overlapping nodes)
    common_nodes = nodes1.intersection(nodes2)
    if len(common_nodes) > 0:
        importances1 = [circuit1.node_importances.get(node, 0.0) for node in common_nodes]
        importances2 = [circuit2.node_importances.get(node, 0.0) for node in common_nodes]
        
        if np.std(importances1) > 0 and np.std(importances2) > 0:
            importance_corr = np.corrcoef(importances1, importances2)[0, 1]
            if np.isnan(importance_corr):
                importance_corr = 0.0
        else:
            importance_corr = 0.0
    else:
        importance_corr = 0.0
    
    # Weighted combination: 40% nodes, 30% edges, 30% importance correlation
    combined_similarity = 0.4 * node_similarity + 0.3 * edge_similarity + 0.3 * abs(importance_corr)
    
    return combined_similarity


class CircuitSimilarity:
    """Computes and manages circuit similarities with equalization."""
    
    def __init__(self, config: Dict):
        """Initialize similarity analyzer."""
        self.config = config
        self.equalize_for_comparison = config.get('analysis', {}).get('equalize_circuits', True)
    
    def compute_all_similarities(
        self,
        circuits: Dict[str, SparseFeatureCircuit],
        equalize: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute similarities between all circuit pairs.
        
        Args:
            circuits: Dictionary mapping category to circuit
            equalize: Whether to equalize circuits before comparison
            
        Returns:
            Dictionary of similarity matrices
        """
        if equalize and self.equalize_for_comparison:
            circuits = equalize_circuits(circuits)
        
        categories = sorted(list(circuits.keys()))
        similarity_matrix = {}
        
        for cat1 in categories:
            similarity_matrix[cat1] = {}
            for cat2 in categories:
                if cat1 == cat2:
                    similarity_matrix[cat1][cat2] = 1.0
                else:
                    similarity = compute_circuit_similarity(
                        circuits[cat1], circuits[cat2]
                    )
                    similarity_matrix[cat1][cat2] = similarity
        
        return similarity_matrix

