"""
Circuit data structures for representing sparse feature circuits.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class CircuitConfig:
    """Configuration for circuit discovery"""
    node_threshold: float = 0.1
    edge_threshold: float = 0.01
    aggregation_method: str = "none"  # "none", "mean", "max"
    attribution_method: str = "ig"  # "ig" (integrated gradients) or "atp" (attribution patching)
    n_integration_steps: int = 10
    top_n_per_layer: int = 100  # Top N features per layer when using magnitude-based discovery


class SparseFeatureCircuit:
    """Represents a discovered sparse feature circuit"""
    
    def __init__(self):
        self.nodes = {}  # feature -> node data
        self.edges = {}  # (source, target) -> edge data
        self.node_importances = {}
        self.edge_importances = {}
        
    def add_node(
        self,
        feature_id: str,
        layer: int,
        position: int,
        importance: float,
        feature_type: str = "sae_feature"
    ):
        """Add a node to the circuit"""
        node_id = f"{feature_type}_{layer}_{position}_{feature_id}"
        self.nodes[node_id] = {
            'feature_id': feature_id,
            'layer': layer,
            'position': position,
            'importance': importance,
            'type': feature_type
        }
        self.node_importances[node_id] = importance
        return node_id
    
    def add_edge(self, source: str, target: str, importance: float):
        """Add an edge to the circuit"""
        edge_key = (source, target)
        self.edges[edge_key] = {
            'source': source,
            'target': target,
            'importance': importance
        }
        self.edge_importances[edge_key] = importance
        
    def get_top_nodes(self, n: int = 50) -> List[str]:
        """Get top n most important nodes"""
        return sorted(
            self.node_importances.keys(),
            key=lambda x: abs(self.node_importances[x]),
            reverse=True
        )[:n]
    
    def get_top_edges(self, n: int = 50) -> List[Tuple[str, str]]:
        """Get top n most important edges"""
        return sorted(
            self.edge_importances.keys(),
            key=lambda x: abs(self.edge_importances[x]),
            reverse=True
        )[:n]

