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

    @staticmethod
    def load_from_json(filepath: str) -> 'SparseFeatureCircuit':
        """Load a circuit from JSON file"""
        import json
        from pathlib import Path
        
        with open(filepath, 'r') as f:
            circuit_data = json.load(f)
        
        circuit = SparseFeatureCircuit()
        
        # Load nodes
        for node_id, node_info in circuit_data['nodes'].items():
            circuit.add_node(
                feature_id=node_info['feature_id'],
                layer=node_info['layer'],
                position=node_info['position'],
                importance=node_info['importance'],
                feature_type=node_info.get('type', 'sae_feature')
            )
        
        # Load edges
        for edge_key, edge_info in circuit_data['edges'].items():
            # Edge key format: "source_target" or we can use the stored source/target
            if 'source' in edge_info and 'target' in edge_info:
                src = edge_info['source']
                tgt = edge_info['target']
            else:
                # Fallback: try to split the key
                parts = edge_key.split('_', 1)
                if len(parts) == 2:
                    src, tgt = parts
                else:
                    continue
            
            # Find matching node IDs (exact match or check if they exist)
            if src in circuit.nodes and tgt in circuit.nodes:
                circuit.add_edge(src, tgt, edge_info['importance'])
        
        return circuit

