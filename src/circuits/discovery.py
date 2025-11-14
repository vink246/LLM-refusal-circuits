"""
Circuit discovery using trained sparse autoencoders.

Discovers sparse feature circuits that explain refusal behavior by identifying
important SAE features and their connections.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
from scipy import stats

from .circuit import SparseFeatureCircuit, CircuitConfig
from ..sae.sae_manager import SAEManager


class CircuitDiscoverer:
    """Discovers sparse feature circuits from activations and SAEs"""
    
    def __init__(self, config: Dict[str, Any], sae_manager: SAEManager):
        """
        Initialize circuit discoverer.
        
        Args:
            config: Configuration dictionary
            sae_manager: SAE manager with trained SAEs
        """
        self.config = config
        self.sae_manager = sae_manager
        self.circuit_config = CircuitConfig(
            node_threshold=config.get('circuit', {}).get('node_threshold', 0.1),
            edge_threshold=config.get('circuit', {}).get('edge_threshold', 0.01),
            aggregation_method=config.get('circuit', {}).get('aggregation_method', 'none'),
            attribution_method=config.get('circuit', {}).get('attribution_method', 'stats')
        )
        self.separate_safe_toxic = config.get('discovery', {}).get('separate_safe_toxic', False)
        self.circuits_dir = Path(config.get('output', {}).get('circuits_dir', 'results/circuits'))
        self.circuits_dir.mkdir(parents=True, exist_ok=True)
        
        # Store discovered circuits
        self.circuits = {}  # category -> circuit or (safe_circuit, toxic_circuit)
        self.safe_circuits = {}  # category -> safe_circuit
        self.toxic_circuits = {}  # category -> toxic_circuit
    
    def discover_circuit_for_category(
        self,
        category: str,
        activation_file: str,
        refusal_labels: List[bool],
        model_name: str
    ) -> SparseFeatureCircuit:
        """
        Discover circuit for a single category.
        
        Args:
            category: Category name
            activation_file: Path to activation file
            refusal_labels: List of refusal labels (True = refusal/toxic, False = safe)
            model_name: Model name
            
        Returns:
            Discovered circuit (or tuple if separate_safe_toxic)
        """
        print(f"\nDiscovering circuit for category: {category}")
        
        # Load activations
        print(f"  Loading activations from: {activation_file}")
        raw_activations = torch.load(activation_file, map_location='cpu')
        
        # Get layers
        layers = list(raw_activations.keys())
        print(f"  Layers: {layers}")
        
        # Load SAEs for this model
        saes = self.sae_manager.load_saes_for_model(model_name, layers)
        if not saes:
            raise ValueError(f"No SAEs found for model {model_name} and layers {layers}")
        
        # Encode activations with SAEs
        print("  Encoding activations with SAEs...")
        encoded_activations = self.sae_manager.encode_activations(raw_activations, saes)
        
        if self.separate_safe_toxic:
            # Split into safe and toxic
            refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.bool)
            safe_indices = ~refusal_labels_tensor
            toxic_indices = refusal_labels_tensor
            
            safe_encoded = {}
            toxic_encoded = {}
            for layer, features in encoded_activations.items():
                if features.dim() == 3:
                    # (batch, seq, hidden)
                    safe_encoded[layer] = features[safe_indices]
                    toxic_encoded[layer] = features[toxic_indices]
                else:
                    # (batch, hidden)
                    safe_encoded[layer] = features[safe_indices]
                    toxic_encoded[layer] = features[toxic_indices]
            
            safe_labels = [False] * safe_indices.sum().item()
            toxic_labels = [True] * toxic_indices.sum().item()
            
            print(f"  Safe samples: {len(safe_labels)}, Toxic samples: {len(toxic_labels)}")
            
            # Discover circuits separately
            safe_circuit = self._discover_single_circuit(
                safe_encoded, safe_labels, model_name, category, "safe"
            )
            toxic_circuit = self._discover_single_circuit(
                toxic_encoded, toxic_labels, model_name, category, "toxic"
            )
            
            self.safe_circuits[category] = safe_circuit
            self.toxic_circuits[category] = toxic_circuit
            
            return (safe_circuit, toxic_circuit)
        else:
            # Discover single circuit
            circuit = self._discover_single_circuit(
                encoded_activations, refusal_labels, model_name, category, "combined"
            )
            self.circuits[category] = circuit
            return circuit
    
    def _discover_single_circuit(
        self,
        encoded_activations: Dict[str, torch.Tensor],
        refusal_labels: List[bool],
        model_name: str,
        category: str,
        circuit_type: str
    ) -> SparseFeatureCircuit:
        """
        Discover a single circuit from encoded activations.
        
        Uses statistical correlation to identify important features.
        """
        circuit = SparseFeatureCircuit()
        refusal_labels_tensor = torch.tensor(refusal_labels, dtype=torch.float32)
        
        print(f"    Discovering {circuit_type} circuit...")
        
        # Aggregate activations across sequence dimension if needed
        aggregated_features = {}
        for layer, features in encoded_activations.items():
            if features.dim() == 3:
                # (batch, seq, hidden) -> aggregate across sequence
                if self.circuit_config.aggregation_method == "mean":
                    aggregated_features[layer] = features.mean(dim=1)  # (batch, hidden)
                elif self.circuit_config.aggregation_method == "max":
                    aggregated_features[layer] = features.max(dim=1)[0]  # (batch, hidden)
                else:
                    # Use last token
                    aggregated_features[layer] = features[:, -1, :]  # (batch, hidden)
            else:
                aggregated_features[layer] = features  # (batch, hidden)
        
        # Discover important features (nodes)
        all_nodes = []
        for layer, features in aggregated_features.items():
            # features: (batch, hidden_dim)
            batch_size, hidden_dim = features.shape
            
            # Compute correlation between each feature and refusal labels
            features_np = features.cpu().numpy()
            labels_np = refusal_labels_tensor.cpu().numpy()
            
            correlations = []
            for feat_idx in range(hidden_dim):
                feat_values = features_np[:, feat_idx]
                # Compute Pearson correlation
                if np.std(feat_values) > 0 and np.std(labels_np) > 0:
                    corr, p_value = stats.pearsonr(feat_values, labels_np)
                    correlations.append((feat_idx, abs(corr), corr, p_value))
                else:
                    correlations.append((feat_idx, 0.0, 0.0, 1.0))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Add top features as nodes
            layer_idx = int(layer.split('_')[1]) if '_' in layer else 0
            for feat_idx, abs_corr, corr, p_value in correlations:
                if abs_corr >= self.circuit_config.node_threshold:
                    # Extract position (use 0 for aggregated, or could use sequence position)
                    position = 0
                    node_id = circuit.add_node(
                        feature_id=str(feat_idx),
                        layer=layer_idx,
                        position=position,
                        importance=abs_corr,
                        feature_type="sae_feature"
                    )
                    all_nodes.append((node_id, layer_idx, abs_corr))
        
        print(f"      Found {len(all_nodes)} important features")
        
        # Discover edges (connections between features across layers)
        if len(aggregated_features) > 1:
            layers_list = sorted(aggregated_features.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            
            for i in range(len(layers_list) - 1):
                layer1 = layers_list[i]
                layer2 = layers_list[i + 1]
                
                features1 = aggregated_features[layer1]
                features2 = aggregated_features[layer2]
                
                # Find nodes in these layers
                layer1_nodes = [n for n in all_nodes if n[1] == int(layer1.split('_')[1] if '_' in layer1 else 0)]
                layer2_nodes = [n for n in all_nodes if n[1] == int(layer2.split('_')[1] if '_' in layer2 else 0)]
                
                # Compute correlations between features in adjacent layers
                for node1_id, _, _ in layer1_nodes:
                    feat1_idx = int(circuit.nodes[node1_id]['feature_id'])
                    feat1_values = features1[:, feat1_idx].cpu().numpy()
                    
                    for node2_id, _, _ in layer2_nodes:
                        feat2_idx = int(circuit.nodes[node2_id]['feature_id'])
                        feat2_values = features2[:, feat2_idx].cpu().numpy()
                        
                        if np.std(feat1_values) > 0 and np.std(feat2_values) > 0:
                            corr, _ = stats.pearsonr(feat1_values, feat2_values)
                            edge_importance = abs(corr)
                            
                            if edge_importance >= self.circuit_config.edge_threshold:
                                circuit.add_edge(node1_id, node2_id, edge_importance)
        
        print(f"      Found {len(circuit.edges)} edges")
        
        return circuit
    
    def discover_all(
        self,
        result_dir: str,
        model_name: str,
        categories: List[str]
    ):
        """
        Discover circuits for all categories.
        
        Args:
            result_dir: Result directory containing activations and refusal labels
            model_name: Model name
            categories: List of categories to process
        """
        result_path = Path(result_dir)
        activation_dir = result_path / "activations"
        refusal_dir = result_path / "refusal_labels"
        
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        
        print("=" * 80)
        print("Discovering Circuits for All Categories")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Categories: {categories}")
        print(f"Separate safe/toxic: {self.separate_safe_toxic}")
        
        for category in categories:
            activation_file = activation_dir / f"{safe_model_name}_{category}_activations.pt"
            refusal_file = refusal_dir / f"{safe_model_name}_{category}_refusal.json"
            
            if not activation_file.exists():
                print(f"Warning: Activation file not found: {activation_file}")
                continue
            
            if not refusal_file.exists():
                print(f"Warning: Refusal labels file not found: {refusal_file}")
                continue
            
            # Load refusal labels
            with open(refusal_file, 'r') as f:
                refusal_labels = json.load(f)
            
            # Discover circuit
            try:
                self.discover_circuit_for_category(
                    category, str(activation_file), refusal_labels, model_name
                )
            except Exception as e:
                print(f"Error discovering circuit for {category}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save circuits
        self.save_circuits(model_name)
    
    def save_circuits(self, model_name: str):
        """Save discovered circuits to JSON files"""
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        
        if self.separate_safe_toxic:
            # Save safe and toxic circuits separately
            for category, circuit in self.safe_circuits.items():
                circuit_file = self.circuits_dir / f"{safe_model_name}_{category}_safe_circuit.json"
                self._save_circuit_to_json(circuit, circuit_file)
                print(f"  Saved safe circuit: {circuit_file}")
            
            for category, circuit in self.toxic_circuits.items():
                circuit_file = self.circuits_dir / f"{safe_model_name}_{category}_toxic_circuit.json"
                self._save_circuit_to_json(circuit, circuit_file)
                print(f"  Saved toxic circuit: {circuit_file}")
        else:
            # Save combined circuits
            for category, circuit in self.circuits.items():
                circuit_file = self.circuits_dir / f"{safe_model_name}_{category}_circuit.json"
                self._save_circuit_to_json(circuit, circuit_file)
                print(f"  Saved circuit: {circuit_file}")
    
    def _save_circuit_to_json(self, circuit: SparseFeatureCircuit, filepath: Path):
        """Save a circuit to JSON file"""
        circuit_data = {
            'nodes': {
                node_id: {
                    'feature_id': node_data['feature_id'],
                    'layer': node_data['layer'],
                    'position': node_data['position'],
                    'importance': circuit.node_importances.get(node_id, 0.0),
                    'type': node_data.get('type', 'sae_feature')
                }
                for node_id, node_data in circuit.nodes.items()
            },
            'edges': {
                f"{src}_{tgt}": {
                    'source': src,
                    'target': tgt,
                    'importance': circuit.edge_importances.get((src, tgt), 0.0)
                }
                for (src, tgt), edge_data in circuit.edges.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(circuit_data, f, indent=2)
    
    @staticmethod
    def load_circuit_from_json(filepath: Path) -> SparseFeatureCircuit:
        """Load a circuit from JSON file"""
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

