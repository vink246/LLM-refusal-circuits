"""
SAE Manager for training and managing Sparse Autoencoders.

Ensures single SAE is trained on all data (safe + toxic, all categories)
to guarantee feature space comparability.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm

from .sae_model import SparseAutoencoder
from .sae_trainer import SAETrainer
from ..visualization.training_plots import plot_sae_training_history, plot_reconstruction_loss_only


class ActivationDataset(Dataset):
    """Dataset for training SAEs on collected activations."""
    
    def __init__(
        self,
        activation_files: List[str],
        layer: str,
        max_samples: int = 100000,
        refusal_label_files: Optional[List[str]] = None,
        balance_safe_toxic: bool = False,
        category_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize activation dataset.
        
        Args:
            activation_files: List of paths to activation files
            layer: Layer name to extract
            max_samples: Maximum number of samples to use
            refusal_label_files: Optional list of refusal label files
            balance_safe_toxic: If True, balance safe and toxic samples
            category_weights: Optional category weights for handling imbalance
        """
        self.activations = []
        self.refusal_labels = []
        self.category_labels = []
        self.weights = []
        
        print(f"Loading activations for layer {layer}...")
        
        for i, file_path in enumerate(tqdm(activation_files)):
            if not Path(file_path).exists():
                continue
            
            try:
                data = torch.load(file_path)
                if layer not in data:
                    continue
                
                layer_activations = data[layer].float()
                if layer_activations.dim() > 2:
                    layer_activations = layer_activations.view(layer_activations.size(0), -1)
                
                self.activations.append(layer_activations)
                
                # Load refusal labels if provided
                if refusal_label_files and i < len(refusal_label_files):
                    label_file = refusal_label_files[i]
                    if Path(label_file).exists():
                        with open(label_file, 'r') as f:
                            labels = json.load(f)
                        self.refusal_labels.extend(labels)
                    else:
                        self.refusal_labels.extend([False] * layer_activations.size(0))
                else:
                    self.refusal_labels.extend([False] * layer_activations.size(0))
                
                # Extract category from filename for weighting
                filename = Path(file_path).stem
                parts = filename.split('_')
                category = parts[-2] if len(parts) >= 2 else "unknown"
                self.category_labels.extend([category] * layer_activations.size(0))
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not self.activations:
            raise ValueError(f"No activations found for layer {layer}")
        
        self.activations = torch.cat(self.activations, dim=0)
        self.refusal_labels = torch.tensor(self.refusal_labels, dtype=torch.bool)
        
        # Balance safe and toxic if requested
        if balance_safe_toxic and len(self.refusal_labels) > 0:
            safe_indices = ~self.refusal_labels
            toxic_indices = self.refusal_labels
            
            num_safe = safe_indices.sum().item()
            num_toxic = toxic_indices.sum().item()
            
            if num_toxic > 0 and num_safe > 0:
                min_count = min(num_safe, num_toxic)
                
                safe_sample_idx = torch.where(safe_indices)[0][:min_count]
                toxic_sample_idx = torch.where(toxic_indices)[0][:min_count]
                
                balanced_indices = torch.cat([safe_sample_idx, toxic_sample_idx])
                self.activations = self.activations[balanced_indices]
                self.refusal_labels = self.refusal_labels[balanced_indices]
                self.category_labels = [self.category_labels[i] for i in balanced_indices.tolist()]
                
                print(f"Balanced dataset: {min_count} safe + {min_count} toxic = {len(self.activations)} total")
        
        # Apply category weights if provided
        if category_weights:
            self.weights = torch.ones(len(self.activations))
            for i, category in enumerate(self.category_labels):
                if category in category_weights:
                    self.weights[i] = category_weights[category]
        else:
            self.weights = torch.ones(len(self.activations))
        
        # Limit samples if needed
        if len(self.activations) > max_samples:
            indices = torch.randperm(len(self.activations))[:max_samples]
            self.activations = self.activations[indices]
            self.refusal_labels = self.refusal_labels[indices]
            self.weights = self.weights[indices]
            self.category_labels = [self.category_labels[i] for i in indices.tolist()]
        
        print(f"Loaded {len(self.activations)} activation samples")
        if len(self.refusal_labels) > 0:
            num_toxic = self.refusal_labels.sum().item()
            num_safe = len(self.refusal_labels) - num_toxic
            print(f"  Safe: {num_safe}, Toxic: {num_toxic}")
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


class SAEManager:
    """
    Manager for training and using Sparse Autoencoders.
    
    IMPORTANT: Trains a SINGLE SAE on ALL data (safe + toxic, all categories)
    to ensure feature space comparability across categories and refusal types.
    """
    
    def __init__(self, sae_dir: str = "results/saes"):
        """
        Initialize SAE manager.
        
        Args:
            sae_dir: Directory to save/load SAEs
        """
        self.sae_dir = Path(sae_dir)
        self.sae_dir.mkdir(parents=True, exist_ok=True)
        self.trained_saes = {}
    
    def train_saes_for_model(
        self,
        model_name: str,
        activation_files: List[str],
        layers: List[str],
        sae_hidden_dim: int = 8192,
        max_samples: int = 100000,
        batch_size: int = 512,
        epochs: int = 100,
        sparsity_coeff: float = 0.01,
        balance_safe_toxic: bool = False,
        result_dir: Optional[str] = None,
        category_weights: Optional[Dict[str, float]] = None
    ):
        """
        Train SAEs for all layers of a model.
        
        CRITICAL: This trains a SINGLE SAE on ALL data (both safe and toxic, all categories).
        This ensures that safe and toxic circuits use the same feature space,
        making them directly comparable.
        
        Args:
            model_name: Model name
            activation_files: List of activation file paths
            layers: List of layer names
            sae_hidden_dim: SAE hidden dimension
            max_samples: Maximum samples to use
            batch_size: Training batch size
            epochs: Training epochs
            sparsity_coeff: Sparsity coefficient
            balance_safe_toxic: If True, balance safe and toxic samples during training
            result_dir: Result directory to find refusal label files
            category_weights: Optional category weights for handling category imbalance
        """
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = self.sae_dir / safe_model_name
        model_sae_dir.mkdir(exist_ok=True)
        
        # Get refusal label files if balancing is requested
        refusal_label_files = None
        if balance_safe_toxic and result_dir:
            refusal_label_files = []
            refusal_dir = Path(result_dir) / "refusal_labels"
            for act_file in activation_files:
                filename = Path(act_file).stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    category = parts[-2]
                    label_file = refusal_dir / f"{safe_model_name}_{category}_refusal.json"
                    refusal_label_files.append(str(label_file) if label_file.exists() else None)
                else:
                    refusal_label_files.append(None)
        
        for layer in layers:
            print(f"\n=== Training SAE for {model_name} - {layer} ===")
            print("NOTE: Training SINGLE SAE on ALL data (safe + toxic, all categories)")
            print("      This ensures feature space comparability across categories and refusal types")
            
            # Check if SAE already exists
            sae_path = model_sae_dir / f"{layer}_sae.pt"
            if sae_path.exists():
                print(f"SAE already exists at {sae_path}, skipping...")
                continue
            
            try:
                # Create dataset for this layer
                dataset = ActivationDataset(
                    activation_files,
                    layer,
                    max_samples,
                    refusal_label_files=refusal_label_files,
                    balance_safe_toxic=balance_safe_toxic,
                    category_weights=category_weights
                )
                
                if len(dataset) == 0:
                    print(f"No activations found for layer {layer}, skipping...")
                    continue
                
                input_dim = dataset.activations.shape[1]
                
                # Create weighted sampler if category weights are provided
                if category_weights:
                    sampler = torch.utils.data.WeightedRandomSampler(
                        weights=dataset.weights,
                        num_samples=len(dataset),
                        replacement=True
                    )
                    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
                else:
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Train SAE
                trainer = SAETrainer(
                    input_dim=input_dim,
                    hidden_dim=sae_hidden_dim,
                    sparsity_coeff=sparsity_coeff,
                    k_percent=0.05,
                    lr=1e-3,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                
                history = trainer.train(dataloader, epochs)
                
                # Save SAE and training history
                trainer.save(sae_path)
                with open(model_sae_dir / f"{layer}_training_history.json", 'w') as f:
                    json.dump(history, f, indent=2)
                
                # Generate training plots
                print("\nGenerating training plots...")
                plots_dir = model_sae_dir / "training_plots"
                plots_dir.mkdir(exist_ok=True)
                
                try:
                    plot_sae_training_history(history, model_name, layer, plots_dir)
                    plot_reconstruction_loss_only(history, model_name, layer, plots_dir)
                except Exception as e:
                    print(f"  Warning: Could not generate plots: {e}")
                
                print(f"SAE trained and saved to {sae_path}")
                print("✓ This SAE will be used for BOTH safe and toxic circuit discovery")
                print("  (ensuring feature space comparability)")
                
            except Exception as e:
                print(f"Error training SAE for layer {layer}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def load_saes_for_model(self, model_name: str, layers: List[str]) -> Dict[str, SAETrainer]:
        """Load trained SAEs for a model."""
        safe_model_name = model_name.replace('/', '-').replace(' ', '_')
        model_sae_dir = self.sae_dir / safe_model_name
        
        saes = {}
        for layer in layers:
            sae_path = model_sae_dir / f"{layer}_sae.pt"
            if sae_path.exists():
                try:
                    saes[layer] = SAETrainer.load(sae_path)
                    print(f"Loaded SAE for layer {layer}")
                except Exception as e:
                    print(f"Error loading SAE for layer {layer}: {e}")
            else:
                print(f"SAE not found for layer {layer}: {sae_path}")
        
        return saes
    
    def encode_activations(
        self,
        activations: Dict[str, torch.Tensor],
        saes: Dict[str, SAETrainer]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode activations using trained SAEs.
        
        Uses the SAME SAE for all activations (safe and toxic) to ensure comparability.
        """
        encoded = {}
        for layer, activation in activations.items():
            if layer in saes:
                sae = saes[layer]
                
                sae_device = next(sae.sae.parameters()).device
                sae_dtype = next(sae.sae.parameters()).dtype
                
                activation = activation.to(device=sae_device, dtype=sae_dtype)
                
                original_shape = activation.shape
                if activation.dim() > 2:
                    activation = activation.view(activation.size(0), -1)
                
                with torch.no_grad():
                    features = sae.sae.encode(activation)
                
                if len(original_shape) == 3:
                    features = features.view(original_shape[0], original_shape[1], -1)
                
                encoded[layer] = features
        
        return encoded

