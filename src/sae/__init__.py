"""
Sparse Autoencoder module.

Handles SAE model definition, training, and management.

IMPORTANT: All SAE training uses a SINGLE SAE on ALL data to ensure
feature space comparability across categories and refusal types.
"""

# Will be populated when modules are migrated
# from .sae_model import SparseAutoencoder
# from .sae_trainer import SAETrainer
from .sae_manager import SAEManager, ActivationDataset

__all__ = ['SAEManager', 'ActivationDataset']

