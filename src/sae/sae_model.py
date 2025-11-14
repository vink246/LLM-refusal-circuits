"""
Sparse Autoencoder model definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with Top-K sparsity for guaranteed sparse features.
    This ensures sparsity regardless of training dynamics.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 0.01, k_percent: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff
        self.k_percent = k_percent
        self.k = max(1, int(hidden_dim * k_percent))  # Number of features to keep active
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        # Decoder  
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize weights for better sparsity
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to promote sparsity"""
        with torch.no_grad():
            # Initialize encoder with smaller weights to prevent saturation
            nn.init.xavier_uniform_(self.encoder.weight, gain=0.1)
            nn.init.zeros_(self.encoder.bias)
            
            # Initialize decoder as transpose of encoder (tied weights approach)
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            nn.init.zeros_(self.decoder.bias)
    
    def top_k_sparse_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity - only keep top k activations per sample"""
        # Get the top-k indices
        topk_vals, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # Create sparse activation tensor
        sparse_activation = torch.zeros_like(x)
        sparse_activation.scatter_(-1, topk_indices, topk_vals)
        
        return sparse_activation
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass with guaranteed top-k sparsity
        
        Returns:
            reconstructed: Reconstructed input
            features: Sparse feature activations (guaranteed sparse)
            loss: Reconstruction loss (sparsity enforced structurally)
        """
        # Encode
        pre_activation = self.encoder(x)
        
        # Apply ReLU then top-k sparsity for guaranteed sparsity
        relu_features = F.relu(pre_activation)
        features = self.top_k_sparse_activation(relu_features)
        
        # Decode
        reconstructed = self.decoder(features)
        
        # Loss is just reconstruction (sparsity is enforced structurally)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        return reconstructed, features, reconstruction_loss
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features"""
        pre_activation = self.encoder(x)
        relu_features = F.relu(pre_activation)
        return self.top_k_sparse_activation(relu_features)
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features to reconstruction"""
        return self.decoder(features)

