"""
SAE Trainer for training sparse autoencoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
from .sae_model import SparseAutoencoder


class SAETrainer:
    """Trainer for sparse autoencoders"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coeff: float = 0.01,
        k_percent: float = 0.05,
        lr: float = 1e-3,
        device: str = "cuda"
    ):
        self.device = device
        self.sae = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff, k_percent).to(device)
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train(self, dataloader: DataLoader, epochs: int = 100) -> Dict[str, List[float]]:
        """Train the SAE"""
        
        history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'sparsity_loss': [],
            'l0_norm': []  # Average number of active features
        }
        
        self.sae.train()
        
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_sparsity_loss = 0.0
            epoch_l0_norm = 0.0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch = batch.to(self.device).float()
                
                self.optimizer.zero_grad()
                
                reconstructed, features, reconstruction_loss = self.sae(batch)
                
                # The loss is now just reconstruction loss (sparsity enforced structurally)
                total_loss = reconstruction_loss
                
                # Compute additional metrics
                with torch.no_grad():
                    sparsity_loss = torch.tensor(0.0)  # No explicit sparsity loss
                    l0_norm = (features > 0).float().sum(dim=-1).mean()
                
                total_loss.backward()
                self.optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += reconstruction_loss.item()
                epoch_sparsity_loss += sparsity_loss.item()
                epoch_l0_norm += l0_norm.item()
                num_batches += 1
            
            # Average over batches
            epoch_total_loss /= num_batches
            epoch_recon_loss /= num_batches  
            epoch_sparsity_loss /= num_batches
            epoch_l0_norm /= num_batches
            
            history['total_loss'].append(epoch_total_loss)
            history['reconstruction_loss'].append(epoch_recon_loss)
            history['sparsity_loss'].append(epoch_sparsity_loss)
            history['l0_norm'].append(epoch_l0_norm)
            
            self.lr_scheduler.step(epoch_total_loss)
            
            print(f"Epoch {epoch+1}: Total Loss: {epoch_total_loss:.4f}, "
                  f"Recon: {epoch_recon_loss:.4f}, Sparsity: {epoch_sparsity_loss:.4f}, "
                  f"L0: {epoch_l0_norm:.1f}")
            
            # Early stopping check
            if epoch_total_loss < 1e-4:  # Convergence threshold
                print("Convergence reached, stopping early")
                break
        
        return history
    
    def save(self, filepath: str):
        """Save the trained SAE"""
        torch.save({
            'model_state_dict': self.sae.state_dict(),
            'input_dim': self.sae.input_dim,
            'hidden_dim': self.sae.hidden_dim,
            'sparsity_coeff': self.sae.sparsity_coeff,
            'k_percent': self.sae.k_percent
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str, device: str = "cuda"):
        """Load a trained SAE"""
        checkpoint = torch.load(filepath, map_location=device)
        trainer = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            sparsity_coeff=checkpoint['sparsity_coeff'],
            k_percent=checkpoint.get('k_percent', 0.05),
            device=device
        )
        trainer.sae.load_state_dict(checkpoint['model_state_dict'])
        return trainer

