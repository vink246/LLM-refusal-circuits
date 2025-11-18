"""
Visualization functions for SAE training metrics.

Generates plots for:
- Reconstruction loss over epochs
- Total loss over epochs
- Sparsity loss over epochs
- L0 norm (active features) over epochs
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from pathlib import Path
import json


def plot_sae_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    layer: str,
    output_dir: Path,
    figsize: tuple = (12, 8)
):
    """
    Plot SAE training history showing loss curves over epochs.
    
    Args:
        history: Dictionary containing loss history with keys:
                 'reconstruction_loss', 'total_loss', 'sparsity_loss', 'l0_norm'
        model_name: Name of the model
        layer: Layer name
        output_dir: Directory to save the plot
        figsize: Figure size (width, height)
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'SAE Training History - {model_name} - {layer}', fontsize=16, y=0.995)
    
    epochs = range(1, len(history['reconstruction_loss']) + 1)
    
    # Plot reconstruction loss
    axes[0, 0].plot(epochs, history['reconstruction_loss'], 'b-', linewidth=2, label='Reconstruction Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Reconstruction Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot total loss
    axes[0, 1].plot(epochs, history['total_loss'], 'r-', linewidth=2, label='Total Loss')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Loss', fontsize=11)
    axes[0, 1].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot sparsity loss
    axes[1, 0].plot(epochs, history['sparsity_loss'], 'g-', linewidth=2, label='Sparsity Loss')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].set_title('Sparsity Loss', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot L0 norm (active features)
    axes[1, 1].plot(epochs, history['l0_norm'], 'm-', linewidth=2, label='Active Features')
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Count', fontsize=11)
    axes[1, 1].set_title('L0 Norm (Active Features)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    plot_path = output_dir / f'{safe_model_name}_{layer}_training_history.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved training plot: {plot_path}")


def plot_reconstruction_loss_only(
    history: Dict[str, List[float]],
    model_name: str,
    layer: str,
    output_dir: Path,
    figsize: tuple = (10, 6)
):
    """
    Plot only the reconstruction loss curve.
    
    Args:
        history: Dictionary containing loss history with key 'reconstruction_loss'
        model_name: Name of the model
        layer: Layer name
        output_dir: Directory to save the plot
        figsize: Figure size (width, height)
    """
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(history['reconstruction_loss']) + 1)
    
    ax.plot(epochs, history['reconstruction_loss'], 'b-', linewidth=2.5, marker='o', 
            markersize=4, markevery=max(1, len(epochs)//20))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title(f'SAE Reconstruction Loss - {model_name} - {layer}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add final loss as text
    final_loss = history['reconstruction_loss'][-1]
    ax.text(0.95, 0.95, f'Final Loss: {final_loss:.4f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    plot_path = output_dir / f'{safe_model_name}_{layer}_reconstruction_loss.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved reconstruction loss plot: {plot_path}")


def plot_all_layers_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    model_name: str,
    output_dir: Path,
    metric: str = 'reconstruction_loss',
    figsize: tuple = (12, 6)
):
    """
    Plot comparison of a specific metric across all layers.
    
    Args:
        histories: Dictionary mapping layer name to training history
        model_name: Name of the model
        output_dir: Directory to save the plot
        metric: Metric to plot ('reconstruction_loss', 'total_loss', etc.)
        figsize: Figure size (width, height)
    """
    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each layer
    colors = plt.cm.tab10(range(len(histories)))
    for (layer, history), color in zip(histories.items(), colors):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], '-', linewidth=2, 
                   label=layer, color=color)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison - {model_name}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.replace('/', '-').replace(' ', '_')
    plot_path = output_dir / f'{safe_model_name}_all_layers_{metric}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved multi-layer comparison: {plot_path}")


def load_and_plot_from_json(
    history_file: Path,
    model_name: str,
    layer: str,
    output_dir: Path
):
    """
    Load training history from JSON file and create plots.
    
    Args:
        history_file: Path to training history JSON file
        model_name: Name of the model
        layer: Layer name
        output_dir: Directory to save plots
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # Plot full training history
    plot_sae_training_history(history, model_name, layer, output_dir)
    
    # Plot reconstruction loss only
    plot_reconstruction_loss_only(history, model_name, layer, output_dir)

