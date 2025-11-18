"""
Visualization module.

Handles circuit visualization, heatmaps, similarity plots, and training metrics.
"""

from .heatmaps import (
    create_similarity_heatmap,
    create_safe_toxic_heatmaps,
    create_cross_refusal_heatmap
)

from .training_plots import (
    plot_sae_training_history,
    plot_reconstruction_loss_only,
    plot_all_layers_comparison,
    load_and_plot_from_json
)

__all__ = [
    'create_similarity_heatmap',
    'create_safe_toxic_heatmaps',
    'create_cross_refusal_heatmap',
    'plot_sae_training_history',
    'plot_reconstruction_loss_only',
    'plot_all_layers_comparison',
    'load_and_plot_from_json'
]
