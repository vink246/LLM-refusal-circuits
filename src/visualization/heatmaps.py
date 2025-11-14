"""
Visualization functions for circuit similarity heatmaps.

Generates heatmaps for:
- Safe circuits: category-to-category similarity
- Toxic circuits: category-to-category similarity  
- Cross-refusal: safe category vs toxic category similarity
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from pathlib import Path

from ..circuits.circuit import SparseFeatureCircuit
from ..circuits.similarity import compute_circuit_similarity


def create_safe_toxic_heatmaps(
    safe_circuits: Dict[str, SparseFeatureCircuit],
    toxic_circuits: Dict[str, SparseFeatureCircuit],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 12)
):
    """
    Create two heatmaps: one for safe circuits, one for toxic circuits.
    Shows category-to-category similarity within each refusal type.
    
    Args:
        safe_circuits: Dictionary mapping category to safe circuit
        toxic_circuits: Dictionary mapping category to toxic circuit
        output_dir: Directory to save heatmaps
        figsize: Figure size
    """
    categories = sorted(list(set(list(safe_circuits.keys()) + list(toxic_circuits.keys()))))
    
    # Safe circuits heatmap
    if safe_circuits:
        safe_matrix = np.zeros((len(categories), len(categories)))
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if cat1 in safe_circuits and cat2 in safe_circuits:
                    if i == j:
                        safe_matrix[i, j] = 1.0
                    else:
                        safe_matrix[i, j] = compute_circuit_similarity(
                            safe_circuits[cat1], safe_circuits[cat2]
                        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            safe_matrix,
            xticklabels=categories,
            yticklabels=categories,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Similarity'}
        )
        plt.title('Safe Circuit Similarity Across Categories', fontsize=16, pad=20)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'safe_circuit_similarity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: safe_circuit_similarity_heatmap.png")
    
    # Toxic circuits heatmap
    if toxic_circuits:
        toxic_matrix = np.zeros((len(categories), len(categories)))
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if cat1 in toxic_circuits and cat2 in toxic_circuits:
                    if i == j:
                        toxic_matrix[i, j] = 1.0
                    else:
                        toxic_matrix[i, j] = compute_circuit_similarity(
                            toxic_circuits[cat1], toxic_circuits[cat2]
                        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            toxic_matrix,
            xticklabels=categories,
            yticklabels=categories,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Similarity'}
        )
        plt.title('Toxic Circuit Similarity Across Categories', fontsize=16, pad=20)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'toxic_circuit_similarity_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: toxic_circuit_similarity_heatmap.png")


def create_cross_refusal_heatmap(
    safe_circuits: Dict[str, SparseFeatureCircuit],
    toxic_circuits: Dict[str, SparseFeatureCircuit],
    output_dir: Path,
    figsize: Tuple[int, int] = (14, 12)
):
    """
    Create heatmap comparing safe circuits (rows) vs toxic circuits (columns).
    Main diagonal shows if refusal circuits for a category are the same as safe circuits.
    
    Args:
        safe_circuits: Dictionary mapping category to safe circuit
        toxic_circuits: Dictionary mapping category to toxic circuit
        output_dir: Directory to save heatmap
        figsize: Figure size
    """
    categories = sorted(list(set(list(safe_circuits.keys()) + list(toxic_circuits.keys()))))
    
    # Cross-refusal similarity matrix
    cross_matrix = np.zeros((len(categories), len(categories)))
    for i, safe_cat in enumerate(categories):
        for j, toxic_cat in enumerate(categories):
            if safe_cat in safe_circuits and toxic_cat in toxic_circuits:
                cross_matrix[i, j] = compute_circuit_similarity(
                    safe_circuits[safe_cat], toxic_circuits[toxic_cat]
                )
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cross_matrix,
        xticklabels=[f"{cat} (toxic)" for cat in categories],
        yticklabels=[f"{cat} (safe)" for cat in categories],
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Similarity'}
    )
    plt.title('Safe vs Toxic Circuit Similarity (Cross-Refusal)', fontsize=16, pad=20)
    plt.xlabel('Toxic Circuits (by Category)', fontsize=12)
    plt.ylabel('Safe Circuits (by Category)', fontsize=12)
    
    # Highlight diagonal (same category comparisons)
    for i in range(len(categories)):
        plt.gca().add_patch(
            plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=2)
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_refusal_similarity_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: cross_refusal_similarity_heatmap.png")
    
    # Print diagonal values (same category, safe vs toxic)
    print("\n  Same-category safe vs toxic similarities (diagonal):")
    for i, cat in enumerate(categories):
        if i < len(categories):
            similarity = cross_matrix[i, i]
            print(f"    {cat}: {similarity:.3f}")


def create_similarity_heatmap(
    circuits: Dict[str, SparseFeatureCircuit],
    output_dir: Path,
    title: str = "Circuit Similarity",
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create a general similarity heatmap for circuits.
    
    Args:
        circuits: Dictionary mapping category to circuit
        output_dir: Directory to save heatmap
        title: Heatmap title
        figsize: Figure size
    """
    categories = sorted(list(circuits.keys()))
    
    similarity_matrix = np.zeros((len(categories), len(categories)))
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = compute_circuit_similarity(
                    circuits[cat1], circuits[cat2]
                )
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Similarity'}
    )
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f'{title.lower().replace(" ", "_")}_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

