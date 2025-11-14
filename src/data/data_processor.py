"""
Data processor for OR-Bench dataset.

Handles data preprocessing, balancing, and preparation for inference and SAE training.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from .orbench_loader import ORBenchLoader


class DataProcessor:
    """Processes and balances OR-Bench data for analysis."""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize data processor.
        
        Args:
            dataset_dir: Path to OR-Bench dataset directory
        """
        self.loader = ORBenchLoader(dataset_dir)
    
    def prepare_balanced_dataset(
        self,
        categories: Optional[List[str]] = None,
        strategy: str = "use_all",
        shuffle: bool = True
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
        """
        Prepare balanced dataset for activation collection.
        
        Uses two-stage approach:
        - Stage 1: Collect all available toxic samples per category, match with safe
        - Stage 2: SAE training will use single SAE on all data (ensures comparability)
        
        Args:
            categories: List of categories to process (None = all available)
            strategy: "use_all" (use all toxic) or "equalize" (use minimum)
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (data_by_category, toxic_counts)
        """
        if categories is None:
            categories = self.loader.get_available_categories()
        
        print("=" * 80)
        print("Preparing Balanced Dataset")
        print("=" * 80)
        print(f"Strategy: {strategy}")
        print(f"Categories: {', '.join(categories)}")
        
        # Load balanced dataset
        data_by_category, toxic_counts = self.loader.load_balanced_dataset(
            categories=categories,
            strategy=strategy,
            shuffle=shuffle
        )
        
        # Print summary
        total_samples = sum(len(data) for data in data_by_category.values())
        total_toxic = sum(toxic_counts.values())
        total_safe = total_samples - total_toxic
        
        print("\n" + "=" * 80)
        print("Dataset Summary")
        print("=" * 80)
        print(f"Total samples: {total_samples}")
        print(f"  Safe: {total_safe} ({total_safe/total_samples*100:.1f}%)")
        print(f"  Toxic: {total_toxic} ({total_toxic/total_samples*100:.1f}%)")
        print(f"\nSamples per category:")
        for category in categories:
            num_samples = len(data_by_category[category])
            num_toxic = toxic_counts[category]
            print(f"  {category}: {num_samples} total ({num_toxic} toxic, {num_samples-num_toxic} safe)")
        
        return data_by_category, toxic_counts
    
    def compute_category_weights(self, toxic_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Compute inverse frequency weights for categories.
        
        Used during SAE training to handle category imbalance.
        
        Args:
            toxic_counts: Dictionary mapping category to toxic sample count
            
        Returns:
            Dictionary mapping category to weight
        """
        # Inverse frequency weighting
        total_samples = sum(toxic_counts.values())
        num_categories = len(toxic_counts)
        
        weights = {}
        for category, count in toxic_counts.items():
            # Weight inversely proportional to frequency
            # Categories with fewer samples get higher weight
            weights[category] = total_samples / (num_categories * count)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {cat: w / total_weight for cat, w in weights.items()}
        
        return weights

