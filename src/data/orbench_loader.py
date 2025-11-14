"""
OR-Bench dataset loader.

Handles loading and balancing OR-Bench dataset with support for different strategies
to handle safe/toxic and cross-category imbalances.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random


class ORBenchLoader:
    """Loads and processes OR-Bench dataset with balancing strategies."""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize OR-Bench loader.
        
        Args:
            dataset_dir: Path to OR-Bench dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self._validate_dataset_files()
    
    def _validate_dataset_files(self):
        """Validate that required OR-Bench files exist."""
        required_files = [
            "or-bench-80k.csv",
            "or-bench-hard-1k.csv",
            "or-bench-toxic.csv"
        ]
        
        for filename in required_files:
            file_path = self.dataset_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Required OR-Bench file not found: {file_path}")
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        file_path = self.dataset_dir / filename
        return pd.read_csv(file_path)
    
    def get_toxic_counts(self, categories: List[str]) -> Dict[str, int]:
        """
        Get count of toxic samples per category.
        
        Args:
            categories: List of categories to check
            
        Returns:
            Dictionary mapping category to toxic sample count
        """
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        counts = {}
        
        for category in categories:
            counts[category] = len(df_toxic[df_toxic['category'] == category])
        
        return counts
    
    def get_safe_counts(self, categories: List[str]) -> Dict[str, int]:
        """
        Get count of safe samples per category.
        
        Args:
            categories: List of categories to check
            
        Returns:
            Dictionary mapping category to safe sample count
        """
        df_80k = self.load_csv_data("or-bench-80k.csv")
        df_hard = self.load_csv_data("or-bench-hard-1k.csv")
        counts = {}
        
        for category in categories:
            safe_80k = len(df_80k[df_80k['category'] == category])
            hard_1k = len(df_hard[df_hard['category'] == category])
            counts[category] = safe_80k + hard_1k
        
        return counts
    
    def extract_category_data(
        self,
        category: str,
        num_safe_samples: int,
        num_toxic_samples: int,
        include_hard: bool = False,
        shuffle: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract data for a specific category.
        
        Args:
            category: Category name
            num_safe_samples: Number of safe samples to extract
            num_toxic_samples: Number of toxic samples to extract
            include_hard: Whether to include hard samples in safe count
            shuffle: Whether to shuffle results
            
        Returns:
            List of processed samples
        """
        category_data = []
        
        # Extract safe prompts from 80k dataset
        df_80k = self.load_csv_data("or-bench-80k.csv")
        safe_80k = df_80k[df_80k['category'] == category]
        
        if include_hard:
            df_hard = self.load_csv_data("or-bench-hard-1k.csv")
            hard_samples = df_hard[df_hard['category'] == category]
            safe_samples = pd.concat([safe_80k, hard_samples])
        else:
            safe_samples = safe_80k
        
        if len(safe_samples) > 0:
            if shuffle:
                safe_samples = safe_samples.sample(
                    n=min(num_safe_samples, len(safe_samples)),
                    random_state=42
                )
            else:
                safe_samples = safe_samples.head(num_safe_samples)
            
            for _, row in safe_samples.iterrows():
                category_data.append({
                    'prompt': str(row['prompt']).strip(),
                    'category': category,
                    'refusal_label': False,
                    'dataset_type': 'safe_80k' if not include_hard else 'safe',
                    'original_source': 'or-bench-80k.csv'
                })
        
        # Extract toxic prompts
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        toxic_samples = df_toxic[df_toxic['category'] == category]
        
        if len(toxic_samples) > 0:
            if shuffle:
                toxic_samples = toxic_samples.sample(
                    n=min(num_toxic_samples, len(toxic_samples)),
                    random_state=42
                )
            else:
                toxic_samples = toxic_samples.head(num_toxic_samples)
            
            for _, row in toxic_samples.iterrows():
                category_data.append({
                    'prompt': str(row['prompt']).strip(),
                    'category': category,
                    'refusal_label': True,
                    'dataset_type': 'toxic',
                    'original_source': 'or-bench-toxic.csv'
                })
        
        if shuffle:
            random.shuffle(category_data)
        
        return category_data
    
    def load_balanced_dataset(
        self,
        categories: List[str],
        strategy: str = "use_all",
        shuffle: bool = True
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
        """
        Load balanced dataset using specified strategy.
        
        Two-stage approach:
        - Stage 1: Use all available toxic samples per category, match with safe
        - Stage 2: SAE training will handle category imbalance via weighting
        
        Args:
            categories: List of categories to load
            strategy: "use_all" (use all toxic samples) or "equalize" (use minimum)
            shuffle: Whether to shuffle data
            
        Returns:
            Tuple of (data_by_category, toxic_counts)
            - data_by_category: Dictionary mapping category to sample list
            - toxic_counts: Dictionary mapping category to toxic sample count
        """
        # Get toxic counts per category
        toxic_counts = self.get_toxic_counts(categories)
        
        if strategy == "equalize":
            # Use minimum toxic samples across all categories
            min_toxic = min(toxic_counts.values())
            print(f"\nEqualizing: Using {min_toxic} samples per category")
            samples_per_category = {cat: min_toxic for cat in categories}
            
        elif strategy == "use_all":
            # Use all available toxic samples per category
            print("\nUsing all available toxic samples per category:")
            for cat, count in toxic_counts.items():
                print(f"  {cat}: {count} toxic samples")
            samples_per_category = toxic_counts
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'use_all' or 'equalize'")
        
        # Load balanced data for each category
        data_by_category = {}
        for category in categories:
            num_samples = samples_per_category[category]
            
            # Match safe samples to toxic count
            category_data = self.extract_category_data(
                category=category,
                num_safe_samples=num_samples,
                num_toxic_samples=num_samples,
                include_hard=False,
                shuffle=shuffle
            )
            
            data_by_category[category] = category_data
            
            # Verify balance
            num_safe = sum(1 for item in category_data if not item['refusal_label'])
            num_toxic = sum(1 for item in category_data if item['refusal_label'])
            print(f"  {category}: {num_safe} safe + {num_toxic} toxic = {len(category_data)} total")
        
        return data_by_category, toxic_counts
    
    def get_available_categories(self) -> List[str]:
        """Get all available categories in the dataset."""
        df_80k = self.load_csv_data("or-bench-80k.csv")
        df_hard = self.load_csv_data("or-bench-hard-1k.csv")
        df_toxic = self.load_csv_data("or-bench-toxic.csv")
        
        all_categories = set()
        all_categories.update(df_80k['category'].unique())
        all_categories.update(df_hard['category'].unique())
        all_categories.update(df_toxic['category'].unique())
        
        return sorted(list(all_categories))

