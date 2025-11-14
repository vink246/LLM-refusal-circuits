"""
Data analyzer for OR-Bench dataset.

Analyzes data splits across safe, toxic, and hard samples to identify imbalances.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


class DataAnalyzer:
    """Analyzes data distributions and splits in OR-Bench dataset."""
    
    def __init__(self, dataset_dir: str, config_path: Optional[str] = None):
        """
        Initialize data analyzer.
        
        Args:
            dataset_dir: Path to OR-Bench dataset directory
            config_path: Optional path to configuration YAML file
        """
        self.dataset_dir = Path(dataset_dir)
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize processor (will be imported from data_processor when migrated)
        # For now, we'll implement the analysis directly
        self._validate_dataset_files()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
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
    
    def _load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        file_path = self.dataset_dir / filename
        return pd.read_csv(file_path)
    
    def analyze_splits(self, output_dir: str = "results/data_analysis"):
        """
        Analyze and visualize data splits across safe, toxic, and hard samples.
        
        Args:
            output_dir: Output directory for analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("OR-Bench Data Split Analysis")
        print("=" * 80)
        
        # Load all datasets
        print("\nLoading datasets...")
        df_80k = self._load_csv_data("or-bench-80k.csv")
        df_hard = self._load_csv_data("or-bench-hard-1k.csv")
        df_toxic = self._load_csv_data("or-bench-toxic.csv")
        
        # Get categories to analyze
        config_categories = self.config.get('dataset', {}).get('categories', [])
        if config_categories:
            available_categories = config_categories
            print(f"\nAnalyzing specified categories: {', '.join(available_categories)}")
        else:
            # Get all available categories
            all_categories_80k = set(df_80k['category'].unique())
            all_categories_hard = set(df_hard['category'].unique())
            all_categories_toxic = set(df_toxic['category'].unique())
            available_categories = sorted(list(all_categories_80k | all_categories_hard | all_categories_toxic))
            print(f"\nFound {len(available_categories)} categories: {', '.join(available_categories)}")
        
        # Analyze splits per category
        category_stats = {}
        
        for category in available_categories:
            safe_80k = len(df_80k[df_80k['category'] == category])
            hard_1k = len(df_hard[df_hard['category'] == category])
            toxic = len(df_toxic[df_toxic['category'] == category])
            
            total = safe_80k + hard_1k + toxic
            
            category_stats[category] = {
                'safe_80k': safe_80k,
                'hard_1k': hard_1k,
                'toxic': toxic,
                'total': total,
                'safe_ratio': (safe_80k + hard_1k) / total if total > 0 else 0,
                'toxic_ratio': toxic / total if total > 0 else 0
            }
        
        # Print summary statistics
        self._print_summary(category_stats, df_80k, df_hard, df_toxic)
        
        # Generate visualizations
        if self.config.get('analysis', {}).get('generate_plots', True):
            self._create_visualizations(category_stats, available_categories, output_path)
        
        # Save statistics
        if self.config.get('output', {}).get('save_statistics', True):
            self._save_statistics(category_stats, df_80k, df_hard, df_toxic, output_path)
        
        # Print warnings
        self._print_warnings(category_stats)
    
    def _print_summary(self, category_stats: Dict, df_80k: pd.DataFrame, 
                      df_hard: pd.DataFrame, df_toxic: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "=" * 80)
        print("Category-wise Data Split Summary")
        print("=" * 80)
        print(f"{'Category':<20} {'Safe (80k)':<15} {'Hard (1k)':<15} {'Toxic':<15} {'Total':<10} {'Toxic %':<10}")
        print("-" * 80)
        
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            toxic_pct = stats['toxic_ratio'] * 100
            print(f"{category:<20} {stats['safe_80k']:<15} {stats['hard_1k']:<15} "
                  f"{stats['toxic']:<15} {stats['total']:<10} {toxic_pct:<10.2f}%")
        
        # Overall statistics
        total_safe_80k = len(df_80k)
        total_hard_1k = len(df_hard)
        total_toxic = len(df_toxic)
        total_all = total_safe_80k + total_hard_1k + total_toxic
        
        print("\n" + "=" * 80)
        print("Overall Dataset Statistics")
        print("=" * 80)
        print(f"Safe (80k dataset): {total_safe_80k:,} samples ({total_safe_80k/total_all*100:.2f}%)")
        print(f"Hard (1k dataset):  {total_hard_1k:,} samples ({total_hard_1k/total_all*100:.2f}%)")
        print(f"Toxic dataset:     {total_toxic:,} samples ({total_toxic/total_all*100:.2f}%)")
        print(f"Total:              {total_all:,} samples")
        print(f"\nToxic/Safe Ratio: {total_toxic/(total_safe_80k+total_hard_1k):.4f}")
    
    def _create_visualizations(self, category_stats: Dict, categories: List[str], 
                               output_path: Path):
        """Create visualization plots."""
        print("\nGenerating visualizations...")
        
        # Set style
        plot_config = self.config.get('analysis', {}).get('plots', {})
        figsize = plot_config.get('figsize', [16, 12])
        dpi = plot_config.get('dpi', 150)
        
        if plot_config.get('style'):
            plt.style.use(plot_config['style'])
        
        # Prepare data
        safe_counts = [category_stats[cat]['safe_80k'] for cat in categories]
        hard_counts = [category_stats[cat]['hard_1k'] for cat in categories]
        toxic_counts = [category_stats[cat]['toxic'] for cat in categories]
        total_counts = [category_stats[cat]['total'] for cat in categories]
        toxic_ratios = [category_stats[cat]['toxic_ratio'] * 100 for cat in categories]
        
        x_pos = range(len(categories))
        width = 0.6
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Stacked bar chart
        axes[0, 0].bar(x_pos, safe_counts, width, label='Safe (80k)', color='green', alpha=0.7)
        axes[0, 0].bar(x_pos, hard_counts, width, bottom=safe_counts, label='Hard (1k)', color='orange', alpha=0.7)
        axes[0, 0].bar(x_pos, toxic_counts, width, 
                      bottom=[s+h for s, h in zip(safe_counts, hard_counts)], 
                      label='Toxic', color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Number of Samples')
        axes[0, 0].set_title('Data Split by Category (Stacked)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Toxic ratio per category
        axes[0, 1].bar(x_pos, toxic_ratios, width, color='red', alpha=0.7)
        axes[0, 1].axhline(y=50, color='black', linestyle='--', label='Balanced (50%)')
        axes[0, 1].set_xlabel('Category')
        axes[0, 1].set_ylabel('Toxic Percentage (%)')
        axes[0, 1].set_title('Toxic Sample Percentage by Category')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Total samples per category
        axes[1, 0].bar(x_pos, total_counts, width, color='blue', alpha=0.7)
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Total Samples')
        axes[1, 0].set_title('Total Samples per Category')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Heatmap
        heatmap_data = pd.DataFrame({
            'Safe (80k)': safe_counts,
            'Hard (1k)': hard_counts,
            'Toxic': toxic_counts
        }, index=categories)
        
        sns.heatmap(heatmap_data.T, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=axes[1, 1], cbar_kws={'label': 'Sample Count'})
        axes[1, 1].set_title('Sample Count Heatmap by Category')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Dataset Type')
        
        plt.tight_layout()
        plot_path = output_path / "data_split_analysis.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization: {plot_path}")
    
    def _save_statistics(self, category_stats: Dict, df_80k: pd.DataFrame,
                        df_hard: pd.DataFrame, df_toxic: pd.DataFrame, output_path: Path):
        """Save statistics to JSON."""
        total_safe_80k = len(df_80k)
        total_hard_1k = len(df_hard)
        total_toxic = len(df_toxic)
        total_all = total_safe_80k + total_hard_1k + total_toxic
        
        stats = {
            'overall': {
                'total_safe_80k': total_safe_80k,
                'total_hard_1k': total_hard_1k,
                'total_toxic': total_toxic,
                'total_all': total_all,
                'toxic_ratio': total_toxic / total_all if total_all > 0 else 0
            },
            'by_category': category_stats
        }
        
        stats_file = output_path / "data_split_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved statistics: {stats_file}")
    
    def _print_warnings(self, category_stats: Dict):
        """Print warnings about data imbalances."""
        print("\n" + "=" * 80)
        print("Data Imbalance Warnings")
        print("=" * 80)
        
        min_toxic_ratio = self.config.get('analysis', {}).get('min_toxic_ratio_warning', 10.0)
        imbalanced_categories = []
        
        for category, stats in category_stats.items():
            toxic_pct = stats['toxic_ratio'] * 100
            if toxic_pct < min_toxic_ratio:
                imbalanced_categories.append((category, toxic_pct))
                print(f"⚠️  {category}: Only {toxic_pct:.2f}% toxic samples "
                      f"({stats['toxic']} toxic vs {stats['safe_80k'] + stats['hard_1k']} safe)")
            elif toxic_pct > 90:
                print(f"⚠️  {category}: {toxic_pct:.2f}% toxic samples "
                      f"({stats['toxic']} toxic vs {stats['safe_80k'] + stats['hard_1k']} safe)")
        
        if imbalanced_categories:
            print(f"\n⚠️  Found {len(imbalanced_categories)} categories with <{min_toxic_ratio}% toxic samples.")
        else:
            print("✓ No severe imbalances detected.")
        
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)

