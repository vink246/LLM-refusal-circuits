#!/usr/bin/env python3
"""
Analyze data splits across safe, toxic, and hard samples for OR-Bench dataset.

This script helps identify data imbalances that may affect SAE training.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_analyzer import DataAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data splits across safe, toxic, and hard samples in OR-Bench"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to OR-Bench dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/data_analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to data configuration YAML file (optional, overrides defaults)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer with dataset directory
    analyzer = DataAnalyzer(dataset_dir=args.dataset_dir, config_path=args.config)
    
    # Run analysis
    analyzer.analyze_splits(output_dir=args.output_dir)
    
    print(f"\n✓ Data analysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

