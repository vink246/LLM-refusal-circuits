#!/usr/bin/env python3
"""
Generate comprehensive final analysis report.

This script combines all analysis results into a comprehensive report
including data statistics, circuit discoveries, comparisons, and conclusions.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.reporting import generate_report
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive final analysis report"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory containing all results"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: results/reports/final_report.md)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['markdown', 'html', 'pdf', 'all'],
        default='markdown',
        help="Output format(s)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "reports" / "final_report.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Generating Final Report")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output: {output_path}")
    
    # Generate report
    report_generator = generate_report(results_dir, output_path, format=args.format)
    
    print("\n✓ Report generation complete!")
    print(f"  Report saved to: {output_path}")


if __name__ == "__main__":
    main()

