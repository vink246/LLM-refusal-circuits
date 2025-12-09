"""
Reporting module for generating analysis reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

def generate_report(results_dir: Path, output_path: Path, format: str = 'markdown'):
    """
    Generate a comprehensive report from analysis results.
    
    Args:
        results_dir: Directory containing results
        output_path: Path to save the report
        format: Output format (currently only 'markdown')
    """
    if format != 'markdown':
        raise NotImplementedError(f"Format '{format}' not supported yet.")
        
    print(f"Generating report from {results_dir} to {output_path}")
    
    content = []
    content.append("# Circuit Analysis Report")
    content.append("\n## Executive Summary")
    content.append("This report summarizes the evaluation of discovered circuits for refusal behavior.")
    
    # 1. Faithfulness & Completeness
    content.append("\n## Faithfulness & Completeness")
    
    eval_dir = results_dir / "evaluation"
    if not eval_dir.exists():
        content.append("\nNo evaluation results found.")
    else:
        # Find all metrics json files
        metric_files = list(eval_dir.glob("*_metrics.json"))
        if not metric_files:
             content.append("\nNo metrics files found.")
        else:
            for m_file in metric_files:
                try:
                    with open(m_file, 'r') as f:
                        data = json.load(f)
                    
                    # Parse filename to get info
                    # Filename: {model}_{category}_metrics.json
                    name_parts = m_file.stem.split('_metrics')[0]
                    
                    content.append(f"\n### Analysis: {name_parts}")
                    
                    # Create table of results
                    if "thresholds" in data and "faithfulness" in data:
                        df = pd.DataFrame({
                            "Threshold": data.get("thresholds", []),
                            "Nodes": data.get("n_nodes", []),
                            "Faithfulness": data.get("faithfulness", []),
                            "Completeness": data.get("completeness", [])
                        })
                        content.append("\n" + df.to_markdown(index=False))
                    
                    # Embed plot
                    plot_file = eval_dir / f"{name_parts}_metrics.png"
                    if plot_file.exists():
                        # Use relative path for markdown
                        rel_path = plot_file.relative_to(output_path.parent)
                        content.append(f"\n![Faithfulness Plot]({rel_path})")
                        
                except Exception as e:
                    content.append(f"\nError processing {m_file.name}: {e}")

    # 2. Statistical Significance
    content.append("\n## Statistical Significance")
    
    if eval_dir.exists():
        stats_files = list(eval_dir.glob("*_stats.json"))
        if not stats_files:
            content.append("\nNo statistical test results found.")
        else:
            for s_file in stats_files:
                try:
                    with open(s_file, 'r') as f:
                        stats = json.load(f)
                    
                    name_parts = s_file.stem.split('_stats')[0]
                    content.append(f"\n### Stats: {name_parts}")
                    
                    content.append(f"- **Observed Similarity**: {stats.get('observed', 'N/A'):.4f}")
                    content.append(f"- **Random Baseline (Mean)**: {stats.get('random_mean', 'N/A'):.4f} ± {stats.get('random_std', 'N/A'):.4f}")
                    content.append(f"- **Z-Score**: {stats.get('z_score', 'N/A'):.4f}")
                    content.append(f"- **P-Value (Empirical)**: {stats.get('p_value_empirical', stats.get('p_value', 'N/A')):.4e}")
                    if 'p_value_normal' in stats:
                        content.append(f"- **P-Value (Normal Approx.)**: {stats.get('p_value_normal', 'N/A'):.4e}")
                    content.append(f"- **Permutations**: {stats.get('n_permutations', 'N/A')}")
                    content.append(f"- **95% CI**: [{stats.get('ci_lower', 'N/A'):.4f}, {stats.get('ci_upper', 'N/A'):.4f}]")
                    
                    p_val = stats.get('p_value_empirical', stats.get('p_value', 1.0))
                    if p_val < 0.05:
                        content.append("\n> **Result**: Significant difference from random baseline (p < 0.05).")
                    else:
                        content.append("\n> **Result**: No significant difference from random baseline.")
                        
                except Exception as e:
                    content.append(f"\nError processing {s_file.name}: {e}")

    # Write report
    with open(output_path, 'w') as f:
        f.write("\n".join(content))
        
    return output_path
