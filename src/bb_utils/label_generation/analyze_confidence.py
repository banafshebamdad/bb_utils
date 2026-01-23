# 
# Banafshe Bamdad + GitHub Copilot (Claude Sonnet 4.5)
# January 23, 2026 16:10 CET
# 
# analyze_confidence.py
# 

#!/usr/bin/env python3
"""
Confidence Score Analysis Tool

Analyzes confidence scores computed from SLAM uncertainty metrics in 
semidense_points.csv.gz files. Generates statistical reports and visualizations
to help understand the distribution and characteristics of confidence values.

Usage:
------
python analyze_confidence.py <path_to_semidense_points.csv.gz> --output <output_dir>

Example:
python src/bb_utils/label_generation/analyze_confidence.py \
    /home/ubuntu/InCrowd-VI_points_observations/AND_corridors/mps_AND_lift_A_to_Lift_C_vrs/semidense_points.csv.gz \
    --output /home/ubuntu/confidence_analysis
"""

import argparse
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================================
# CONFIDENCE COMPUTATION (from incrowdvi_superpoint_labels.py)
# ============================================================================

def compute_confidence_inverse_normalized(inv_dist_std: np.ndarray, 
                                         max_inv_dist_std: float,
                                         **kwargs) -> np.ndarray:
    """
    Compute confidence as normalized inverse of uncertainty.
    
    Higher inv_dist_std (uncertainty) → lower confidence
    Formula: confidence = 1 - (inv_dist_std / max_inv_dist_std)
    """
    if max_inv_dist_std == 0:
        return np.ones_like(inv_dist_std)
    
    confidence = 1.0 - (inv_dist_std / max_inv_dist_std)
    confidence = np.clip(confidence, 0.0, 1.0)
    return confidence


def compute_confidence_exponential(inv_dist_std: np.ndarray,
                                   scale: float = 1.0,
                                   **kwargs) -> np.ndarray:
    """
    Compute confidence using exponential decay.
    
    Formula: confidence = exp(-scale * inv_dist_std)
    """
    confidence = np.exp(-scale * inv_dist_std)
    return confidence


CONFIDENCE_METHODS = {
    'inverse_normalized': compute_confidence_inverse_normalized,
    'exponential': compute_confidence_exponential,
}


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_and_compute_confidence(csv_path: Path, methods: dict) -> pd.DataFrame:
    """
    Load semidense_points.csv.gz and compute confidence scores.
    
    Parameters:
    -----------
    csv_path : Path
        Path to semidense_points.csv.gz
    methods : dict
        Dictionary of confidence computation methods
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with original columns plus confidence scores for each method
    """
    print(f"Loading data from {csv_path}...")
    with gzip.open(csv_path, 'rt') as f:
        # Use tqdm to show progress (pandas doesn't support direct progress for read_csv)
        df = pd.read_csv(f)
    
    print(f"✓ Loaded {len(df):,} points")
    
    # Compute confidence for each method
    max_inv_dist_std = df['inv_dist_std'].max()
    
    print("\nComputing confidence scores...")
    for method_name, method_func in tqdm(methods.items(), desc="Methods"):
        if method_name == 'inverse_normalized':
            df[f'confidence_{method_name}'] = method_func(
                df['inv_dist_std'].values,
                max_inv_dist_std=max_inv_dist_std
            )
        elif method_name == 'exponential':
            # Try different scales
            for scale in [1.0, 5.0, 10.0]:
                col_name = f'confidence_{method_name}_scale{scale}'
                df[col_name] = method_func(
                    df['inv_dist_std'].values,
                    scale=scale
                )
    
    return df


def generate_statistics(df: pd.DataFrame, output_path: Path):
    """
    Generate statistical report for confidence scores.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with confidence scores
    output_path : Path
        Path to save report
    """
    confidence_cols = [col for col in df.columns if col.startswith('confidence_')]
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SLAM Confidence Score Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total points: {len(df):,}\n\n")
        
        # Raw uncertainty statistics
        f.write("Raw Uncertainty Metrics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"inv_dist_std:\n")
        f.write(f"  Min:    {df['inv_dist_std'].min():.6f}\n")
        f.write(f"  Max:    {df['inv_dist_std'].max():.6f}\n")
        f.write(f"  Mean:   {df['inv_dist_std'].mean():.6f}\n")
        f.write(f"  Median: {df['inv_dist_std'].median():.6f}\n")
        f.write(f"  Std:    {df['inv_dist_std'].std():.6f}\n")
        f.write(f"\ndist_std:\n")
        f.write(f"  Min:    {df['dist_std'].min():.6f}\n")
        f.write(f"  Max:    {df['dist_std'].max():.6f}\n")
        f.write(f"  Mean:   {df['dist_std'].mean():.6f}\n")
        f.write(f"  Median: {df['dist_std'].median():.6f}\n")
        f.write(f"  Std:    {df['dist_std'].std():.6f}\n\n")
        
        # Confidence statistics for each method
        f.write("=" * 80 + "\n")
        f.write("Confidence Score Statistics:\n")
        f.write("=" * 80 + "\n\n")
        
        for conf_col in confidence_cols:
            method_name = conf_col.replace('confidence_', '')
            f.write(f"\n{method_name.upper()}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Min:       {df[conf_col].min():.6f}\n")
            f.write(f"  Max:       {df[conf_col].max():.6f}\n")
            f.write(f"  Mean:      {df[conf_col].mean():.6f}\n")
            f.write(f"  Median:    {df[conf_col].median():.6f}\n")
            f.write(f"  Std:       {df[conf_col].std():.6f}\n")
            f.write(f"  Percentiles:\n")
            f.write(f"    1%:      {df[conf_col].quantile(0.01):.6f}\n")
            f.write(f"    5%:      {df[conf_col].quantile(0.05):.6f}\n")
            f.write(f"    10%:     {df[conf_col].quantile(0.10):.6f}\n")
            f.write(f"    25%:     {df[conf_col].quantile(0.25):.6f}\n")
            f.write(f"    50%:     {df[conf_col].quantile(0.50):.6f}\n")
            f.write(f"    75%:     {df[conf_col].quantile(0.75):.6f}\n")
            f.write(f"    90%:     {df[conf_col].quantile(0.90):.6f}\n")
            f.write(f"    95%:     {df[conf_col].quantile(0.95):.6f}\n")
            f.write(f"    99%:     {df[conf_col].quantile(0.99):.6f}\n")
            
            # Distribution analysis
            f.write(f"\n  Distribution Analysis:\n")
            high_conf = (df[conf_col] > 0.9).sum()
            med_conf = ((df[conf_col] >= 0.5) & (df[conf_col] <= 0.9)).sum()
            low_conf = (df[conf_col] < 0.5).sum()
            f.write(f"    High confidence (>0.9):    {high_conf:,} ({100*high_conf/len(df):.2f}%)\n")
            f.write(f"    Medium confidence (0.5-0.9): {med_conf:,} ({100*med_conf/len(df):.2f}%)\n")
            f.write(f"    Low confidence (<0.5):     {low_conf:,} ({100*low_conf/len(df):.2f}%)\n")
    
    print(f"✓ Statistics saved to: {output_path}")


def save_sorted_data_all_methods(df: pd.DataFrame, output_path: Path):
    """
    Save all points with all confidence methods to text file.
    Sorted by inverse_normalized method (primary method used in label generation).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with confidence scores for all methods
    output_path : Path
        Path to save sorted data
    """
    # Get all confidence columns
    confidence_cols = [col for col in df.columns if col.startswith('confidence_')]
    
    print(f"\nSorting {len(df):,} points by confidence_inverse_normalized...")
    df_sorted = df.sort_values(by='confidence_inverse_normalized', ascending=False)
    
    print(f"Writing sorted data with all methods to {output_path.name}...")
    
    # Prepare columns to write
    base_cols = ['uid', 'inv_dist_std', 'dist_std']
    all_cols = base_cols + confidence_cols
    subset = df_sorted[all_cols]
    
    with open(output_path, 'w') as f:
        # Write header
        f.write(f"All {len(df_sorted):,} points sorted by confidence_inverse_normalized (descending)\n")
        f.write("=" * 150 + "\n")
        
        # Dynamic header based on methods present
        header_parts = [f"{'uid':>10}", f"{'inv_dist_std':>15}", f"{'dist_std':>15}"]
        for conf_col in confidence_cols:
            method_name = conf_col.replace('confidence_', '')[:15]  # truncate if needed
            header_parts.append(f"{method_name:>15}")
        f.write(" ".join(header_parts) + "\n")
        f.write("-" * 150 + "\n")
        
        # Write data in chunks with progress bar
        chunk_size = 10000
        for i in tqdm(range(0, len(subset), chunk_size), desc="Writing chunks"):
            chunk = subset.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                # Base columns
                line_parts = [
                    f"{row['uid']:>10}",
                    f"{row['inv_dist_std']:>15.6f}",
                    f"{row['dist_std']:>15.6f}"
                ]
                # Confidence columns
                for conf_col in confidence_cols:
                    line_parts.append(f"{row[conf_col]:>15.6f}")
                f.write(" ".join(line_parts) + "\n")
    
    print(f"✓ Sorted data saved to: {output_path}")


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """
    Generate visualization plots for confidence analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with confidence scores
    output_dir : Path
        Directory to save plots
    """
    confidence_cols = [col for col in df.columns if col.startswith('confidence_')]
    
    # Set style
    sns.set_style("whitegrid")
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # 1. Distribution histograms - separate image for each method
    print("\n[1/4] Creating histograms...")
    
    for conf_col in confidence_cols:
        method_name = conf_col.replace('confidence_', '')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.hist(df[conf_col], bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Confidence Distribution: {method_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = df[conf_col].mean()
        median_val = df[conf_col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.legend()
        
        plt.tight_layout()
        hist_path = output_dir / f'confidence_histogram_{method_name}.png'
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {hist_path.name}")
    
    # 2. Cumulative distribution functions
    print("\n[2/4] Creating CDF plots...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for conf_col in confidence_cols:
        method_name = conf_col.replace('confidence_', '')
        sorted_vals = np.sort(df[conf_col])
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cumulative, linewidth=2, label=method_name)
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Functions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    cdf_path = output_dir / 'confidence_cdf.png'
    plt.savefig(cdf_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {cdf_path.name}")
    
    # 3. Confidence vs uncertainty scatter plots - separate image for each method
    print("\n[3/4] Creating scatter plots...")
    
    for conf_col in confidence_cols:
        method_name = conf_col.replace('confidence_', '')
        
        # Sample points for faster plotting (if too many)
        if len(df) > 50000:
            df_sample = df.sample(n=50000, random_state=42)
            print(f"  Sampling 50,000/{len(df):,} points for {method_name}...")
        else:
            df_sample = df
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Confidence vs inv_dist_std
        axes[0].scatter(df_sample['inv_dist_std'], df_sample[conf_col], 
                       alpha=0.3, s=1)
        axes[0].set_xlabel('inv_dist_std', fontsize=12)
        axes[0].set_ylabel('Confidence', fontsize=12)
        axes[0].set_title(f'{method_name}: Confidence vs inv_dist_std', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence vs dist_std
        axes[1].scatter(df_sample['dist_std'], df_sample[conf_col], 
                       alpha=0.3, s=1)
        axes[1].set_xlabel('dist_std', fontsize=12)
        axes[1].set_ylabel('Confidence', fontsize=12)
        axes[1].set_title(f'{method_name}: Confidence vs dist_std', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        scatter_path = output_dir / f'confidence_vs_uncertainty_{method_name}.png'
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {scatter_path.name}")
    
    # 4. Box plots comparing methods
    print("\n[4/4] Creating box plots...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    box_data = [df[col] for col in confidence_cols]
    labels = [col.replace('confidence_', '') for col in confidence_cols]
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(confidence_cols)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Confidence Score', fontsize=12)
    ax.set_title('Confidence Score Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    box_path = output_dir / 'confidence_boxplots.png'
    plt.savefig(box_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {box_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze confidence scores from semidense_points.csv.gz",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('csv_path', type=Path,
                       help='Path to semidense_points.csv.gz file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SLAM Confidence Score Analysis")
    print("="*80)
    
    # Load and compute confidence
    df = load_and_compute_confidence(args.csv_path, CONFIDENCE_METHODS)
    
    # Generate statistics report
    print("\nGenerating statistics report...")
    stats_path = args.output / 'confidence_statistics.txt'
    generate_statistics(df, stats_path)
    
    # Save sorted data with all methods
    sorted_path = args.output / 'sorted_confidence_all_methods.txt'
    save_sorted_data_all_methods(df, sorted_path)
    
    # Generate visualizations
    generate_visualizations(df, args.output)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
