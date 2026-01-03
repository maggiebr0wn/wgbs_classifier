"""
Module 3: Visualization & Batch Effects

Purpose:
    Generate required assignment plots and assess batch effects.
    
Required Plots:
    1. Fragment length distribution
    2. Start/end position distributions
    3. End motif distribution
    4. Methylation analysis

Batch Effects:
    - Test Discovery vs Validation differences
    - Check for confounding with disease status

Input:
    - data/processed/all_features.csv

Output:
    - results/figures/required_plots/ (4 required plots)
    - results/tables/batch_effects_summary.csv

Usage:
    As a script:
        python src/visualization.py
    
    In a notebook:
        from src.visualization import run_module_3
        run_module_3()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    ALL_FEATURES,
    REQUIRED_PLOTS_DIR,
    BATCH_EFFECTS_FILE,
    FRAGMENT_SIZE_CATEGORIES,
    RESULTS_DIR
)


# ============================================================================
# Required Assignment Plots
# ============================================================================

def plot_fragment_length_distribution(df, output_dir):
    """
    Generate fragment length distribution plot (REQUIRED).
    
    Shows:
    - Box plots of mean fragment sizes (ALS vs Control)
    - Fragment size category distributions
    """
    print("\n1. Fragment Length Distribution")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Box plot of mean fragment size
    ax = axes[0]
    
    # Prepare data
    plot_data = df[['disease_status', 'frag_mean']].copy()
    plot_data.columns = ['Disease Status', 'Mean Fragment Size (bp)']
    
    # Box plot
    sns.boxplot(data=plot_data, x='Disease Status', y='Mean Fragment Size (bp)', 
                palette={'als': 'red', 'ctrl': 'blue'}, ax=ax)
    
    # Add individual points
    sns.stripplot(data=plot_data, x='Disease Status', y='Mean Fragment Size (bp)',
                 color='black', alpha=0.5, size=6, ax=ax)
    
    # Add mono-nucleosome reference line
    ax.axhline(167, color='green', linestyle='--', linewidth=2, alpha=0.7, 
              label='Mono-nucleosome (167 bp)')
    
    ax.set_xlabel('Disease Status', fontsize=12)
    ax.set_ylabel('Mean Fragment Size (bp)', fontsize=12)
    ax.set_title('Fragment Size Distribution', fontsize=13, fontweight='bold')
    ax.set_xticklabels(['ALS', 'Control'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Statistical test
    als_frag = df[df['disease_status'] == 'als']['frag_mean']
    ctrl_frag = df[df['disease_status'] == 'ctrl']['frag_mean']
    stat, p_val = mannwhitneyu(als_frag, ctrl_frag)
    
    ax.text(0.02, 0.98, f'Mann-Whitney U\np = {p_val:.4f}', 
           transform=ax.transAxes, fontsize=10, va='top', ha='left',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: Fragment size categories (stacked bar)
    ax = axes[1]
    cat_cols = [c for c in df.columns if c.startswith('frag_pct_')]
    categories = [c.replace('frag_pct_', '').replace('_', ' ').title() for c in cat_cols]
    
    als_cats = df[df['disease_status'] == 'als'][cat_cols].mean().values
    ctrl_cats = df[df['disease_status'] == 'ctrl'][cat_cols].mean().values
    
    # Stacked bar chart
    x = ['ALS', 'Control']
    width = 0.6
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    bottom_als = 0
    bottom_ctrl = 0
    
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax.bar(0, als_cats[i], width, bottom=bottom_als, 
              color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.bar(1, ctrl_cats[i], width, bottom=bottom_ctrl, 
              color=color, alpha=0.8, label=cat, edgecolor='black', linewidth=0.5)
        
        bottom_als += als_cats[i]
        bottom_ctrl += ctrl_cats[i]
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Fragment Size Categories', fontsize=13, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x)
    ax.set_ylim([0, 100])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / '01_fragment_length_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def plot_position_distributions(df, output_dir):
    """
    Generate start/end position distribution plot (REQUIRED).
    
    Shows:
    - Mean coverage profile across chr21 (ALS vs Control)
    """
    print("\n2. Start/End Position Distributions")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get coverage bins (exclude metadata)
    coverage_cols = [c for c in df.columns 
                    if c.startswith('coverage_bin_') 
                    and c != 'coverage_bin_size']
    
    # Sort numerically
    coverage_cols = sorted(coverage_cols, key=lambda x: int(x.split('_')[-1]))
    
    if len(coverage_cols) == 0:
        print("  ⚠️  No coverage bins found")
        return
    
    print(f"  Using {len(coverage_cols)} coverage bins")
    
    # Calculate mean coverage
    als_coverage = df[df['disease_status'] == 'als'][coverage_cols].mean().values
    ctrl_coverage = df[df['disease_status'] == 'ctrl'][coverage_cols].mean().values
    
    print(f"  Coverage range: {als_coverage.min():.3f}% - {als_coverage.max():.3f}%")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    bin_positions = np.arange(len(coverage_cols))
    
    ax.plot(bin_positions, als_coverage, color='red', linewidth=1.5, label='ALS', alpha=0.8)
    ax.plot(bin_positions, ctrl_coverage, color='blue', linewidth=1.5, label='Control', alpha=0.8)
    ax.fill_between(bin_positions, als_coverage, ctrl_coverage, 
                    where=(als_coverage > ctrl_coverage), 
                    color='red', alpha=0.2)
    ax.fill_between(bin_positions, als_coverage, ctrl_coverage, 
                    where=(als_coverage <= ctrl_coverage), 
                    color='blue', alpha=0.2)
    
    ax.set_xlabel('Genomic Position (100 kb bins across chr21)', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Fragment Start Position Distribution Across chr21', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / '02_position_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def plot_motif_distribution(df, output_dir):
    """
    Generate end motif distribution plot (REQUIRED).
    
    Shows:
    - Top 20 most frequent k-mer proportions (ALS vs Control)
    - Motif diversity comparison
    """
    print("\n3. End Motif Distribution")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========================================================================
    # Plot 1: Top 20 k-mers (averaged across samples)
    # ========================================================================
    ax = axes[0]
    
    # Get all k-mer columns
    kmer_cols = [c for c in df.columns if c.startswith('kmer_')]
    
    if len(kmer_cols) == 0:
        print("  ⚠️  No k-mer columns found")
        return
    
    # Calculate mean frequency for each k-mer across all samples
    kmer_means = df[kmer_cols].mean().sort_values(ascending=False)
    
    # Get top 20
    top_20_kmers = kmer_means.head(20).index.tolist()
    
    # Calculate means for ALS vs Control
    als_means = df[df['disease_status'] == 'als'][top_20_kmers].mean()
    ctrl_means = df[df['disease_status'] == 'ctrl'][top_20_kmers].mean()
    
    x = np.arange(20)
    width = 0.35
    
    # Clean k-mer names for display (remove 'kmer_' prefix)
    kmer_names = [k.replace('kmer_', '') for k in top_20_kmers]
    
    ax.bar(x - width/2, als_means.values, width, label='ALS', color='red', alpha=0.7)
    ax.bar(x + width/2, ctrl_means.values, width, label='Control', color='blue', alpha=0.7)
    
    ax.set_xlabel('End Motif (4-mer)', fontsize=12)
    ax.set_ylabel('Frequency (%)', fontsize=12)
    ax.set_title('Top 20 End Motif Frequencies', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(kmer_names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # Plot 2: Motif diversity
    # ========================================================================
    ax = axes[1]
    
    if 'motif_diversity' in df.columns:
        df.boxplot(column='motif_diversity', by='disease_status', ax=ax)
        ax.set_title('Motif Diversity (Shannon Entropy)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Disease Status')
        ax.set_ylabel('Shannon Entropy (bits)')
        plt.sca(ax)
        plt.xticks(rotation=0)
    else:
        ax.text(0.5, 0.5, 'Motif diversity not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Motif Diversity', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / '03_end_motif_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


def plot_methylation_analysis(df, output_dir):
    """
    Generate methylation analysis plot (REQUIRED).
    
    Shows:
    - Global CpG methylation (ALS vs Control)
    - Methylation distribution (high/low/intermediate)
    - Regional methylation coverage
    """
    print("\n4. Methylation Analysis")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Mean CpG methylation
    ax = axes[0]
    if 'meth_mean_cpg' in df.columns:
        df['meth_mean_cpg_pct'] = df['meth_mean_cpg'] * 100
        df.boxplot(column='meth_mean_cpg_pct', by='disease_status', ax=ax)
        ax.set_title('Mean CpG Methylation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Disease Status')
        ax.set_ylabel('CpG Methylation (%)')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    # Plot 2: Methylation distribution
    ax = axes[1]
    
    if all(c in df.columns for c in ['meth_pct_high', 'meth_pct_low', 'meth_pct_intermediate']):
        als_dist = df[df['disease_status'] == 'als'][['meth_pct_high', 'meth_pct_intermediate', 'meth_pct_low']].mean()
        ctrl_dist = df[df['disease_status'] == 'ctrl'][['meth_pct_high', 'meth_pct_intermediate', 'meth_pct_low']].mean()
        
        categories = ['High\n(>80%)', 'Intermediate\n(20-80%)', 'Low\n(<20%)']
        x = np.arange(3)
        width = 0.35
        
        ax.bar(x - width/2, [als_dist['meth_pct_high'], als_dist['meth_pct_intermediate'], als_dist['meth_pct_low']], 
               width, label='ALS', color='red', alpha=0.7)
        ax.bar(x + width/2, [ctrl_dist['meth_pct_high'], ctrl_dist['meth_pct_intermediate'], ctrl_dist['meth_pct_low']], 
               width, label='Control', color='blue', alpha=0.7)
        
        ax.set_xlabel('Methylation Level', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Methylation Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Regional methylation coverage
    ax = axes[2]
    
    if 'regional_meth_n_bins_with_data' in df.columns:
        # Calculate coverage percentage
        n_bins = df['regional_meth_n_bins'].iloc[0] if 'regional_meth_n_bins' in df.columns else 467
        df['regional_coverage_pct'] = (df['regional_meth_n_bins_with_data'] / n_bins) * 100
        
        df.boxplot(column='regional_coverage_pct', by='disease_status', ax=ax)
        ax.set_title('Regional Methylation Coverage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Disease Status')
        ax.set_ylabel('Coverage (%)')
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    output_file = output_dir / '04_methylation_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_file.name}")


# ============================================================================
# Batch Effect Assessment
# ============================================================================

def assess_batch_effects(df, output_file):
    """
    Assess batch effects (Discovery vs Validation).
    
    Tests:
    1. Batch vs disease confounding (chi-square)
    2. Key features by batch (Mann-Whitney U)
    """
    print("\nBatch Effect Assessment")
    print("=" * 70)
    
    # Test 1: Batch-disease confounding
    print("\n1. Batch-Disease Confounding:")
    print("-" * 70)
    
    contingency = pd.crosstab(df['disease_status'], df['batch'])
    print(contingency)
    
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square test: χ² = {chi2:.4f}, p = {p_val:.4f}")
    
    if p_val > 0.05:
        print("✓ No significant confounding between batch and disease (p > 0.05)")
    else:
        print("⚠️  Batch and disease are confounded (p < 0.05)")
    
    # Test 2: Key features by batch
    print("\n\n2. Key Features by Batch:")
    print("-" * 70)
    
    # Select key features to test
    key_features = [
        'frag_mean',
        'frag_pct_long',
        'meth_mean_cpg',
        'regional_meth_mean',
        'motif_diversity'
    ]
    
    results = []
    
    for feat in key_features:
        if feat in df.columns:
            disc = df[df['batch'] == 'discovery'][feat].dropna()
            valid = df[df['batch'] == 'validation'][feat].dropna()
            
            if len(disc) > 0 and len(valid) > 0:
                stat, p_val = mannwhitneyu(disc, valid)
                
                results.append({
                    'feature': feat,
                    'discovery_mean': disc.mean(),
                    'discovery_std': disc.std(),
                    'validation_mean': valid.mean(),
                    'validation_std': valid.std(),
                    'p_value': p_val,
                    'significant': 'Yes' if p_val < 0.05 else 'No'
                })
                
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                print(f"{feat:25s}: Discovery={disc.mean():.4f}, Validation={valid.mean():.4f}, p={p_val:.4f} {sig}")
    
    # Save results
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved batch effect results to: {output_file}")
        
        n_sig = results_df['significant'].eq('Yes').sum()
        print(f"  Significant batch effects: {n_sig}/{len(results_df)} features tested")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_module_3():
    """
    Run complete Module 3 pipeline.
    
    Returns
    -------
    pd.DataFrame
        Loaded features dataframe
    """
    print("\n" + "=" * 70)
    print("MODULE 3: Visualization & Batch Effects")
    print("=" * 70)
    
    # Load features
    print(f"\nLoading features from: {ALL_FEATURES}")
    
    if not ALL_FEATURES.exists():
        raise FileNotFoundError(
            f"Feature file not found: {ALL_FEATURES}\n"
            f"Please run Module 2 first."
        )
    
    df = pd.read_csv(ALL_FEATURES)
    print(f"✓ Loaded: {df.shape[0]} samples × {df.shape[1]} features")
    
    # Generate required plots
    print("\n" + "=" * 70)
    print("GENERATING REQUIRED PLOTS")
    print("=" * 70)
    
    plot_fragment_length_distribution(df, REQUIRED_PLOTS_DIR)
    plot_position_distributions(df, REQUIRED_PLOTS_DIR)
    plot_motif_distribution(df, REQUIRED_PLOTS_DIR)
    plot_methylation_analysis(df, REQUIRED_PLOTS_DIR)
    
    # Assess batch effects
    print("\n" + "=" * 70)
    print("BATCH EFFECT ASSESSMENT")
    print("=" * 70)
    
    assess_batch_effects(df, BATCH_EFFECTS_FILE)
    
    print("\n" + "=" * 70)
    print("MODULE 3 COMPLETE")
    print("=" * 70)
    print(f"\nRequired plots saved to: {REQUIRED_PLOTS_DIR}")
    print(f"Batch effects summary: {BATCH_EFFECTS_FILE}")
    print("\n" + "=" * 70 + "\n")
    
    return df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 3 as a standalone script
    features_df = run_module_3()