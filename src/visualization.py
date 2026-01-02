"""
Module 3: Exploratory Analysis & Visualization

Purpose:
    Generate required plots and perform exploratory data analysis.
    
Required Plots:
    1. Fragment length distribution
    2. Start and end position distributions
    3. End motif distribution
    4. Methylation analysis
    
Additional Analysis:
    - PCA and clustering
    - Batch effect visualization
    - Feature correlations
    - Univariate statistical tests

Input:
    - data/processed/all_features.csv (from Module 2)
    - data/processed/qc_metrics.csv (from Module 1)

Output:
    - results/figures/visualization/ (required plots)
    - results/figures/eda/ (exploratory plots)
    - results/tables/ (summary statistics)

Functions:
    - plot_fragment_distribution(): Fragment size distribution plots
    - plot_position_distributions(): Start/end position plots
    - plot_motif_distribution(): End motif frequency plots
    - plot_methylation_analysis(): Methylation analysis plots
    - perform_pca(): PCA analysis and visualization
    - plot_feature_correlations(): Correlation heatmaps
    - univariate_tests(): Statistical tests for all features
    - run_module_3(): Execute complete Module 3 pipeline

Usage:
    As a script:
        python src/visualization.py
    
    In a notebook or other module:
        from src.visualization import run_module_3
        run_module_3()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    ALL_FEATURES,
    QC_METRICS,
    VIZ_FIGURES_DIR,
    EDA_FIGURES_DIR,
    SUMMARY_TABLES_DIR,
    FEATURE_SUMMARY,
    UNIVARIATE_TESTS,
    RESULTS_DIR,
    FIGURES_DIR
)


def plot_fragment_distribution(features_df, output_dir):
    """
    Generate fragment length distribution plots (REQUIRED FOR ASSIGNMENT).
    
    Creates:
    1. Histogram of fragment sizes (ALS vs Control)
    2. Violin plots by disease status and batch
    3. Summary statistics comparison
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with fragment size features
    output_dir : Path
        Directory to save plots
    """
    print("\n1. Fragment Length Distribution (REQUIRED)")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Histogram - Fragment size distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overlaid histograms
    ax = axes[0]
    als_mean = features_df[features_df['disease_status'] == 'als']['frag_mean']
    ctrl_mean = features_df[features_df['disease_status'] == 'ctrl']['frag_mean']
    
    ax.hist(als_mean, bins=15, alpha=0.6, label='ALS', color='red', edgecolor='black')
    ax.hist(ctrl_mean, bins=15, alpha=0.6, label='Control', color='blue', edgecolor='black')
    ax.set_xlabel('Mean Fragment Size (bp)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Fragment Size Distribution: ALS vs Control', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add statistical test
    stat, p_value = mannwhitneyu(als_mean, ctrl_mean)
    ax.text(0.98, 0.95, f'Mann-Whitney U\np = {p_value:.4f}', 
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Violin plot by disease and batch
    ax = axes[1]
    sns.violinplot(data=features_df, x='disease_status', y='frag_mean',
                  hue='batch', ax=ax, inner='box', split=False)
    ax.set_xlabel('Disease Status', fontsize=12)
    ax.set_ylabel('Mean Fragment Size (bp)', fontsize=12)
    ax.set_title('Fragment Size by Disease and Batch', fontsize=13, fontweight='bold')
    ax.legend(title='Batch', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fragment_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fragment_length_distribution.png")
    
    # Plot 2: Detailed fragment size statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fragment Size Features: ALS vs Control', fontsize=14, fontweight='bold')
    
    # Mean, median, std, IQR
    metrics = [
        ('frag_mean', 'Mean Fragment Size (bp)'),
        ('frag_median', 'Median Fragment Size (bp)'),
        ('frag_std', 'Fragment Size Std Dev (bp)'),
        ('frag_iqr', 'Fragment Size IQR (bp)')
    ]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if metric in features_df.columns:
            sns.violinplot(data=features_df, x='disease_status', y=metric,
                          hue='batch', ax=ax, inner='quartile')
            ax.set_xlabel('Disease Status', fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.legend(title='Batch', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fragment_size_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fragment_size_detailed.png")
    
    # Plot 3: Fragment size bins
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_cols = [c for c in features_df.columns if c.startswith('frag_pct_')]
    if len(bin_cols) > 0:
        # Prepare data for plotting
        bin_data = []
        for bin_col in bin_cols:
            bin_name = bin_col.replace('frag_pct_', '')
            for idx, row in features_df.iterrows():
                bin_data.append({
                    'bin': bin_name,
                    'percentage': row[bin_col],
                    'disease_status': row['disease_status'],
                    'batch': row['batch']
                })
        
        bin_df = pd.DataFrame(bin_data)
        
        sns.barplot(data=bin_df, x='bin', y='percentage', hue='disease_status', ax=ax)
        ax.set_xlabel('Fragment Size Bin', fontsize=12)
        ax.set_ylabel('Percentage of Fragments (%)', fontsize=12)
        ax.set_title('Fragment Size Distribution by Bins', fontsize=13, fontweight='bold')
        ax.legend(title='Disease Status', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fragment_size_bins.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fragment_size_bins.png")


def plot_position_distributions(features_df, output_dir):
    """
    Generate start and end position distribution plots (REQUIRED FOR ASSIGNMENT).
    
    Creates:
    1. Coverage across chromosome (binned)
    2. Position statistics comparison
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with position features
    output_dir : Path
        Directory to save plots
    """
    print("\n2. Start and End Position Distributions (REQUIRED)")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Coverage across chr21 (binned)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Get position bin columns
    pos_bin_cols = [c for c in features_df.columns if c.startswith('pos_bin_')]
    
    if len(pos_bin_cols) > 0:
        # ALS samples
        ax = axes[0]
        als_samples = features_df[features_df['disease_status'] == 'als']
        for idx, row in als_samples.iterrows():
            coverage = [row[col] for col in pos_bin_cols]
            ax.plot(coverage, alpha=0.3, color='red')
        
        # Mean coverage
        als_mean_coverage = als_samples[pos_bin_cols].mean()
        ax.plot(als_mean_coverage.values, color='darkred', linewidth=2, label='Mean ALS')
        ax.set_ylabel('Coverage (%)', fontsize=12)
        ax.set_title('Fragment Start Position Distribution - ALS Samples', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Control samples
        ax = axes[1]
        ctrl_samples = features_df[features_df['disease_status'] == 'ctrl']
        for idx, row in ctrl_samples.iterrows():
            coverage = [row[col] for col in pos_bin_cols]
            ax.plot(coverage, alpha=0.3, color='blue')
        
        # Mean coverage
        ctrl_mean_coverage = ctrl_samples[pos_bin_cols].mean()
        ax.plot(ctrl_mean_coverage.values, color='darkblue', linewidth=2, label='Mean Control')
        ax.set_xlabel('Position Bin (across chr21)', fontsize=12)
        ax.set_ylabel('Coverage (%)', fontsize=12)
        ax.set_title('Fragment Start Position Distribution - Control Samples', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: position_distributions.png")
    
    # Plot 2: Position statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean start position
    ax = axes[0]
    if 'pos_mean_start' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='pos_mean_start',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('Mean Start Position', fontsize=12)
        ax.set_title('Fragment Start Position: ALS vs Control', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Batch')
    
    # Mean end position
    ax = axes[1]
    if 'pos_mean_end' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='pos_mean_end',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('Mean End Position', fontsize=12)
        ax.set_title('Fragment End Position: ALS vs Control', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Batch')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: position_statistics.png")


def plot_motif_distribution(features_df, output_dir):
    """
    Generate end motif distribution plots (REQUIRED FOR ASSIGNMENT).
    
    Creates:
    1. Top motifs comparison (ALS vs Control)
    2. Motif diversity comparison
    3. Heatmap of motif frequencies
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with motif features
    output_dir : Path
        Directory to save plots
    """
    print("\n3. End Motif Distribution (REQUIRED)")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get motif frequency columns (4-mers)
    motif_cols = [c for c in features_df.columns 
                  if c.startswith('motif_') and len(c) == 10]  # motif_XXXX
    
    if len(motif_cols) == 0:
        print("  ⚠️  No motif features found")
        return
    
    # Calculate mean frequency for each motif by disease status
    als_motifs = features_df[features_df['disease_status'] == 'als'][motif_cols].mean()
    ctrl_motifs = features_df[features_df['disease_status'] == 'ctrl'][motif_cols].mean()
    
    # Plot 1: Top 20 motifs comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get top 20 by overall frequency
    overall_freq = features_df[motif_cols].mean().sort_values(ascending=False).head(20)
    top_motifs = overall_freq.index
    
    # Prepare data for plotting
    x = np.arange(len(top_motifs))
    width = 0.35
    
    als_vals = [als_motifs[m] for m in top_motifs]
    ctrl_vals = [ctrl_motifs[m] for m in top_motifs]
    motif_names = [m.replace('motif_', '') for m in top_motifs]
    
    ax.bar(x - width/2, als_vals, width, label='ALS', color='red', alpha=0.7)
    ax.bar(x + width/2, ctrl_vals, width, label='Control', color='blue', alpha=0.7)
    
    ax.set_xlabel('End Motif (4-mer)', fontsize=12)
    ax.set_ylabel('Mean Frequency (%)', fontsize=12)
    ax.set_title('Top 20 End Motif Frequencies: ALS vs Control', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(motif_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'end_motif_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: end_motif_distribution.png")
    
    # Plot 2: Motif diversity and GC content
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Motif diversity
    ax = axes[0]
    if 'motif_diversity' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='motif_diversity',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('Motif Diversity (Shannon Entropy)', fontsize=12)
        ax.set_title('End Motif Diversity: ALS vs Control', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Batch')
    
    # Motif GC content
    ax = axes[1]
    if 'motif_gc_content' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='motif_gc_content',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('Motif GC Content (%)', fontsize=12)
        ax.set_title('End Motif GC Content: ALS vs Control', 
                    fontsize=13, fontweight='bold')
        ax.legend(title='Batch')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motif_diversity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: motif_diversity.png")
    
    # Plot 3: Heatmap of top motifs
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Select top 30 motifs
    top_30_motifs = overall_freq.head(30).index
    heatmap_data = features_df[['sample_id', 'disease_status'] + list(top_30_motifs)].copy()
    heatmap_data = heatmap_data.set_index('sample_id')
    
    # Sort by disease status
    heatmap_data = heatmap_data.sort_values('disease_status')
    
    # Create heatmap
    motif_data = heatmap_data[top_30_motifs]
    motif_names_clean = [m.replace('motif_', '') for m in top_30_motifs]
    
    sns.heatmap(motif_data.T, cmap='YlOrRd', cbar_kws={'label': 'Frequency (%)'},
               yticklabels=motif_names_clean, xticklabels=False, ax=ax)
    ax.set_ylabel('End Motif (4-mer)', fontsize=12)
    ax.set_xlabel('Samples (sorted by disease status)', fontsize=12)
    ax.set_title('Top 30 End Motif Frequencies Across Samples', 
                fontsize=13, fontweight='bold')
    
    # Add disease status indicator
    disease_colors = heatmap_data['disease_status'].map({'als': 'red', 'ctrl': 'blue'})
    for i, color in enumerate(disease_colors):
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, color=color, clip_on=False))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'motif_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: motif_heatmap.png")


def plot_methylation_analysis(features_df, output_dir):
    """
    Generate methylation analysis plots (REQUIRED FOR ASSIGNMENT).
    
    Creates:
    1. CpG methylation rate comparison
    2. Methylation variance comparison
    3. Per-read methylation statistics
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with methylation features
    output_dir : Path
        Directory to save plots
    """
    print("\n4. Methylation Analysis (REQUIRED)")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: CpG methylation rate
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Methylation Analysis: ALS vs Control', fontsize=14, fontweight='bold')
    
    # CpG methylation rate
    ax = axes[0, 0]
    if 'meth_cpg_rate' in features_df.columns:
        features_df['meth_cpg_pct'] = features_df['meth_cpg_rate'] * 100
        sns.violinplot(data=features_df, x='disease_status', y='meth_cpg_pct',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=11)
        ax.set_ylabel('CpG Methylation Rate (%)', fontsize=11)
        ax.set_title('CpG Methylation Rate', fontsize=12, fontweight='bold')
        ax.legend(title='Batch')
        
        # Add statistical test
        als_meth = features_df[features_df['disease_status'] == 'als']['meth_cpg_rate'].dropna()
        ctrl_meth = features_df[features_df['disease_status'] == 'ctrl']['meth_cpg_rate'].dropna()
        if len(als_meth) > 0 and len(ctrl_meth) > 0:
            stat, p_val = mannwhitneyu(als_meth, ctrl_meth)
            ax.text(0.98, 0.98, f'p = {p_val:.4f}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Methylation variance (per-read std)
    ax = axes[0, 1]
    if 'meth_std_per_read' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='meth_std_per_read',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=11)
        ax.set_ylabel('Methylation Std Dev (per read)', fontsize=11)
        ax.set_title('Methylation Variability', fontsize=12, fontweight='bold')
        ax.legend(title='Batch')
    
    # Mean methylation per read
    ax = axes[1, 0]
    if 'meth_mean_per_read' in features_df.columns:
        features_df['meth_mean_per_read_pct'] = features_df['meth_mean_per_read'] * 100
        sns.violinplot(data=features_df, x='disease_status', y='meth_mean_per_read_pct',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=11)
        ax.set_ylabel('Mean Methylation per Read (%)', fontsize=11)
        ax.set_title('Per-Read Methylation', fontsize=12, fontweight='bold')
        ax.legend(title='Batch')
    
    # Total CpG sites covered
    ax = axes[1, 1]
    if 'meth_total_cpg_sites' in features_df.columns:
        sns.violinplot(data=features_df, x='disease_status', y='meth_total_cpg_sites',
                      hue='batch', ax=ax, inner='box')
        ax.set_xlabel('Disease Status', fontsize=11)
        ax.set_ylabel('Total CpG Sites', fontsize=11)
        ax.set_title('CpG Coverage', fontsize=12, fontweight='bold')
        ax.legend(title='Batch')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'methylation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: methylation_analysis.png")
    
    # Plot 2: CHH methylation (QC check)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'meth_chh_rate' in features_df.columns:
        features_df['meth_chh_pct'] = features_df['meth_chh_rate'] * 100
        sns.violinplot(data=features_df, x='disease_status', y='meth_chh_pct',
                      hue='batch', ax=ax, inner='box')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
                  label='QC Threshold (1%)')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('CHH Methylation Rate (%)', fontsize=12)
        ax.set_title('CHH Methylation (Conversion Efficiency Check)', 
                    fontsize=13, fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'chh_methylation_qc.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: chh_methylation_qc.png")


def perform_pca(features_df, output_dir):
    """
    Perform PCA analysis and generate visualization plots.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix
    output_dir : Path
        Directory to save plots
    """
    print("\n5. PCA Analysis")
    print("-" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare feature matrix (exclude metadata)
    metadata_cols = ['sample_id', 'disease_status', 'batch', 'age']
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    
    X = features_df[feature_cols].copy()
    
    # Handle missing values (fill with median)
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'sample_id': features_df['sample_id'].values,
        'disease_status': features_df['disease_status'].values,
        'batch': features_df['batch'].values
    })
    
    # Plot 1: PC1 vs PC2 colored by disease
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # By disease status
    ax = axes[0]
    for disease, color in [('als', 'red'), ('ctrl', 'blue')]:
        mask = pca_df['disease_status'] == disease
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                  c=color, label=disease.upper(), s=100, alpha=0.7, edgecolors='black')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA: Colored by Disease Status', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # By batch
    ax = axes[1]
    for batch, color in [('discovery', 'green'), ('validation', 'orange')]:
        mask = pca_df['batch'] == batch
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                  c=color, label=batch.capitalize(), s=100, alpha=0.7, edgecolors='black')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('PCA: Colored by Batch', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: pca_analysis.png")
    
    # Plot 2: Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    explained_var = pca.explained_variance_ratio_[:10] * 100
    cumulative_var = np.cumsum(explained_var)
    
    ax.bar(range(1, 11), explained_var, alpha=0.6, label='Individual', color='steelblue')
    ax.plot(range(1, 11), cumulative_var, 'ro-', linewidth=2, label='Cumulative')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('PCA Variance Explained (Top 10 Components)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: pca_scree_plot.png")
    
    print(f"  Total variance explained by PC1-PC2: {pca.explained_variance_ratio_[:2].sum()*100:.1f}%")


def univariate_tests(features_df, output_file):
    """
    Perform univariate statistical tests for all features (ALS vs Control).
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix
    output_file : Path
        Path to save results CSV
    """
    print("\n6. Univariate Statistical Tests")
    print("-" * 70)
    
    # Get feature columns
    metadata_cols = ['sample_id', 'disease_status', 'batch', 'age']
    feature_cols = [c for c in features_df.columns if c not in metadata_cols]
    
    results = []
    
    for feature in feature_cols:
        als_vals = features_df[features_df['disease_status'] == 'als'][feature].dropna()
        ctrl_vals = features_df[features_df['disease_status'] == 'ctrl'][feature].dropna()
        
        if len(als_vals) > 0 and len(ctrl_vals) > 0:
            # Mann-Whitney U test
            stat, p_value = mannwhitneyu(als_vals, ctrl_vals, alternative='two-sided')
            
            # Effect size (difference in means)
            als_mean = als_vals.mean()
            ctrl_mean = ctrl_vals.mean()
            difference = als_mean - ctrl_mean
            
            # Fold change (avoid division by zero)
            if ctrl_mean != 0:
                fold_change = als_mean / ctrl_mean
            else:
                fold_change = np.nan
            
            results.append({
                'feature': feature,
                'als_mean': als_mean,
                'als_std': als_vals.std(),
                'ctrl_mean': ctrl_mean,
                'ctrl_std': ctrl_vals.std(),
                'difference': difference,
                'fold_change': fold_change,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # Create dataframe and sort by p-value
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    print(f"  ✓ Saved univariate test results: {output_file}")
    print(f"  Features tested: {len(results_df)}")
    print(f"  Significant features (p < 0.05): {results_df['significant'].sum()}")
    print(f"\nTop 10 most significant features:")
    print(results_df[['feature', 'p_value', 'difference', 'fold_change']].head(10).to_string(index=False))
    
    return results_df


def run_module_3():
    """
    Run complete Module 3: Exploratory Analysis & Visualization pipeline.
    
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    print("\n" + "=" * 70)
    print("MODULE 3: Exploratory Analysis & Visualization")
    print("=" * 70)
    
    # Ensure output directories exist
    VIZ_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EDA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load features from Module 2
    print(f"\nLoading features from: {ALL_FEATURES}")
    
    if not ALL_FEATURES.exists():
        raise FileNotFoundError(
            f"Feature file not found: {ALL_FEATURES}\n"
            f"Please run Module 2 first."
        )
    
    features_df = pd.read_csv(ALL_FEATURES)
    print(f"✓ Loaded features: {features_df.shape[0]} samples × {features_df.shape[1]} columns")
    
    # Generate required plots for assignment
    print("\n" + "=" * 70)
    print("GENERATING REQUIRED PLOTS FOR ASSIGNMENT")
    print("=" * 70)
    
    plot_fragment_distribution(features_df, VIZ_FIGURES_DIR)
    plot_position_distributions(features_df, VIZ_FIGURES_DIR)
    plot_motif_distribution(features_df, VIZ_FIGURES_DIR)
    plot_methylation_analysis(features_df, VIZ_FIGURES_DIR)
    
    # Additional exploratory analysis
    print("\n" + "=" * 70)
    print("ADDITIONAL EXPLORATORY ANALYSIS")
    print("=" * 70)
    
    perform_pca(features_df, EDA_FIGURES_DIR)
    univariate_results = univariate_tests(features_df, UNIVARIATE_TESTS)
    
    print("\n" + "=" * 70)
    print("MODULE 3 COMPLETE")
    print("=" * 70)
    print(f"\nRequired plots saved to: {VIZ_FIGURES_DIR}")
    print(f"EDA plots saved to: {EDA_FIGURES_DIR}")
    print(f"Statistical results saved to: {SUMMARY_TABLES_DIR}")
    print("\n" + "=" * 70 + "\n")
    
    return {
        'features': features_df,
        'univariate_results': univariate_results
    }


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 3 as a standalone script
    results = run_module_3()