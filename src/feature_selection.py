"""
Module 4: PCA & Feature Selection

Purpose:
    Perform PCA and feature selection on DISCOVERY SET ONLY (n=8).
    Compare 4 feature selection strategies to find best discriminative features.
    
Four Analyses:
    1. High-Variance Fragmentomics (K-mers)
    2. Discriminative Fragmentomics (K-mers)
    3. High-Variance Methylation (Regional bins)
    4. Discriminative Methylation (Regional bins)
    
Strategy:
    - Use only Discovery set (n=8: 4 ALS, 4 Control) for all feature selection
    - Validation set (n=14) is locked until Module 6
    - Filter features by sample coverage (present in ≥25%)
    - Select top 30 features per analysis
    - Perform PCA to visualize separation
    - Compare analyses to select best feature set for classification

Input:
    - all_features.csv from Module 2

Output:
    - 4 PCA plots, assess disease separation
    - 4 feature ranking tables
    - Final selected feature set for Module 5

Usage:
    As a script:
        python src/feature_selection.py
    
    In a notebook:
        from src.feature_selection import run_module_4
        results = run_module_4()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    ALL_FEATURES,
    PCA_FIGURES_DIR,
    FEATURE_RANKINGS_DIR,
    SELECTED_FEATURES_FILE,
    PCA_COMPARISON_FILE,
    DISCOVERY_BATCH,
    VALIDATION_BATCH,
    MIN_SAMPLE_COVERAGE,
    MIN_VARIANCE_THRESHOLD,
    N_TOP_FEATURES_PER_ANALYSIS,
    N_PCA_COMPONENTS,
    FRAG_HIGHVAR_FEATURES,
    FRAG_DISCRIM_FEATURES,
    METH_HIGHVAR_FEATURES,
    METH_DISCRIM_FEATURES,
    RESULTS_DIR
)


# ============================================================================
# Data Preparation
# ============================================================================

def load_and_split_data(filepath):
    """
    Load features and split into discovery/validation.
    
    CRITICAL: Only discovery set is used for feature selection!
    
    Parameters
    ----------
    filepath : Path
        Path to all_features.csv
        
    Returns
    -------
    tuple
        (discovery_df, validation_df, all_df)
    """
    print("\nLoading and splitting data:")
    print("-" * 70)
    
    # Load all features
    df = pd.read_csv(filepath)
    print(f"Total samples: {len(df)}")
    
    # Split by batch
    discovery_df = df[df['batch'] == DISCOVERY_BATCH].copy()
    validation_df = df[df['batch'] == VALIDATION_BATCH].copy()
    
    print(f"\nDiscovery set: {len(discovery_df)} samples")
    print(f"  ALS: {len(discovery_df[discovery_df['disease_status'] == 'als'])}")
    print(f"  Control: {len(discovery_df[discovery_df['disease_status'] == 'ctrl'])}")
    
    print(f"\nValidation set: {len(validation_df)} samples (LOCKED until Module 6)")
    print(f"  ALS: {len(validation_df[validation_df['disease_status'] == 'als'])}")
    print(f"  Control: {len(validation_df[validation_df['disease_status'] == 'ctrl'])}")
    
    return discovery_df, validation_df, df


# ============================================================================
# Feature Filtering
# ============================================================================

def filter_features_by_coverage(df, feature_cols, min_coverage=MIN_SAMPLE_COVERAGE):
    """
    Keep only features with sufficient sample coverage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Discovery set (n=8)
    feature_cols : list
        Feature columns to filter
    min_coverage : float
        Minimum proportion of samples with data (0-1)
        
    Returns
    -------
    list
        Filtered feature columns
    """
    n_samples = len(df)
    min_samples = int(np.ceil(n_samples * min_coverage))
    
    # Count non-NaN values per feature
    coverage = df[feature_cols].notna().sum()
    
    # Keep features with sufficient coverage
    sufficient_coverage = coverage[coverage >= min_samples].index.tolist()
    
    print(f"  Coverage filter (≥{min_samples}/{n_samples} samples):")
    print(f"    Input: {len(feature_cols)} features")
    print(f"    Output: {len(sufficient_coverage)} features")
    print(f"    Removed: {len(feature_cols) - len(sufficient_coverage)} features")
    
    return sufficient_coverage


def filter_low_variance(df, feature_cols, min_var=MIN_VARIANCE_THRESHOLD):
    """
    Remove features with very low variance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Discovery set
    feature_cols : list
        Features to filter
    min_var : float
        Minimum variance threshold
        
    Returns
    -------
    list
        Features with sufficient variance
    """
    variances = df[feature_cols].var()
    high_var = variances[variances >= min_var].index.tolist()
    
    print(f"  Variance filter (var ≥ {min_var}):")
    print(f"    Input: {len(feature_cols)} features")
    print(f"    Output: {len(high_var)} features")
    print(f"    Removed: {len(feature_cols) - len(high_var)} features")
    
    return high_var


# ============================================================================
# Feature Selection Strategies
# ============================================================================

def select_high_variance_features(df, feature_cols, n_top=N_TOP_FEATURES_PER_ANALYSIS):
    """
    Select features with highest variance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Discovery set
    feature_cols : list
        Features to rank
    n_top : int
        Number of top features to select
        
    Returns
    -------
    pd.DataFrame
        Ranking table with variance scores
    """
    print(f"\n  Selecting top {n_top} high-variance features...")
    
    results = []
    
    for feat in feature_cols:
        vals = df[feat].dropna()
        
        if len(vals) > 1:
            variance = vals.var()
            
            results.append({
                'feature': feat,
                'variance': variance,
                'mean': vals.mean(),
                'std': vals.std(),
                'n_samples': len(vals)
            })
    
    # Rank by variance
    results_df = pd.DataFrame(results).sort_values('variance', ascending=False)
    
    print(f"    Ranked {len(results_df)} features by variance")
    print(f"    Top feature: {results_df.iloc[0]['feature']} (var={results_df.iloc[0]['variance']:.4f})")
    
    return results_df


def select_discriminative_features(df, feature_cols, n_top=N_TOP_FEATURES_PER_ANALYSIS):
    """
    Select features that discriminate ALS vs Control (Mann-Whitney U test).
    
    Parameters
    ----------
    df : pd.DataFrame
        Discovery set
    feature_cols : list
        Features to test
    n_top : int
        Number of top features to select
        
    Returns
    -------
    pd.DataFrame
        Ranking table with p-values and effect sizes
    """
    print(f"\n  Selecting top {n_top} discriminative features...")
    
    results = []
    
    for feat in feature_cols:
        als_vals = df[df['disease_status'] == 'als'][feat].dropna()
        ctrl_vals = df[df['disease_status'] == 'ctrl'][feat].dropna()
        
        if len(als_vals) >= 2 and len(ctrl_vals) >= 2:
            # Mann-Whitney U test
            stat, p_val = mannwhitneyu(als_vals, ctrl_vals)
            
            # Effect size (difference in means)
            effect_size = abs(als_vals.mean() - ctrl_vals.mean())
            
            results.append({
                'feature': feat,
                'p_value': p_val,
                'effect_size': effect_size,
                'als_mean': als_vals.mean(),
                'ctrl_mean': ctrl_vals.mean(),
                'als_std': als_vals.std(),
                'ctrl_std': ctrl_vals.std(),
                'n_als': len(als_vals),
                'n_ctrl': len(ctrl_vals)
            })
    
    # Rank by p-value
    results_df = pd.DataFrame(results).sort_values('p_value')
    
    n_sig = (results_df['p_value'] < 0.05).sum()
    
    print(f"    Ranked {len(results_df)} features by discriminative power")
    print(f"    Significant (p<0.05): {n_sig}/{len(results_df)}")
    print(f"    Top feature: {results_df.iloc[0]['feature']} (p={results_df.iloc[0]['p_value']:.4f})")
    
    return results_df


# ============================================================================
# PCA Analysis
# ============================================================================

def perform_pca_analysis(df, feature_cols, analysis_name, output_dir):
    """
    Perform PCA and generate visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Discovery set (n=8)
    feature_cols : list
        Features to use for PCA
    analysis_name : str
        Name of analysis (for plot title)
    output_dir : Path
        Output directory for plot
        
    Returns
    -------
    dict
        PCA results including variance explained
    """
    print(f"\n  Performing PCA...")
    
    # Prepare data
    X = df[feature_cols].copy()
    
    # Handle missing values (impute with median)
    X = X.fillna(X.median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA (max components = n_samples)
    n_components = min(N_PCA_COMPONENTS, len(df), len(feature_cols))
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Store results in dataframe
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=df.index
    )
    pca_df['disease_status'] = df['disease_status'].values
    pca_df['sample_id'] = df['sample_id'].values
    
    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    pc1_var = var_explained[0] * 100
    pc2_var = var_explained[1] * 100 if len(var_explained) > 1 else 0
    total_var = (var_explained[0] + var_explained[1]) * 100 if len(var_explained) > 1 else pc1_var
    
    print(f"    PCA computed: {len(feature_cols)} features → {n_components} PCs")
    print(f"    PC1 variance: {pc1_var:.1f}%")
    print(f"    PC2 variance: {pc2_var:.1f}%")
    print(f"    PC1+PC2 total: {total_var:.1f}%")
    
    # Generate plot
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: PC1 vs PC2 colored by disease
    ax = axes[0]
    for status, color, marker in [('als', 'red', 'o'), ('ctrl', 'blue', 's')]:
        mask = pca_df['disease_status'] == status
        ax.scatter(pca_df.loc[mask, 'PC1'], pca_df.loc[mask, 'PC2'],
                  c=color, label=status.upper(), alpha=0.8, s=150, 
                  edgecolors='black', linewidths=1.5, marker=marker)
    
    ax.set_xlabel(f'PC1 ({pc1_var:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pc2_var:.1f}% variance)', fontsize=12)
    ax.set_title(f'{analysis_name}\nDisease Status', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add sample labels
    for idx, row in pca_df.iterrows():
        ax.annotate(row['sample_id'][-4:], 
                   (row['PC1'], row['PC2']),
                   fontsize=7, alpha=0.6, 
                   xytext=(3, 3), textcoords='offset points')
    
    # Plot 2: Scree plot
    ax = axes[1]
    pcs = np.arange(1, len(var_explained) + 1)
    ax.plot(pcs, var_explained * 100, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Scree Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pcs)
    
    # Add cumulative variance line
    cumvar = np.cumsum(var_explained) * 100
    ax2 = ax.twinx()
    ax2.plot(pcs, cumvar, 's--', linewidth=1.5, markersize=6, 
            color='orange', alpha=0.7, label='Cumulative')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save plot
    filename = analysis_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_file = output_dir / f'pca_{filename}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_file.name}")
    
    # Return results
    return {
        'analysis': analysis_name,
        'n_features': len(feature_cols),
        'n_components': n_components,
        'pc1_variance': pc1_var,
        'pc2_variance': pc2_var,
        'total_variance': total_var,
        'all_variance': var_explained,
        'pca_df': pca_df
    }


# ============================================================================
# Main Analysis Functions
# ============================================================================

def analyze_fragmentomics_high_variance(discovery_df):
    """
    Analysis 1: High-variance K-mer features.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: High-Variance Fragmentomics (K-mers)")
    print("=" * 70)
    
    # Get k-mer columns
    kmer_cols = [c for c in discovery_df.columns if c.startswith('kmer_')]
    
    print(f"\nStarting features: {len(kmer_cols)} k-mers")
    
    # Filter by coverage
    kmer_cols = filter_features_by_coverage(discovery_df, kmer_cols)
    
    # Filter by variance
    kmer_cols = filter_low_variance(discovery_df, kmer_cols)
    
    # Select high-variance features
    rankings = select_high_variance_features(discovery_df, kmer_cols, N_TOP_FEATURES_PER_ANALYSIS)
    
    # Save rankings
    FEATURE_RANKINGS_DIR.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(FRAG_HIGHVAR_FEATURES, index=False)
    print(f"\n  ✓ Saved rankings to: {FRAG_HIGHVAR_FEATURES.name}")
    
    # Select top features for PCA
    top_features = rankings.head(N_TOP_FEATURES_PER_ANALYSIS)['feature'].tolist()
    
    # Perform PCA
    pca_results = perform_pca_analysis(
        discovery_df, 
        top_features,
        'Fragmentomics: High-Variance K-mers',
        PCA_FIGURES_DIR
    )
    
    return rankings, pca_results


def analyze_fragmentomics_discriminative(discovery_df):
    """
    Analysis 2: Discriminative K-mer features (ALS vs Control).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Discriminative Fragmentomics (K-mers)")
    print("=" * 70)
    
    # Get k-mer columns
    kmer_cols = [c for c in discovery_df.columns if c.startswith('kmer_')]
    
    print(f"\nStarting features: {len(kmer_cols)} k-mers")
    
    # Filter by coverage
    kmer_cols = filter_features_by_coverage(discovery_df, kmer_cols)
    
    # Filter by variance
    kmer_cols = filter_low_variance(discovery_df, kmer_cols)
    
    # Select discriminative features
    rankings = select_discriminative_features(discovery_df, kmer_cols, N_TOP_FEATURES_PER_ANALYSIS)
    
    # Save rankings
    rankings.to_csv(FRAG_DISCRIM_FEATURES, index=False)
    print(f"\n  ✓ Saved rankings to: {FRAG_DISCRIM_FEATURES.name}")
    
    # Select top features for PCA
    top_features = rankings.head(N_TOP_FEATURES_PER_ANALYSIS)['feature'].tolist()
    
    # Perform PCA
    pca_results = perform_pca_analysis(
        discovery_df,
        top_features,
        'Fragmentomics: Discriminative K-mers',
        PCA_FIGURES_DIR
    )
    
    return rankings, pca_results


def analyze_methylation_high_variance(discovery_df):
    """
    Analysis 3: High-variance regional methylation bins.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: High-Variance Methylation (Regional Bins)")
    print("=" * 70)
    
    # Get regional methylation columns
    meth_cols = [c for c in discovery_df.columns if c.startswith('regional_meth_bin_')]
    
    print(f"\nStarting features: {len(meth_cols)} regional methylation bins")
    
    # Filter by coverage
    meth_cols = filter_features_by_coverage(discovery_df, meth_cols)
    
    # Filter by variance
    meth_cols = filter_low_variance(discovery_df, meth_cols)
    
    # Select high-variance features
    rankings = select_high_variance_features(discovery_df, meth_cols, N_TOP_FEATURES_PER_ANALYSIS)
    
    # Save rankings
    rankings.to_csv(METH_HIGHVAR_FEATURES, index=False)
    print(f"\n  ✓ Saved rankings to: {METH_HIGHVAR_FEATURES.name}")
    
    # Select top features for PCA
    top_features = rankings.head(N_TOP_FEATURES_PER_ANALYSIS)['feature'].tolist()
    
    # Perform PCA
    pca_results = perform_pca_analysis(
        discovery_df,
        top_features,
        'Methylation: High-Variance Regional Bins',
        PCA_FIGURES_DIR
    )
    
    return rankings, pca_results


def analyze_methylation_discriminative(discovery_df):
    """
    Analysis 4: Discriminative regional methylation bins (ALS vs Control).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Discriminative Methylation (Regional Bins)")
    print("=" * 70)
    
    # Get regional methylation columns
    meth_cols = [c for c in discovery_df.columns if c.startswith('regional_meth_bin_')]
    
    print(f"\nStarting features: {len(meth_cols)} regional methylation bins")
    
    # Filter by coverage
    meth_cols = filter_features_by_coverage(discovery_df, meth_cols)
    
    # Filter by variance
    meth_cols = filter_low_variance(discovery_df, meth_cols)
    
    # Select discriminative features
    rankings = select_discriminative_features(discovery_df, meth_cols, N_TOP_FEATURES_PER_ANALYSIS)
    
    # Save rankings
    rankings.to_csv(METH_DISCRIM_FEATURES, index=False)
    print(f"\n  ✓ Saved rankings to: {METH_DISCRIM_FEATURES.name}")
    
    # Select top features for PCA
    top_features = rankings.head(N_TOP_FEATURES_PER_ANALYSIS)['feature'].tolist()
    
    # Perform PCA
    pca_results = perform_pca_analysis(
        discovery_df,
        top_features,
        'Methylation: Discriminative Regional Bins',
        PCA_FIGURES_DIR
    )
    
    return rankings, pca_results


# ============================================================================
# Calculate Separation Metric
# ============================================================================

def calculate_separation_score(pca_df):
    """
    Calculate how well PC1+PC2 separates disease groups.
    
    Uses silhouette score to measure cluster separation.
    Higher score = better separation between ALS and Control.
    
    Parameters
    ----------
    pca_df : pd.DataFrame
        PCA results with PC1, PC2, and disease_status columns
    
    Returns
    -------
    float
        Silhouette score (-1 to 1, higher is better)
    """
    from sklearn.metrics import silhouette_score
    
    # Encode disease status as binary labels
    labels = (pca_df['disease_status'] == 'als').astype(int).values
    
    # Get PC1 and PC2 coordinates
    X = pca_df[['PC1', 'PC2']].values
    
    # Calculate silhouette score (measures cluster separation)
    score = silhouette_score(X, labels)
    
    return score

# ============================================================================
# Feature Set Utilities
# ============================================================================

def get_top_features_from_rankings(rankings, n_top):
    """
    Extract top-N feature names from a ranking DataFrame.
    """
    return rankings.head(n_top)['feature'].tolist()

# ============================================================================
# Comparison & Selection
# ============================================================================

def combine_feature_sets(discovery_df, all_rankings):
    """
    Combine top features from all 4 analyses into a single feature table.

    Returns
    -------
    pd.DataFrame
        Discovery-only feature matrix for classification
    dict
        Dictionary of feature sets by category
    """
    print("\n" + "=" * 70)
    print("COMBINING FEATURE SETS FOR CLASSIFICATION")
    print("=" * 70)

    # Extract top features from each analysis
    feature_sets = {
        'frag_highvar': get_top_features_from_rankings(all_rankings[0], N_TOP_FEATURES_PER_ANALYSIS),
        'frag_discriminative': get_top_features_from_rankings(all_rankings[1], N_TOP_FEATURES_PER_ANALYSIS),
        'meth_highvar': get_top_features_from_rankings(all_rankings[2], N_TOP_FEATURES_PER_ANALYSIS),
        'meth_discriminative': get_top_features_from_rankings(all_rankings[3], N_TOP_FEATURES_PER_ANALYSIS),
    }

    # Flatten and deduplicate
    all_features = sorted(set(
        feat for feats in feature_sets.values() for feat in feats
    ))

    print(f"\nFeature set summary:")
    for k, v in feature_sets.items():
        print(f"  {k}: {len(v)} features")

    print(f"\nTotal unique features: {len(all_features)}")

    # Metadata
    metadata_cols = ['sample_id', 'disease_status', 'batch', 'age']

    # Build final discovery-only table
    final_df = discovery_df[metadata_cols + all_features].copy()

    # Save combined feature table
    final_df.to_csv(SELECTED_FEATURES_FILE, index=False)

    print(f"\n✓ Combined feature table saved to: {SELECTED_FEATURES_FILE}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Samples: {len(final_df)} (DISCOVERY ONLY)")

    return final_df, feature_sets


# ============================================================================
# Main Pipeline
# ============================================================================

def run_module_4():
    """
    Run complete Module 4: PCA & Feature Selection pipeline.
    
    Returns
    -------
    dict
        Results including selected features, rankings, and PCA results
    """
    print("\n" + "=" * 70)
    print("MODULE 4: PCA & Feature Selection")
    print("=" * 70)
    print("\nCRITICAL: Using DISCOVERY SET ONLY (n=8)")
    print("Validation set remains locked until Module 6")
    
    # Load and split data
    discovery_df, validation_df, all_df = load_and_split_data(ALL_FEATURES)
    
    # Run 4 analyses
    frag_hv_rankings, frag_hv_pca = analyze_fragmentomics_high_variance(discovery_df)
    frag_disc_rankings, frag_disc_pca = analyze_fragmentomics_discriminative(discovery_df)
    meth_hv_rankings, meth_hv_pca = analyze_methylation_high_variance(discovery_df)
    meth_disc_rankings, meth_disc_pca = analyze_methylation_discriminative(discovery_df)
    
    # Collect results
    all_rankings = [frag_hv_rankings, frag_disc_rankings, meth_hv_rankings, meth_disc_rankings]
    all_pca_results = [frag_hv_pca, frag_disc_pca, meth_hv_pca, meth_disc_pca]
    
    # Compare and select best
    final_features, feature_sets = combine_feature_sets(
        discovery_df, all_rankings
    )
    
    print("\n" + "=" * 70)
    print("MODULE 4 COMPLETE")
    print("=" * 70)
    print(f"\n✓ Generated 4 PCA plots in: {PCA_FIGURES_DIR}")
    print(f"✓ Selected features from ALL 4 analyses")
    print(f"✓ Total features for classification: {final_features.shape[1] - 4}")
    print(f"✓ Ready for Module 5: Classification")
    print("\n" + "=" * 70 + "\n")
    
    return {
        'discovery_df': discovery_df,
        'validation_df': validation_df,
        'final_features': final_features,
        'feature_sets': feature_sets,
        'all_rankings': all_rankings,
        'all_pca_results': all_pca_results
    }

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 4 as a standalone script
    results = run_module_4()
    
    print(f"\nBest feature set: {results['best_analysis']}")
    print(f"Selected features shape: {results['final_features'].shape}")
    print(f"\nReady for classification in Module 5!")