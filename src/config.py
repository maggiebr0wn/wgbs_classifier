"""
Configuration file for wgbs_classifier pipeline.
Contains paths, constants, and filtering parameters.
"""

import re
from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================================
# MODULE 0: Setup & Data Loading
# ============================================================================
"""
Module 0 Overview:
    - Load sample metadata from CSV
    - Verify BAM files exist for all samples
    - Create sample manifest for downstream modules
    
Input Requirements:
    - sample_metadata.csv in data/metadata/
    - BAM files (*.bam and *.bam.bai) in data/raw/
    
Output:
    - sample_manifest.csv in data/processed/
"""

# Input paths
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "celfie_cfDNA_ss.csv"
BAM_DIR = PROJECT_ROOT / "data" / "raw"
BAM_PATTERN = r"^{run_id}.*\.bam$"  # Will be formatted with actual run_id

# Output paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_MANIFEST = PROCESSED_DIR / "sample_manifest.csv"


# ============================================================================
# MODULE 1: Quality Control & Filtering
# ============================================================================
"""
Module 1 Overview:
    - Calculate BAM-level statistics (reads, mapping quality, etc.)
    - Check bisulfite conversion efficiency
    - Assess batch effects on QC metrics
    - Define filtering criteria for downstream analysis
    
Input:
    - sample_manifest.csv from Module 0
    - BAM files
    
Output:
    - qc_metrics.csv in data/processed/
    - qc_report.txt in results/
    - QC plots in results/figures/qc/
"""

# Output paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
QC_FIGURES_DIR = FIGURES_DIR / "qc"

QC_METRICS = PROCESSED_DIR / "qc_metrics.csv"
QC_REPORT = RESULTS_DIR / "qc_report.txt"

# QC Parameters
CHROMOSOME = "chr21"  # Chromosome to analyze
MIN_MAPQ = 20  # Minimum mapping quality for filtering
MIN_FRAGMENT_SIZE = 50  # Minimum fragment size (bp)
MAX_FRAGMENT_SIZE = 1000  # Maximum fragment size (bp)
BISULFITE_CONVERSION_THRESHOLD = 0.99  # Minimum conversion efficiency (99%)
CHH_METHYLATION_THRESHOLD = 0.01  # Maximum CHH methylation (1%)
QC_SAMPLE_SIZE = 10000  # Number of reads to sample for conversion check


# ============================================================================
# MODULE 2: Feature Extraction
# ============================================================================
"""
Module 2 Overview:
    Extract features using gold-standard cfDNA fragmentomics and WGBS methods.
    
    FRAGMENTOMICS FEATURES (cfDNA standard):
    - Fragment size distribution (summary stats, ratios, size bins)
    - Fragment end motifs (ALL 256 4-mer proportions for assignment)
    - Coverage profile (100 kb bins across chr21)
    
    METHYLATION FEATURES (WGBS standard):
    - Global CpG methylation (mean, variance, distribution)
    - Regional methylation (100 kb bins with ≥20 CpG sites)
    
    Based on published methods:
    - Fragmentomics: Cristiano et al. 2019 (Nature), Snyder et al. 2016 (Cell)
    - Methylation: Loyfer et al. 2023 (Nature), Sun et al. 2015 (Genome Biol)
    
Input:
    - sample_manifest.csv from Module 0
    - BAM files
    
Output:
    - fragmentomics_features.csv (~745 features: 20 stats + 5 categories + 2 summary + 256 k-mers + 468 coverage bins)
    - methylation_features.csv (~475 features: 8 global + 467 regional)
    - all_features.csv (merged, ~1220 features per sample)
"""

# Feature extraction parameters
KMER_SIZE = 4  # 4-mers for end motifs
EXTRACT_ALL_KMERS = True  # Extract all 256 4-mer combinations (not just top 20)

# Fragment size categories (based on nucleosome biology)
FRAGMENT_SIZE_CATEGORIES = {
    'very_short': (50, 100),       # Nucleosome-free DNA
    'short': (100, 150),           # Short nucleosomal
    'mononucleosomal': (150, 220), # Mono-nucleosome (peak ~167 bp)
    'dinucleosomal': (220, 400),   # Di-nucleosome
    'long': (400, 1000)            # Tri+ nucleosome or genomic DNA
}

# Genomic binning (GOLD STANDARD - 100 kb resolution)
FRAGMENTOMICS_BIN_SIZE = 100_000  # 100 kb bins for coverage analysis
METHYLATION_BIN_SIZE = 100_000    # 100 kb bins for regional methylation
MIN_CPG_PER_BIN = 20              # Minimum CpG sites for robust methylation estimate

# For chr21 (~46.7 Mb with 100 kb bins):
#   Expected bins: ~467
#   Expected CpGs per bin: ~1,000-10,000 (robust estimates)

# Methylation distribution thresholds
HIGHLY_METHYLATED_THRESHOLD = 0.8   # CpG methylation > 80%
LOWLY_METHYLATED_THRESHOLD = 0.2    # CpG methylation < 20%

# Output files
FRAGMENTOMICS_FEATURES = PROCESSED_DIR / "fragmentomics_features.csv"
METHYLATION_FEATURES_FILE = PROCESSED_DIR / "methylation_features.csv"
ALL_FEATURES = PROCESSED_DIR / "all_features.csv"

# ============================================================================
# MODULE 3: Visualization & Batch Effects
# ============================================================================
"""
Module 3 Overview:
    Generate required assignment plots and assess batch effects.
    
    REQUIRED PLOTS (Assignment):
    1. Fragment length distribution
    2. Start/end position distributions
    3. End motif distribution
    4. Methylation analysis
    
    BATCH EFFECTS:
    - Test if Discovery vs Validation batches differ
    - Check for confounding between batch and disease
    
Input:
    - all_features.csv from Module 2
    
Output:
    - Required plots in results/figures/required_plots/
    - Batch effect summary table
"""

# Figure output directory
REQUIRED_PLOTS_DIR = FIGURES_DIR / "required_plots"

# Batch effect output
BATCH_EFFECTS_FILE = RESULTS_DIR / "tables" / "batch_effects_summary.csv"

# ============================================================================
# MODULE 4: Feature Selection & Model Training
# ============================================================================
"""
Module 4 Overview (REVISED):
    Robust feature selection for n=8 discovery set using effect sizes and LASSO.
    
    NEW Strategy:
    1. Use ALL fragmentomics summary features (~20 features, NOT k-mers)
       - Fragment size distributions, percentiles, ratios
       - Fragment size category proportions
       
    2. Aggregate methylation to larger regions (500kb bins, ~93 bins)
       - More robust estimates with more CpGs per bin
       - Rank by effect size (Cohen's d)
       - Select top 20-30 regions
       
    3. Feature selection via effect size ranking
       - No p-values (meaningless with n=4 vs n=4)
       - Calculate Cohen's d for all features
       - Rank by absolute effect size
       
    4. Final model: LASSO with LOO-CV
       - Strong L1 regularization
       - Tune penalty via leave-one-out cross-validation
       - Let LASSO select final feature subset (likely 5-15 features)
    
    Rationale:
    - With n=8, we need stable, interpretable features
    - Summary statistics > individual k-mers
    - Larger bins > small bins
    - Effect sizes > p-values
    - Regularization > manual selection
    
Input:
    - all_features.csv from Module 2
    
Output:
    - fragmentomics_summary_features.csv (all frag summaries)
    - methylation_aggregated_features.csv (500kb bins)
    - feature_rankings_by_effect_size.csv
    - selected_features_for_training.csv (final feature set)
    - loo_cv_results.csv (LOO performance for model selection)
    - trained_lasso_model.pkl
"""

# Discovery/Validation split
DISCOVERY_BATCH = 'discovery'
VALIDATION_BATCH = 'validation'

# Feature filtering (for small n=8)
MIN_SAMPLE_COVERAGE = 0.25  # Feature must have data in ≥25% of samples (≥2/8)
MIN_VARIANCE_THRESHOLD = 0.0001  # Minimum variance to keep feature (very low threshold)

# Methylation aggregation
METHYLATION_AGGREGATION_SIZE = 500_000  # 500kb bins (~93 bins for chr21)
MIN_CPG_PER_AGGREGATED_BIN = 50  # Require ≥50 CpGs for aggregated bin

# Feature selection strategy
N_TOP_METHYLATION_REGIONS = 10  # Top 30 methylation regions by effect size
MIN_EFFECT_SIZE = 0.5  # Minimum Cohen's d to consider (small effect)

# LASSO regularization (LOO-CV tuning)
# IMPORTANT: With n=8, we need VERY strong regularization to prevent overfitting
# Lower C = stronger penalty = more regularization
LASSO_C_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # Added stronger penalties
LASSO_MAX_ITER = 5000  # Increase for convergence

# Output files
FEATURE_SELECTION_DIR = RESULTS_DIR / "feature_selection"
FEATURE_RANKINGS_FILE = FEATURE_SELECTION_DIR / "feature_rankings_by_effect_size.csv"
FRAGMENTOMICS_SUMMARY_FILE = FEATURE_SELECTION_DIR / "fragmentomics_summary_features.csv"
METHYLATION_AGGREGATED_FILE = FEATURE_SELECTION_DIR / "methylation_aggregated_features.csv"
SELECTED_FEATURES_FILE = FEATURE_SELECTION_DIR / "selected_features_for_training.csv"
LOO_CV_RESULTS_FILE = FEATURE_SELECTION_DIR / "loo_cv_results.csv"
TRAINED_LASSO_MODEL = FEATURE_SELECTION_DIR / "trained_lasso_model.pkl"

# Legacy outputs (for backward compatibility)
PCA_FIGURES_DIR = FIGURES_DIR / "pca"
FEATURE_RANKINGS_DIR = RESULTS_DIR / "tables" / "feature_rankings"
PCA_COMPARISON_FILE = RESULTS_DIR / "tables" / "pca_comparison_summary.csv"

# ============================================================================
# MODULE 5 & 6: Classification
# ============================================================================

"""
Modules 5 & 6 Overview:

Input:
    
Output:

"""

# Target column
TARGET_COL = 'disease_status'
POSITIVE_CLASS = 'als'

# Metadata columns (excluded from training)
METADATA_COLS = ['sample_id', 'disease_status', 'batch', 'age']

# Model choice: 'logistic' or 'random_forest'
CLASSIFIER_TYPE = 'logistic'

# Logistic Regression params
LOGREG_PARAMS = {
    'penalty': 'l1',        # L1 for feature sparsity
    'solver': 'liblinear',
    'C': 1.0,
    'max_iter': 1000,
    'class_weight': 'balanced'
}

# Random Forest params (conservative for small n)
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'min_samples_leaf': 2,
    'random_state': 42,
    'class_weight': 'balanced'
}

# Output paths
CLASSIFIER_DIR = RESULTS_DIR / 'classification'
TRAINED_MODEL_FILE = CLASSIFIER_DIR / 'trained_model.joblib'
TRAIN_FEATURES_FILE = CLASSIFIER_DIR / 'training_features.csv'

# ============================================================================
# Module 4 & 6 outputs
# ============================================================================
TRAINING_PCA_FILE = RESULTS_DIR / 'training_pca_features.csv'    # Module 4 → Module 5
VALIDATION_FEATURES_FILE = RESULTS_DIR / 'validation_features.csv'  # Module 4 → Module 6
PCA_PICKLE = RESULTS_DIR / 'discovery_pca_objects.pkl'           # Module 4 → Module 6


# ============================================================================
# Test Configuration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("wgbs_classifier CONFIGURATION")
    print("=" * 60)
    
    print(f"\nProject root: {PROJECT_ROOT}")
    
    print(f"\n--- MODULE 0: Setup & Data Loading ---")
    print(f"Input metadata: {METADATA_FILE}")
    print(f"BAM directory: {BAM_DIR}")
    print(f"Output manifest: {SAMPLE_MANIFEST}")
    
    # Check paths
    print(f"\nPath verification:")
    print(f"  Metadata exists: {METADATA_FILE.exists()}")
    print(f"  BAM directory exists: {BAM_DIR.exists()}")
    
    # Create output directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory created/exists: {PROCESSED_DIR.exists()}")
    
    print("\n" + "=" * 60)