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
# MODULE 2: Feature Extraction (GOLD STANDARD)
# ============================================================================
"""
Module 2 Overview:
    Extract features using gold-standard cfDNA fragmentomics and WGBS methods.
    
    FRAGMENTOMICS FEATURES (cfDNA standard):
    - Fragment size distribution (summary stats, ratios, size bins)
    - Fragment end motifs (top 20 most discriminative 4-mers, diversity)
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
    - fragmentomics_features.csv (~509 features per sample)
    - methylation_features.csv (~475 features per sample)
    - all_features.csv (merged, ~984 features per sample)
"""

# Feature extraction parameters
KMER_SIZE = 4  # 4-mers for end motifs
N_TOP_KMERS = 20  # Keep only top 20 most discriminative motifs (instead of all 256 combinations)

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
# MODULE 4: PCA & Feature Selection
# ============================================================================
"""
Module 4 Overview:
    Perform PCA separately on fragmentomics and methylation features.
    Select top discriminative features for classification.
    
    PCA Strategy:
    - Fragmentomics PCA: Fragment size + motifs + coverage (filter low variance)
    - Methylation PCA: Global + regional bins (filter low variance)
    
    Feature Selection:
    - Keep top variable features from each modality
    - Combine for final feature set
    
Input:
    - all_features.csv from Module 2
    
Output:
    - PCA plots (fragmentomics and methylation separately)
    - selected_features.csv (final feature set for classification)
    - feature_importance.csv (ranking of features)
"""

# PCA parameters
N_PCA_COMPONENTS = 10  # Number of PCs to compute
MIN_VARIANCE_THRESHOLD = 0.01  # Minimum variance to keep feature (1%)

# Feature selection for high-dimensional bins
MIN_SAMPLE_COVERAGE = 0.5  # Feature must have data in ≥50% of samples
N_TOP_VARIABLE_BINS = 100  # Keep top 100 most variable bins per modality

# Final feature selection
N_TOP_FEATURES_FRAGMENTOMICS = 50  # Top fragmentomics features
N_TOP_FEATURES_METHYLATION = 50    # Top methylation features

# Output files
PCA_FIGURES_DIR = FIGURES_DIR / "pca"
SELECTED_FEATURES_FILE = PROCESSED_DIR / "selected_features.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "tables" / "feature_importance.csv"

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