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
# MODULE 4: Classification
# ============================================================================
"""
Module 4 Overview:
    Binary classification using final validated approach.
    
    Approach:
    - 5 core fragmentomics features (selected through exploration)
    - Random Forest classifier (max_depth=3, n_estimators=500)
    - Train on discovery set (n=8)
    - Validate on validation set (n=14)
    
Input:
    - all_features.csv from Module 2
    
Output:
    - classification_metrics.csv (Precision, Recall, F1, AUC)
    - validation_predictions.csv
    - roc_curve.png, confusion_matrix.png
    - trained_rf_model.pkl
"""

# Final feature set (determined through exploratory analysis)
FINAL_FEATURES = [
    'frag_mean',
    'frag_pct_short',
    'frag_pct_long',
    'frag_ratio_short_long',
    'frag_pct_mononucleosomal'
]

# Random Forest parameters (validated in exploration)
RF_PARAMS_FINAL = {
    'n_estimators': 500,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# Output paths
CLASSIFICATION_DIR = RESULTS_DIR / 'classification'
CLASSIFICATION_METRICS_FILE = CLASSIFICATION_DIR / 'classification_metrics.csv'
VALIDATION_PREDICTIONS_FILE = CLASSIFICATION_DIR / 'validation_predictions.csv'
TRAINED_RF_MODEL_FILE = CLASSIFICATION_DIR / 'trained_rf_model.pkl'
ROC_CURVE_FILE = FIGURES_DIR / 'classification' / 'roc_curve.png'
CONFUSION_MATRIX_FILE = FIGURES_DIR / 'classification' / 'confusion_matrix.png'


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