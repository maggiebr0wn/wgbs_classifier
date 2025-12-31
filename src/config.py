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
    - Extract fragment size features (full distributions, size bins)
    - Extract fragment start/end positions across chr21
    - Extract end motif features (4-mer frequencies from read ends)
    - Extract methylation features (CpG methylation from XM tags)
    
Input:
    - sample_manifest.csv from Module 0
    - BAM files
    
Output:
    - all_features.csv in data/processed/
    - Individual feature files for inspection
"""

# Feature extraction parameters
KMER_SIZE = 4  # Size of k-mers for end motif analysis (4-mers = 256 features)

# Fragment size bins for feature engineering
FRAGMENT_SIZE_BINS = {
    'very_short': (50, 100),      # < 100 bp
    'short': (100, 150),          # 100-150 bp
    'nucleosomal': (150, 200),    # 150-200 bp (mono-nucleosome)
    'dinucleosomal': (200, 400),  # 200-400 bp (di-nucleosome)
    'long': (400, 1000)           # > 400 bp
}

# Position binning (divide chr21 into bins for coverage features)
N_POSITION_BINS = 100  # Number of bins to divide chromosome into

# Output files
FRAGMENT_FEATURES = PROCESSED_DIR / "fragment_features.csv"
POSITION_FEATURES = PROCESSED_DIR / "position_features.csv"
MOTIF_FEATURES = PROCESSED_DIR / "motif_features.csv"
METHYLATION_FEATURES = PROCESSED_DIR / "methylation_features.csv"
ALL_FEATURES = PROCESSED_DIR / "all_features.csv"


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