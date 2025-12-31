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
# (Will be added later)


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