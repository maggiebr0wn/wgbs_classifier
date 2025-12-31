"""
Module 0: Setup & Data Loading

Purpose:
    Load and validate sample metadata, verify BAM file availability,
    and create a clean sample manifest for downstream analysis.

Input:
    - data/metadata/sample_metadata.csv (user-provided in config)
    - data/raw/*.bam and *.bam.bai files (user-provided in config)

Output:
    - data/processed/sample_manifest.csv
    - Console summary statistics

Functions:
    - load_metadata(): Load and validate metadata CSV
    - verify_bam_files(): Check that all BAM files exist
    - create_manifest(): Create clean sample manifest with standardized columns
    - summarize_manifest(): Print summary statistics
    - run_module_0(): Execute complete Module 0 pipeline

Usage:
    As a script:
        python src/01_data_loader.py
    
    In a notebook or other module:
        from src.data_loader import run_module_0
        manifest = run_module_0()
"""

import pandas as pd
from pathlib import Path

# Import Module 0 configuration
from config import (
    METADATA_FILE,
    BAM_DIR,
    BAM_PATTERN,
    SAMPLE_MANIFEST,
    PROCESSED_DIR
)
import re


def load_metadata():
    """
    Load sample metadata from CSV file.
    
    Returns
    -------
    pd.DataFrame
        Metadata with required columns: Run, disease_status, batch, AGE
        
    Raises
    ------
    FileNotFoundError
        If metadata file doesn't exist
    ValueError
        If required columns are missing
    """
    # Check if file exists
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}\n"
            f"Please place sample_metadata.csv in data/metadata/"
        )
    
    # Load CSV
    metadata = pd.read_csv(METADATA_FILE)
    
    # Verify required columns exist
    required_cols = ['Run', 'disease_status', 'batch', 'AGE']
    missing_cols = [col for col in required_cols if col not in metadata.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(metadata.columns)}"
        )
    
    print(f"✓ Loaded metadata: {len(metadata)} samples")
    print(f"  Columns: {list(metadata.columns)}")
    
    return metadata


def verify_bam_files(metadata):
    """
    Verify that BAM files and index files exist for all samples.
    Uses regex pattern matching to find BAM files.
    Add 'bam_path' column to metadata.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Sample metadata with 'Run' column
        
    Returns
    -------
    pd.DataFrame
        Metadata with added 'bam_path' column
        
    Raises
    ------
    FileNotFoundError
        If any BAM or BAI files are missing
    """
    missing_files = []
    bam_paths = []
    
    print(f"\nVerifying BAM files in: {BAM_DIR}")
    print(f"Using pattern: {{run_id}}.*.bam")
    
    for run_id in metadata['Run']:
        # Create pattern for this specific run
        pattern = BAM_PATTERN.format(run_id=run_id)
        
        # Find matching BAM files
        matching_bams = []
        for bam_file in BAM_DIR.glob(f"{run_id}*.bam"):
            # Exclude .bai files
            if not str(bam_file).endswith('.bam.bai'):
                matching_bams.append(bam_file)
        
        if len(matching_bams) == 0:
            missing_files.append(f"{run_id}.*.bam (no match found)")
            bam_paths.append(None)
        elif len(matching_bams) > 1:
            # Multiple matches - use the first one but warn
            print(f"  Warning: Multiple BAM files found for {run_id}, using: {matching_bams[0].name}")
            bam_file = matching_bams[0]
            bai_file = Path(str(bam_file) + ".bai")
            
            # Check BAI exists
            if not bai_file.exists():
                missing_files.append(f"{bam_file.name}.bai")
            
            bam_paths.append(str(bam_file))
        else:
            # Exactly one match - perfect!
            bam_file = matching_bams[0]
            bai_file = Path(str(bam_file) + ".bai")
            
            # Check BAI exists
            if not bai_file.exists():
                missing_files.append(f"{bam_file.name}.bai")
            
            bam_paths.append(str(bam_file))
    
    # Raise error if any files are missing
    if missing_files:
        raise FileNotFoundError(
            f"\nMissing {len(missing_files)} file(s) in {BAM_DIR}:\n" +
            "\n".join(f"  - {f}" for f in missing_files[:10]) +
            (f"\n  ... and {len(missing_files) - 10} more" if len(missing_files) > 10 else "")
        )
    
    # Add bam_path column
    metadata = metadata.copy()
    metadata['bam_path'] = bam_paths
    
    print(f"✓ Verified all BAM files: {len(metadata)} samples")
    
    # Show example of matched file
    if len(bam_paths) > 0 and bam_paths[0]:
        example = Path(bam_paths[0]).name
        print(f"  Example: {example}")
    
    return metadata


def create_manifest(metadata):
    """
    Create clean sample manifest with standardized column names.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with bam_path column
        
    Returns
    -------
    pd.DataFrame
        Clean manifest with columns: sample_id, disease_status, batch, age, bam_path
    """
    # Select and rename columns
    manifest = metadata[['Run', 'disease_status', 'batch', 'AGE', 'bam_path']].copy()
    
    manifest = manifest.rename(columns={
        'Run': 'sample_id',
        'AGE': 'age'
    })
    
    # Standardize disease_status and batch to lowercase
    manifest['disease_status'] = manifest['disease_status'].str.lower()
    manifest['batch'] = manifest['batch'].str.lower()
    
    print(f"\n✓ Created sample manifest")
    print(f"  Columns: {list(manifest.columns)}")
    
    return manifest


def summarize_manifest(manifest):
    """
    Print summary statistics about the sample manifest.
    
    Parameters
    ----------
    manifest : pd.DataFrame
        Sample manifest
    """
    print("\n" + "=" * 70)
    print("SAMPLE MANIFEST SUMMARY")
    print("=" * 70)
    
    # Overall counts
    total = len(manifest)
    n_als = (manifest['disease_status'] == 'als').sum()
    n_ctrl = (manifest['disease_status'] == 'ctrl').sum()
    
    print(f"\nTotal samples: {total}")
    print(f"  ALS: {n_als} ({100*n_als/total:.1f}%)")
    print(f"  Control: {n_ctrl} ({100*n_ctrl/total:.1f}%)")
    
    # By batch
    n_discovery = (manifest['batch'] == 'discovery').sum()
    n_validation = (manifest['batch'] == 'validation').sum()
    
    print(f"\nBy batch:")
    print(f"  Discovery: {n_discovery}")
    print(f"  Validation: {n_validation}")
    
    # Cross-tabulation: Disease × Batch
    print(f"\nDisease status × Batch:")
    crosstab = pd.crosstab(
        manifest['disease_status'], 
        manifest['batch'],
        margins=True,
        margins_name='Total'
    )
    print(crosstab)
    
    # Age statistics
    print(f"\nAge statistics (all samples):")
    print(f"  Mean ± SD: {manifest['age'].mean():.1f} ± {manifest['age'].std():.1f} years")
    print(f"  Range: {manifest['age'].min():.0f} - {manifest['age'].max():.0f} years")
    
    # Age by disease status
    print(f"\nAge by disease status:")
    age_by_disease = manifest.groupby('disease_status')['age'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(1)
    print(age_by_disease)
    
    # Age by batch
    print(f"\nAge by batch:")
    age_by_batch = manifest.groupby('batch')['age'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(1)
    print(age_by_batch)
    
    print("=" * 70)


def run_module_0(save=True):
    """
    Run complete Module 0: Setup & Data Loading pipeline.
    
    Parameters
    ----------
    save : bool, default=True
        Whether to save the manifest to CSV
        
    Returns
    -------
    pd.DataFrame
        Sample manifest with columns: sample_id, disease_status, batch, age, bam_path
    """
    print("\n" + "=" * 70)
    print("MODULE 0: Setup & Data Loading")
    print("=" * 70)
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load metadata
    print("\nStep 1: Loading metadata...")
    metadata = load_metadata()
    
    # Step 2: Verify BAM files
    print("\nStep 2: Verifying BAM files...")
    metadata = verify_bam_files(metadata)
    
    # Step 3: Create clean manifest
    print("\nStep 3: Creating sample manifest...")
    manifest = create_manifest(metadata)
    
    # Step 4: Print summary
    summarize_manifest(manifest)
    
    # Step 5: Save manifest
    if save:
        manifest.to_csv(SAMPLE_MANIFEST, index=False)
        print(f"\n✓ Saved sample manifest to: {SAMPLE_MANIFEST}")
    
    print("\n" + "=" * 70)
    print("MODULE 0 COMPLETE")
    print("=" * 70 + "\n")
    
    return manifest


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 0 as a standalone script
    manifest = run_module_0()
    
    # Display first few rows
    print("\nFirst 5 rows of sample manifest:")
    print(manifest.head())