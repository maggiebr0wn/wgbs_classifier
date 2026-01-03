"""
Module 2: Feature Extraction

Purpose:
    Extract fragmentomics and methylation features using gold-standard methods
    from published cfDNA and WGBS literature.
    
    This implementation follows best practices from:
    - cfDNA fragmentomics: Cristiano et al. 2019, Snyder et al. 2016
    - WGBS methylation: Loyfer et al. 2023, Sun et al. 2015

Input:
    - data/processed/sample_manifest.csv (from Module 0)
    - BAM files

Output:
    - data/processed/fragmentomics_features.csv (~509 features)
    - data/processed/methylation_features.csv (~475 features)
    - data/processed/all_features.csv (~984 features total)

Functions:
    - get_chromosome_length(): Get chromosome length from BAM header
    - extract_fragmentomics_features(): Fragment size, motifs, coverage
    - extract_methylation_features(): Global + regional methylation
    - run_module_2(): Execute complete pipeline

Usage:
    As a script:
        python src/feature_extraction.py
    
    In a notebook:
        from src.feature_extraction import run_module_2
        features = run_module_2()
"""

import pysam
import pandas as pd
import numpy as np
import math
from pathlib import Path
from collections import Counter
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import itertools
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    SAMPLE_MANIFEST,
    CHROMOSOME,
    MIN_MAPQ,
    MIN_FRAGMENT_SIZE,
    MAX_FRAGMENT_SIZE,
    KMER_SIZE,
    FRAGMENT_SIZE_CATEGORIES,
    FRAGMENTOMICS_BIN_SIZE,
    METHYLATION_BIN_SIZE,
    MIN_CPG_PER_BIN,
    HIGHLY_METHYLATED_THRESHOLD,
    LOWLY_METHYLATED_THRESHOLD,
    FRAGMENTOMICS_FEATURES,
    METHYLATION_FEATURES_FILE,
    ALL_FEATURES,
    PROCESSED_DIR
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_chromosome_length(bam_path, chromosome):
    """
    Get chromosome length from BAM file header.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    chromosome : str
        Chromosome name
        
    Returns
    -------
    int
        Chromosome length in base pairs
    """
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        chr_length = bam.get_reference_length(chromosome)
        bam.close()
        return chr_length
    except Exception as e:
        print(f"  Error getting chromosome length: {e}")
        return 46709983  # Fallback to known chr21 length


# ============================================================================
# Fragmentomics Feature Extraction
# ============================================================================

def extract_fragmentomics_features(bam_path, sample_id, chromosome=CHROMOSOME):
    """
    Extract fragmentomics features using gold-standard cfDNA methods.
    
    Features extracted:
    1. Fragment size distribution (15 features)
       - Summary statistics: mean, median, std, quartiles, skewness, kurtosis
       - Size ratios: short/long, mono/di-nucleosomal
    
    2. Fragment size categories (5 features)
       - % of fragments in each size category
    
    3. End motif features (258 features)
       - ALL 256 4-mer proportions (for assignment requirement)
       - Motif diversity (Shannon entropy)
       - GC content of motifs
    
    4. Coverage profile (N features, N = chr_length / bin_size)
       - % of fragments starting in each bin (100 kb bins)
    
    Total: ~745 features for chr21 with 100 kb bins
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze
        
    Returns
    -------
    dict
        Dictionary of fragmentomics features
    """
    features = {'sample_id': sample_id}
    
    # Data collection
    fragment_sizes = []
    start_positions = []
    end_motifs = Counter()
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        chr_length = bam.get_reference_length(chromosome)
        
        # Collect data from reads
        for read in bam.fetch(chromosome):
            # Apply filters (process read1 only to avoid double-counting)
            if (read.is_proper_pair and 
                read.is_read1 and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    # Fragment size
                    fragment_sizes.append(frag_size)
                    
                    # Start position (for coverage)
                    start_positions.append(read.reference_start)
                    
                    # End motif (last k bases of read sequence)
                    seq = read.query_sequence
                    if seq and len(seq) >= KMER_SIZE:
                        end_motif = seq[-KMER_SIZE:].upper()
                        if all(base in 'ACGT' for base in end_motif):
                            end_motifs[end_motif] += 1
        
        bam.close()
        
        # ====================================================================
        # 1. FRAGMENT SIZE FEATURES
        # ====================================================================
        
        if len(fragment_sizes) == 0:
            # No valid fragments - set all to NaN
            for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q50', 
                       'q75', 'iqr', 'skewness', 'kurtosis', 'ratio_short_long',
                       'ratio_mono_di', 'cv']:
                features[f'frag_{key}'] = np.nan
            
            for cat_name in FRAGMENT_SIZE_CATEGORIES.keys():
                features[f'frag_pct_{cat_name}'] = np.nan
            
            features['n_fragments'] = 0
        else:
            frag_array = np.array(fragment_sizes)
            features['n_fragments'] = len(frag_array)
            
            # Summary statistics
            features['frag_mean'] = np.mean(frag_array)
            features['frag_median'] = np.median(frag_array)
            features['frag_std'] = np.std(frag_array)
            features['frag_min'] = np.min(frag_array)
            features['frag_max'] = np.max(frag_array)
            features['frag_q25'] = np.percentile(frag_array, 25)
            features['frag_q50'] = np.percentile(frag_array, 50)
            features['frag_q75'] = np.percentile(frag_array, 75)
            features['frag_iqr'] = features['frag_q75'] - features['frag_q25']
            
            # Distribution shape
            features['frag_skewness'] = skew(frag_array)
            features['frag_kurtosis'] = kurtosis(frag_array)
            
            # Coefficient of variation
            features['frag_cv'] = features['frag_std'] / features['frag_mean']
            
            # Size ratios (informative for cfDNA analysis)
            short_count = np.sum(frag_array < 150)
            long_count = np.sum(frag_array >= 220)
            mono_count = np.sum((frag_array >= 150) & (frag_array < 220))
            di_count = np.sum((frag_array >= 220) & (frag_array < 400))
            
            features['frag_ratio_short_long'] = short_count / long_count if long_count > 0 else np.nan
            features['frag_ratio_mono_di'] = mono_count / di_count if di_count > 0 else np.nan
            
            # Fragment size categories
            total = len(frag_array)
            for cat_name, (min_size, max_size) in FRAGMENT_SIZE_CATEGORIES.items():
                count = np.sum((frag_array >= min_size) & (frag_array < max_size))
                features[f'frag_pct_{cat_name}'] = (count / total) * 100
        
        # ====================================================================
        # 2. END MOTIF FEATURES (ALL 256 4-MERS)
        # ====================================================================
        
        total_motifs = sum(end_motifs.values())
        
        if total_motifs == 0:
            # No motifs - set all to NaN
            # Generate all 256 possible 4-mers
            all_kmers = [''.join(p) for p in itertools.product('ACGT', repeat=KMER_SIZE)]
            
            for kmer in all_kmers:
                features[f'kmer_{kmer}'] = np.nan
            
            features['motif_diversity'] = np.nan
            features['motif_gc_content'] = np.nan
        else:
            # Generate all 256 possible 4-mers
            all_kmers = [''.join(p) for p in itertools.product('ACGT', repeat=KMER_SIZE)]
            
            # Store proportion for EACH k-mer (all 256)
            for kmer in all_kmers:
                count = end_motifs.get(kmer, 0)
                features[f'kmer_{kmer}'] = (count / total_motifs) * 100
            
            # Motif diversity (Shannon entropy)
            frequencies = np.array([end_motifs.get(kmer, 0) / total_motifs for kmer in all_kmers])
            frequencies = frequencies[frequencies > 0]
            entropy = -np.sum(frequencies * np.log2(frequencies))
            features['motif_diversity'] = entropy
            
            # GC content of motifs
            gc_count = sum(count for kmer, count in end_motifs.items() 
                          if (kmer.count('G') + kmer.count('C')) >= KMER_SIZE / 2)
            features['motif_gc_content'] = (gc_count / total_motifs) * 100
        
        # ====================================================================
        # 3. COVERAGE PROFILE (100 KB BINS)
        # ====================================================================
        
        n_bins = math.ceil(chr_length / FRAGMENTOMICS_BIN_SIZE)
        
        if len(start_positions) == 0:
            features['coverage_chr_length'] = chr_length
            features['coverage_bin_size'] = FRAGMENTOMICS_BIN_SIZE
            features['coverage_n_bins'] = n_bins
            
            for i in range(n_bins):
                features[f'coverage_bin_{i}'] = np.nan
        else:
            features['coverage_chr_length'] = chr_length
            features['coverage_bin_size'] = FRAGMENTOMICS_BIN_SIZE
            features['coverage_n_bins'] = n_bins
            
            # Create histogram of coverage
            bin_edges = np.arange(0, chr_length + FRAGMENTOMICS_BIN_SIZE, FRAGMENTOMICS_BIN_SIZE)
            coverage_hist, _ = np.histogram(start_positions, bins=bin_edges)
            
            # Normalize to percentages
            total_frags = len(start_positions)
            coverage_pct = (coverage_hist / total_frags) * 100
            
            # Store as features
            for i, pct in enumerate(coverage_pct):
                features[f'coverage_bin_{i}'] = pct
        
    except Exception as e:
        print(f"  Error extracting fragmentomics features for {sample_id}: {e}")
        
        # Return NaN-filled features on error
        for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q50', 
                   'q75', 'iqr', 'skewness', 'kurtosis', 'ratio_short_long',
                   'ratio_mono_di', 'cv']:
            features[f'frag_{key}'] = np.nan
        
        for cat_name in FRAGMENT_SIZE_CATEGORIES.keys():
            features[f'frag_pct_{cat_name}'] = np.nan
        
        # All 256 k-mers
        all_kmers = [''.join(p) for p in itertools.product('ACGT', repeat=KMER_SIZE)]
        for kmer in all_kmers:
            features[f'kmer_{kmer}'] = np.nan
        
        features['motif_diversity'] = np.nan
        features['motif_gc_content'] = np.nan
        features['n_fragments'] = 0
        
        # Coverage bins
        chr_length = get_chromosome_length(bam_path, chromosome)
        n_bins = math.ceil(chr_length / FRAGMENTOMICS_BIN_SIZE) if chr_length else 0
        
        features['coverage_chr_length'] = chr_length
        features['coverage_bin_size'] = FRAGMENTOMICS_BIN_SIZE
        features['coverage_n_bins'] = n_bins
        
        for i in range(n_bins):
            features[f'coverage_bin_{i}'] = np.nan
    
    return features


# ============================================================================
# Methylation Feature Extraction
# ============================================================================

def extract_methylation_features(bam_path, sample_id, chromosome=CHROMOSOME):
    """
    Extract methylation features using gold-standard WGBS methods.
    
    Features extracted:
    1. Global methylation (8 features)
       - Mean CpG methylation
       - Methylation variance (std, CV)
       - Distribution: % high (>0.8), % low (<0.2), % intermediate
       - Total CpG sites covered
    
    2. Regional methylation (N features, N = chr_length / bin_size)
       - Average CpG methylation per bin (100 kb bins)
       - Only bins with ≥20 CpG sites
    
    Total: ~475 features for chr21 with 100 kb bins
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze
        
    Returns
    -------
    dict
        Dictionary of methylation features
    """
    features = {'sample_id': sample_id}
    
    # Global methylation counters
    total_cpg_meth = 0
    total_cpg_unmeth = 0
    
    # Per-CpG methylation values (for distribution analysis)
    per_site_methylation = []
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        chr_length = bam.get_reference_length(chromosome)
        n_bins = math.ceil(chr_length / METHYLATION_BIN_SIZE)
        
        # Regional methylation counters
        bin_cpg_meth = np.zeros(n_bins)
        bin_cpg_unmeth = np.zeros(n_bins)
        
        # Collect methylation data
        for read in bam.fetch(chromosome):
            if (read.is_proper_pair and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    if read.has_tag('XM'):
                        xm_tag = read.get_tag('XM')
                        
                        # Count methylation calls
                        z_meth = xm_tag.count('Z')
                        z_unmeth = xm_tag.count('z')
                        
                        # Global counts
                        total_cpg_meth += z_meth
                        total_cpg_unmeth += z_unmeth
                        
                        # Regional counts (assign to bin)
                        pos = read.reference_start + (read.reference_length // 2)
                        bin_idx = int(pos / METHYLATION_BIN_SIZE)
                        bin_idx = min(bin_idx, n_bins - 1)
                        
                        bin_cpg_meth[bin_idx] += z_meth
                        bin_cpg_unmeth[bin_idx] += z_unmeth
                        
                        # Per-site methylation (approximate from read-level)
                        if (z_meth + z_unmeth) > 0:
                            read_meth_rate = z_meth / (z_meth + z_unmeth)
                            per_site_methylation.append(read_meth_rate)
        
        bam.close()
        
        # ====================================================================
        # 1. GLOBAL METHYLATION FEATURES
        # ====================================================================
        
        total_cpg = total_cpg_meth + total_cpg_unmeth
        features['meth_total_cpg_sites'] = total_cpg
        
        if total_cpg > 0:
            # Mean methylation
            mean_meth = total_cpg_meth / total_cpg
            features['meth_mean_cpg'] = mean_meth
            
            # Methylation variance (from per-site values)
            if len(per_site_methylation) > 0:
                features['meth_std'] = np.std(per_site_methylation)
                features['meth_cv'] = features['meth_std'] / mean_meth if mean_meth > 0 else np.nan
            else:
                features['meth_std'] = np.nan
                features['meth_cv'] = np.nan
            
            # Methylation distribution
            if len(per_site_methylation) > 0:
                per_site_array = np.array(per_site_methylation)
                
                high_meth = np.sum(per_site_array > HIGHLY_METHYLATED_THRESHOLD)
                low_meth = np.sum(per_site_array < LOWLY_METHYLATED_THRESHOLD)
                intermediate_meth = len(per_site_array) - high_meth - low_meth
                
                total_sites = len(per_site_array)
                features['meth_pct_high'] = (high_meth / total_sites) * 100
                features['meth_pct_low'] = (low_meth / total_sites) * 100
                features['meth_pct_intermediate'] = (intermediate_meth / total_sites) * 100
            else:
                features['meth_pct_high'] = np.nan
                features['meth_pct_low'] = np.nan
                features['meth_pct_intermediate'] = np.nan
        else:
            features['meth_mean_cpg'] = np.nan
            features['meth_std'] = np.nan
            features['meth_cv'] = np.nan
            features['meth_pct_high'] = np.nan
            features['meth_pct_low'] = np.nan
            features['meth_pct_intermediate'] = np.nan
        
        # ====================================================================
        # 2. REGIONAL METHYLATION FEATURES (100 KB BINS)
        # ====================================================================
        
        features['regional_meth_chr_length'] = chr_length
        features['regional_meth_bin_size'] = METHYLATION_BIN_SIZE
        features['regional_meth_n_bins'] = n_bins
        
        # Calculate methylation per bin
        bins_with_data = 0
        regional_meth_values = []
        
        for i in range(n_bins):
            total_cpg_in_bin = bin_cpg_meth[i] + bin_cpg_unmeth[i]
            
            if total_cpg_in_bin >= MIN_CPG_PER_BIN:
                meth_rate = bin_cpg_meth[i] / total_cpg_in_bin
                features[f'regional_meth_bin_{i}'] = meth_rate
                bins_with_data += 1
                regional_meth_values.append(meth_rate)
            else:
                features[f'regional_meth_bin_{i}'] = np.nan
        
        # Regional methylation summary
        features['regional_meth_n_bins_with_data'] = bins_with_data
        
        if len(regional_meth_values) > 0:
            features['regional_meth_mean'] = np.mean(regional_meth_values)
            features['regional_meth_std'] = np.std(regional_meth_values)
            features['regional_meth_min'] = np.min(regional_meth_values)
            features['regional_meth_max'] = np.max(regional_meth_values)
        else:
            features['regional_meth_mean'] = np.nan
            features['regional_meth_std'] = np.nan
            features['regional_meth_min'] = np.nan
            features['regional_meth_max'] = np.nan
        
    except Exception as e:
        print(f"  Error extracting methylation features for {sample_id}: {e}")
        
        # Return NaN-filled features on error
        features['meth_total_cpg_sites'] = 0
        features['meth_mean_cpg'] = np.nan
        features['meth_std'] = np.nan
        features['meth_cv'] = np.nan
        features['meth_pct_high'] = np.nan
        features['meth_pct_low'] = np.nan
        features['meth_pct_intermediate'] = np.nan
        
        chr_length = get_chromosome_length(bam_path, chromosome)
        n_bins = math.ceil(chr_length / METHYLATION_BIN_SIZE) if chr_length else 0
        
        features['regional_meth_chr_length'] = chr_length
        features['regional_meth_bin_size'] = METHYLATION_BIN_SIZE
        features['regional_meth_n_bins'] = n_bins
        features['regional_meth_n_bins_with_data'] = 0
        features['regional_meth_mean'] = np.nan
        features['regional_meth_std'] = np.nan
        features['regional_meth_min'] = np.nan
        features['regional_meth_max'] = np.nan
        
        for i in range(n_bins):
            features[f'regional_meth_bin_{i}'] = np.nan
    
    return features


# ============================================================================
# Main Pipeline
# ============================================================================

def run_module_2():
    """
    Run complete Module 2: Feature Extraction pipeline.
    
    Extracts fragmentomics and methylation features separately,
    then merges them into a single feature matrix.
    
    Returns
    -------
    tuple
        (fragmentomics_df, methylation_df, all_features_df)
    """
    print("\n" + "=" * 70)
    print("MODULE 2: Feature Extraction")
    print("=" * 70)
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load sample manifest
    print(f"\nLoading sample manifest from: {SAMPLE_MANIFEST}")
    
    if not SAMPLE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Sample manifest not found: {SAMPLE_MANIFEST}\n"
            f"Please run Module 0 first."
        )
    
    manifest = pd.read_csv(SAMPLE_MANIFEST)
    print(f"✓ Loaded manifest: {len(manifest)} samples")
    
    # Get chromosome info from first BAM
    first_bam = manifest.iloc[0]['bam_path']
    chr_length = get_chromosome_length(first_bam, CHROMOSOME)
    
    n_coverage_bins = math.ceil(chr_length / FRAGMENTOMICS_BIN_SIZE)
    n_meth_bins = math.ceil(chr_length / METHYLATION_BIN_SIZE)
    
    print(f"\nConfiguration:")
    print(f"  Chromosome: {CHROMOSOME}")
    print(f"  Chromosome length: {chr_length:,} bp")
    print(f"  Fragmentomics bin size: {FRAGMENTOMICS_BIN_SIZE:,} bp → {n_coverage_bins} bins")
    print(f"  Methylation bin size: {METHYLATION_BIN_SIZE:,} bp → {n_meth_bins} bins")
    print(f"  Min CpG per bin: {MIN_CPG_PER_BIN}")
    
    # ========================================================================
    # EXTRACT FRAGMENTOMICS FEATURES
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXTRACTING FRAGMENTOMICS FEATURES")
    print("=" * 70)
    
    fragmentomics_data = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), 
                         desc="Fragmentomics"):
        sample_id = row['sample_id']
        bam_path = row['bam_path']
        
        # Extract fragmentomics features
        frag_features = extract_fragmentomics_features(bam_path, sample_id)
        
        # Add metadata
        frag_features['disease_status'] = row['disease_status']
        frag_features['batch'] = row['batch']
        frag_features['age'] = row['age']
        
        fragmentomics_data.append(frag_features)
    
    # Convert to DataFrame
    fragmentomics_df = pd.DataFrame(fragmentomics_data)
    
    # Move metadata to front
    metadata_cols = ['sample_id', 'disease_status', 'batch', 'age']
    other_cols = [c for c in fragmentomics_df.columns if c not in metadata_cols]
    fragmentomics_df = fragmentomics_df[metadata_cols + other_cols]
    
    print(f"\n✓ Fragmentomics extraction complete")
    print(f"  Samples: {len(fragmentomics_df)}")
    print(f"  Total features: {len(fragmentomics_df.columns) - 4}")
    
    # Save fragmentomics features
    fragmentomics_df.to_csv(FRAGMENTOMICS_FEATURES, index=False)
    print(f"  Saved to: {FRAGMENTOMICS_FEATURES}")
    
    # ========================================================================
    # EXTRACT METHYLATION FEATURES
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("EXTRACTING METHYLATION FEATURES")
    print("=" * 70)
    
    methylation_data = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), 
                         desc="Methylation"):
        sample_id = row['sample_id']
        bam_path = row['bam_path']
        
        # Extract methylation features
        meth_features = extract_methylation_features(bam_path, sample_id)
        
        # Add metadata
        meth_features['disease_status'] = row['disease_status']
        meth_features['batch'] = row['batch']
        meth_features['age'] = row['age']
        
        methylation_data.append(meth_features)
    
    # Convert to DataFrame
    methylation_df = pd.DataFrame(methylation_data)
    
    # Move metadata to front
    methylation_df = methylation_df[metadata_cols + 
                                   [c for c in methylation_df.columns if c not in metadata_cols]]
    
    print(f"\n✓ Methylation extraction complete")
    print(f"  Samples: {len(methylation_df)}")
    print(f"  Total features: {len(methylation_df.columns) - 4}")
    
    # Check regional methylation coverage
    if 'regional_meth_n_bins_with_data' in methylation_df.columns:
        mean_bins = methylation_df['regional_meth_n_bins_with_data'].mean()
        coverage_pct = (mean_bins / n_meth_bins) * 100
        print(f"  Regional methylation coverage: {mean_bins:.0f}/{n_meth_bins} bins ({coverage_pct:.1f}%)")
    
    # Save methylation features
    methylation_df.to_csv(METHYLATION_FEATURES_FILE, index=False)
    print(f"  Saved to: {METHYLATION_FEATURES_FILE}")
    
    # ========================================================================
    # MERGE ALL FEATURES
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("MERGING FEATURES")
    print("=" * 70)
    
    # Merge on sample_id
    all_features_df = fragmentomics_df.merge(
        methylation_df.drop(columns=['disease_status', 'batch', 'age']),
        on='sample_id',
        how='inner'
    )
    
    print(f"\n✓ Merge complete")
    print(f"  Samples: {len(all_features_df)}")
    print(f"  Total columns: {len(all_features_df.columns)}")
    print(f"  Total features: {len(all_features_df.columns) - 4}")
    
    # Feature breakdown
    n_frag_size = len([c for c in all_features_df.columns if c.startswith('frag_')])
    n_motif = len([c for c in all_features_df.columns if c.startswith('motif_')])
    n_coverage = len([c for c in all_features_df.columns if c.startswith('coverage_bin_')])
    n_global_meth = len([c for c in all_features_df.columns if c.startswith('meth_') and 'regional' not in c])
    n_regional_meth = len([c for c in all_features_df.columns if c.startswith('regional_meth_bin_')])
    
    print(f"\nFeature breakdown:")
    print(f"  Fragment size features: {n_frag_size}")
    print(f"  End motif features: {n_motif}")
    print(f"  Coverage bins: {n_coverage}")
    print(f"  Global methylation: {n_global_meth}")
    print(f"  Regional methylation bins: {n_regional_meth}")
    print(f"  Total: {n_frag_size + n_motif + n_coverage + n_global_meth + n_regional_meth}")
    
    # Save merged features
    all_features_df.to_csv(ALL_FEATURES, index=False)
    file_size_mb = ALL_FEATURES.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved all features to: {ALL_FEATURES}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print("\n" + "=" * 70)
    print("MODULE 2 COMPLETE")
    print("=" * 70 + "\n")
    
    return fragmentomics_df, methylation_df, all_features_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 2 as a standalone script
    frag_df, meth_df, all_df = run_module_2()
    
    print("\nFeature matrix shapes:")
    print(f"  Fragmentomics: {frag_df.shape}")
    print(f"  Methylation: {meth_df.shape}")
    print(f"  All features: {all_df.shape}")