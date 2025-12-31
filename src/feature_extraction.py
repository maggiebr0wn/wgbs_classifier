"""
Module 2: Feature Extraction

Purpose:
    Extract multi-modal features from BAM files for classification.
    This module addresses the core assignment requirements:
    - Fragment size distribution features
    - Fragment start/end position features
    - End motif frequency features (4-mers)
    - Methylation features from XM tags

Input:
    - data/processed/sample_manifest.csv (from Module 0)
    - BAM files listed in manifest

Output:
    - data/processed/fragment_features.csv
    - data/processed/position_features.csv
    - data/processed/motif_features.csv
    - data/processed/methylation_features.csv
    - data/processed/all_features.csv (merged)

Functions:
    - extract_fragment_features(): Fragment size statistics and distributions
    - extract_position_features(): Start/end position distributions
    - extract_motif_features(): End motif 4-mer frequencies
    - extract_methylation_features(): CpG methylation from XM tags
    - extract_all_features_from_bam(): Extract all features from single BAM
    - run_module_2(): Run all Module 2 functions

Usage:
    As a script:
        python src/feature_extraction.py
    
    In a notebook or other module:
        from src.feature_extraction import run_module_2
        features = run_module_2()
"""

import pysam
import pandas as pd
import numpy as np
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
    FRAGMENT_SIZE_BINS,
    N_POSITION_BINS,
    FRAGMENT_FEATURES,
    POSITION_FEATURES,
    MOTIF_FEATURES,
    METHYLATION_FEATURES,
    ALL_FEATURES,
    PROCESSED_DIR
)


def extract_fragment_features(bam_path, sample_id, chromosome=CHROMOSOME):
    """
    Extract fragment size features from BAM file.
    
    Extracts:
    - Summary statistics (mean, median, std, quartiles, skewness, kurtosis)
    - Size bin proportions (very_short, short, nucleosomal, dinucleosomal, long)
    - Full fragment size distribution for downstream visualization
    
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
        Dictionary of fragment size features
    """
    features = {'sample_id': sample_id}
    fragment_sizes = []
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        # Collect fragment sizes (process read1 only to avoid double-counting)
        for read in bam.fetch(chromosome):
            # Apply filters
            if (read.is_proper_pair and 
                read.is_read1 and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                # Apply size filter
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    fragment_sizes.append(frag_size)
        
        bam.close()
        
        if len(fragment_sizes) == 0:
            # No valid fragments found
            features['n_fragments'] = 0
            for key in ['mean', 'median', 'std', 'q25', 'q50', 'q75', 'iqr', 
                       'skewness', 'kurtosis']:
                features[f'frag_{key}'] = np.nan
            
            for bin_name in FRAGMENT_SIZE_BINS.keys():
                features[f'frag_pct_{bin_name}'] = np.nan
            
            return features
        
        # Convert to numpy array for efficient computation
        frag_array = np.array(fragment_sizes)
        
        # Summary statistics
        features['n_fragments'] = len(frag_array)
        features['frag_mean'] = np.mean(frag_array)
        features['frag_median'] = np.median(frag_array)
        features['frag_std'] = np.std(frag_array)
        features['frag_min'] = np.min(frag_array)
        features['frag_max'] = np.max(frag_array)
        
        # Quartiles and IQR
        features['frag_q25'] = np.percentile(frag_array, 25)
        features['frag_q50'] = np.percentile(frag_array, 50)  # median
        features['frag_q75'] = np.percentile(frag_array, 75)
        features['frag_iqr'] = features['frag_q75'] - features['frag_q25']
        
        # Distribution shape
        features['frag_skewness'] = skew(frag_array)
        features['frag_kurtosis'] = kurtosis(frag_array)
        
        # Size bin proportions
        total = len(frag_array)
        for bin_name, (min_size, max_size) in FRAGMENT_SIZE_BINS.items():
            count = np.sum((frag_array >= min_size) & (frag_array < max_size))
            features[f'frag_pct_{bin_name}'] = (count / total) * 100
        
    except Exception as e:
        print(f"  Error extracting fragment features for {sample_id}: {e}")
        features['n_fragments'] = 0
        for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q50', 'q75', 
                   'iqr', 'skewness', 'kurtosis']:
            features[f'frag_{key}'] = np.nan
        
        for bin_name in FRAGMENT_SIZE_BINS.keys():
            features[f'frag_pct_{bin_name}'] = np.nan
    
    return features


def extract_position_features(bam_path, sample_id, chromosome=CHROMOSOME, 
                              n_bins=N_POSITION_BINS):
    """
    Extract fragment position features (start and end positions).
    
    Extracts:
    - Binned coverage across chromosome
    - Start position statistics
    - End position statistics
    - Position spread/variance
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze
    n_bins : int
        Number of bins to divide chromosome into
        
    Returns
    -------
    dict
        Dictionary of position features
    """
    features = {'sample_id': sample_id}
    start_positions = []
    end_positions = []
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        # Get chromosome length
        chr_length = bam.get_reference_length(chromosome)
        
        # Collect positions
        for read in bam.fetch(chromosome):
            # Apply same filters as fragment features
            if (read.is_proper_pair and 
                read.is_read1 and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    start_pos = read.reference_start
                    end_pos = read.reference_start + frag_size
                    
                    start_positions.append(start_pos)
                    end_positions.append(end_pos)
        
        bam.close()
        
        if len(start_positions) == 0:
            features['n_positions'] = 0
            features['pos_mean_start'] = np.nan
            features['pos_mean_end'] = np.nan
            features['pos_std_start'] = np.nan
            features['pos_std_end'] = np.nan
            
            # Binned coverage (all NaN)
            for i in range(n_bins):
                features[f'pos_bin_{i}'] = np.nan
            
            return features
        
        # Position statistics
        features['n_positions'] = len(start_positions)
        features['pos_mean_start'] = np.mean(start_positions)
        features['pos_mean_end'] = np.mean(end_positions)
        features['pos_std_start'] = np.std(start_positions)
        features['pos_std_end'] = np.std(end_positions)
        
        # Binned coverage across chromosome
        bin_edges = np.linspace(0, chr_length, n_bins + 1)
        coverage_hist, _ = np.histogram(start_positions, bins=bin_edges)
        
        # Normalize by total fragments (to get proportions)
        total_frags = len(start_positions)
        coverage_pct = (coverage_hist / total_frags) * 100
        
        # Store binned coverage as features
        for i, pct in enumerate(coverage_pct):
            features[f'pos_bin_{i}'] = pct
        
    except Exception as e:
        print(f"  Error extracting position features for {sample_id}: {e}")
        features['n_positions'] = 0
        features['pos_mean_start'] = np.nan
        features['pos_mean_end'] = np.nan
        features['pos_std_start'] = np.nan
        features['pos_std_end'] = np.nan
        
        for i in range(n_bins):
            features[f'pos_bin_{i}'] = np.nan
    
    return features


def extract_motif_features(bam_path, sample_id, chromosome=CHROMOSOME, k=KMER_SIZE):
    """
    Extract end motif features (k-mer frequencies from fragment ends).
    
    NOTE: Data is bisulfite-treated, so motifs reflect methylation status.
    Using as-read approach (no C-correction) to capture methylation signal.
    
    Extracts:
    - Frequency of all possible k-mers (4-mers = 256 features)
    - Motif diversity (Shannon entropy)
    - GC content of end motifs
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze
    k : int
        K-mer size (default: 4)
        
    Returns
    -------
    dict
        Dictionary of motif features
    """
    features = {'sample_id': sample_id}
    motif_counts = Counter()
    total_motifs = 0
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        # Collect end motifs
        for read in bam.fetch(chromosome):
            # Apply same filters
            if (read.is_proper_pair and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    # Get read sequence
                    seq = read.query_sequence
                    
                    if seq and len(seq) >= k:
                        # Extract last k bases (end motif)
                        # Use as-read (includes bisulfite conversion signal)
                        end_motif = seq[-k:].upper()
                        
                        # Only count if valid DNA (no N's)
                        if all(base in 'ACGT' for base in end_motif):
                            motif_counts[end_motif] += 1
                            total_motifs += 1
        
        bam.close()
        
        if total_motifs == 0:
            # No valid motifs found
            features['n_motifs'] = 0
            
            # Generate all possible k-mers
            bases = ['A', 'C', 'G', 'T']
            all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
            
            for kmer in all_kmers:
                features[f'motif_{kmer}'] = np.nan
            
            features['motif_diversity'] = np.nan
            features['motif_gc_content'] = np.nan
            
            return features
        
        # Calculate frequencies
        features['n_motifs'] = total_motifs
        
        # Generate all possible k-mers and get frequencies
        bases = ['A', 'C', 'G', 'T']
        all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
        
        for kmer in all_kmers:
            count = motif_counts.get(kmer, 0)
            features[f'motif_{kmer}'] = (count / total_motifs) * 100  # Convert to percentage
        
        # Motif diversity (Shannon entropy)
        frequencies = np.array([motif_counts.get(kmer, 0) / total_motifs 
                               for kmer in all_kmers])
        # Remove zeros for entropy calculation
        frequencies = frequencies[frequencies > 0]
        entropy = -np.sum(frequencies * np.log2(frequencies))
        features['motif_diversity'] = entropy
        
        # GC content of motifs
        gc_count = sum(motif_counts.get(kmer, 0) 
                      for kmer in all_kmers 
                      if kmer.count('G') + kmer.count('C') >= k/2)
        features['motif_gc_content'] = (gc_count / total_motifs) * 100
        
    except Exception as e:
        print(f"  Error extracting motif features for {sample_id}: {e}")
        features['n_motifs'] = 0
        
        bases = ['A', 'C', 'G', 'T']
        all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
        
        for kmer in all_kmers:
            features[f'motif_{kmer}'] = np.nan
        
        features['motif_diversity'] = np.nan
        features['motif_gc_content'] = np.nan
    
    return features


def extract_methylation_features(bam_path, sample_id, chromosome=CHROMOSOME):
    """
    Extract methylation features from XM tags.
    
    Extracts:
    - CpG methylation rate (global)
    - Total CpG sites covered
    - Methylation variance across reads
    - Per-read methylation statistics
    
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
    
    # Methylation call counters
    cpg_methylated = 0  # Z
    cpg_unmethylated = 0  # z
    chg_methylated = 0  # X
    chg_unmethylated = 0  # x
    chh_methylated = 0  # H
    chh_unmethylated = 0  # h
    
    per_read_meth_rates = []  # CpG methylation rate per read
    reads_with_xm = 0
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        for read in bam.fetch(chromosome):
            # Apply same filters
            if (read.is_proper_pair and 
                not read.is_unmapped and
                read.mapping_quality >= MIN_MAPQ and
                not read.is_duplicate):
                
                frag_size = abs(read.template_length)
                
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    # Check if read has XM tag
                    if read.has_tag('XM'):
                        reads_with_xm += 1
                        xm_tag = read.get_tag('XM')
                        
                        # Count methylation calls
                        z_count = xm_tag.count('Z')
                        z_lower = xm_tag.count('z')
                        
                        cpg_methylated += z_count
                        cpg_unmethylated += z_lower
                        chg_methylated += xm_tag.count('X')
                        chg_unmethylated += xm_tag.count('x')
                        chh_methylated += xm_tag.count('H')
                        chh_unmethylated += xm_tag.count('h')
                        
                        # Per-read CpG methylation rate
                        total_cpg = z_count + z_lower
                        if total_cpg > 0:
                            read_meth_rate = z_count / total_cpg
                            per_read_meth_rates.append(read_meth_rate)
        
        bam.close()
        
        # Calculate features
        features['meth_reads_with_xm'] = reads_with_xm
        
        total_cpg = cpg_methylated + cpg_unmethylated
        total_chg = chg_methylated + chg_unmethylated
        total_chh = chh_methylated + chh_unmethylated
        
        features['meth_total_cpg_sites'] = total_cpg
        features['meth_total_chg_sites'] = total_chg
        features['meth_total_chh_sites'] = total_chh
        
        # CpG methylation rate
        if total_cpg > 0:
            features['meth_cpg_rate'] = cpg_methylated / total_cpg
        else:
            features['meth_cpg_rate'] = np.nan
        
        # CHG methylation rate
        if total_chg > 0:
            features['meth_chg_rate'] = chg_methylated / total_chg
        else:
            features['meth_chg_rate'] = np.nan
        
        # CHH methylation rate (QC metric)
        if total_chh > 0:
            features['meth_chh_rate'] = chh_methylated / total_chh
        else:
            features['meth_chh_rate'] = np.nan
        
        # Per-read statistics
        if len(per_read_meth_rates) > 0:
            features['meth_mean_per_read'] = np.mean(per_read_meth_rates)
            features['meth_std_per_read'] = np.std(per_read_meth_rates)
            features['meth_median_per_read'] = np.median(per_read_meth_rates)
        else:
            features['meth_mean_per_read'] = np.nan
            features['meth_std_per_read'] = np.nan
            features['meth_median_per_read'] = np.nan
        
    except Exception as e:
        print(f"  Error extracting methylation features for {sample_id}: {e}")
        for key in ['reads_with_xm', 'total_cpg_sites', 'total_chg_sites', 
                   'total_chh_sites', 'cpg_rate', 'chg_rate', 'chh_rate',
                   'mean_per_read', 'std_per_read', 'median_per_read']:
            features[f'meth_{key}'] = np.nan if 'rate' in key or 'per_read' in key else 0
    
    return features


def extract_all_features_from_bam(bam_path, sample_id):
    """
    Extract all feature types from a single BAM file.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
        
    Returns
    -------
    dict
        Combined dictionary of all features
    """
    # Extract each feature type
    frag_features = extract_fragment_features(bam_path, sample_id)
    pos_features = extract_position_features(bam_path, sample_id)
    motif_features = extract_motif_features(bam_path, sample_id)
    meth_features = extract_methylation_features(bam_path, sample_id)
    
    # Combine all features
    all_features = {**frag_features, **pos_features, **motif_features, **meth_features}
    
    return all_features


def run_module_2():
    """
    Run complete Module 2: Feature Extraction pipeline.
    
    Returns
    -------
    pd.DataFrame
        All features for all samples
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
    
    # Extract features from all samples
    print("\nExtracting features from all samples...")
    print("=" * 70)
    
    all_sample_features = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), 
                         desc="Processing samples"):
        sample_id = row['sample_id']
        bam_path = row['bam_path']
        
        # Extract all features
        features = extract_all_features_from_bam(bam_path, sample_id)
        
        # Add metadata
        features['disease_status'] = row['disease_status']
        features['batch'] = row['batch']
        features['age'] = row['age']
        
        all_sample_features.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_sample_features)
    
    # Move metadata columns to front
    metadata_cols = ['sample_id', 'disease_status', 'batch', 'age']
    other_cols = [col for col in features_df.columns if col not in metadata_cols]
    features_df = features_df[metadata_cols + other_cols]
    
    print(f"\n✓ Feature extraction complete")
    print(f"  Samples: {len(features_df)}")
    print(f"  Total features: {len(features_df.columns) - 4}")  # Exclude metadata
    
    # Save all features
    features_df.to_csv(ALL_FEATURES, index=False)
    print(f"\n✓ Saved all features to: {ALL_FEATURES}")
    
    # Show feature breakdown
    n_frag = len([c for c in features_df.columns if c.startswith('frag_') or c == 'n_fragments'])
    n_pos = len([c for c in features_df.columns if c.startswith('pos_') or c == 'n_positions'])
    n_motif = len([c for c in features_df.columns if c.startswith('motif_') or c == 'n_motifs'])
    n_meth = len([c for c in features_df.columns if c.startswith('meth_')])
    
    print(f"\nFeature breakdown:")
    print(f"  Fragment size features: {n_frag}")
    print(f"  Position features: {n_pos}")
    print(f"  Motif features: {n_motif}")
    print(f"  Methylation features: {n_meth}")
    print(f"  Total: {n_frag + n_pos + n_motif + n_meth}")
    
    print("\n" + "=" * 70)
    print("MODULE 2 COMPLETE")
    print("=" * 70 + "\n")
    
    return features_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 2 as a standalone script
    features = run_module_2()
    
    # Display summary
    print("\nFeature matrix shape:", features.shape)
    print("\nFirst few rows:")
    print(features.head())