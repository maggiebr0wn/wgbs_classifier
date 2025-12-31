"""
Module 1: Quality Control & Filtering

Purpose:
    Assess data quality from BAM files, check bisulfite conversion efficiency,
    and evaluate batch effects on QC metrics.

Input:
    - data/processed/sample_manifest.csv (from Module 0)
    - BAM files listed in manifest

Output:
    - data/processed/qc_metrics.csv
    - results/qc_report.txt
    - results/figures/qc/*.png

Functions:
    - calculate_bam_stats(): Extract basic BAM statistics
    - check_bisulfite_conversion(): Assess conversion efficiency from CHH methylation
    - assess_batch_effects(): Compare QC metrics between batches
    - generate_qc_plots(): Create QC visualization plots
    - run_module_1(): Execute all Module 1 steps

Usage:
    As a script:
        python src/qc.py
    
    In a notebook or other module:
        from src.qc import run_module_1
        qc_metrics = run_module_1()
"""

import pysam
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    SAMPLE_MANIFEST,
    QC_METRICS,
    QC_REPORT,
    QC_FIGURES_DIR,
    CHROMOSOME,
    MIN_MAPQ,
    MIN_FRAGMENT_SIZE,
    MAX_FRAGMENT_SIZE,
    BISULFITE_CONVERSION_THRESHOLD,
    CHH_METHYLATION_THRESHOLD,
    QC_SAMPLE_SIZE,
    PROCESSED_DIR,
    RESULTS_DIR
)


def calculate_bam_stats(bam_path, sample_id, chromosome=CHROMOSOME):
    """
    Calculate basic statistics from a BAM file.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze (default: chr21)
        
    Returns
    -------
    dict
        Dictionary of QC metrics
    """
    stats = {'sample_id': sample_id}
    
    try:
        # Open BAM file
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        # Initialize counters
        total_reads = 0
        mapped_reads = 0
        properly_paired = 0
        duplicates = 0
        mapq_sum = 0
        fragment_sizes = []
        
        # Iterate through reads
        for read in bam.fetch(chromosome):
            total_reads += 1
            
            if not read.is_unmapped:
                mapped_reads += 1
                mapq_sum += read.mapping_quality
            
            if read.is_proper_pair:
                properly_paired += 1
            
            if read.is_duplicate:
                duplicates += 1
            
            # Collect fragment sizes (only for read1 to avoid double counting)
            if read.is_proper_pair and read.is_read1:
                frag_size = abs(read.template_length)
                if MIN_FRAGMENT_SIZE <= frag_size <= MAX_FRAGMENT_SIZE:
                    fragment_sizes.append(frag_size)
        
        bam.close()
        
        # Calculate statistics
        stats['total_reads'] = total_reads
        stats['mapped_reads'] = mapped_reads
        stats['mapped_pct'] = (mapped_reads / total_reads * 100) if total_reads > 0 else 0
        stats['properly_paired'] = properly_paired
        stats['properly_paired_pct'] = (properly_paired / total_reads * 100) if total_reads > 0 else 0
        stats['duplicates'] = duplicates
        stats['duplicate_rate'] = (duplicates / total_reads * 100) if total_reads > 0 else 0
        stats['mean_mapq'] = (mapq_sum / mapped_reads) if mapped_reads > 0 else 0
        
        # Fragment size statistics
        if len(fragment_sizes) > 0:
            stats['fragment_count'] = len(fragment_sizes)
            stats['mean_fragment_size'] = np.mean(fragment_sizes)
            stats['median_fragment_size'] = np.median(fragment_sizes)
            stats['std_fragment_size'] = np.std(fragment_sizes)
            stats['min_fragment_size'] = np.min(fragment_sizes)
            stats['max_fragment_size'] = np.max(fragment_sizes)
        else:
            stats['fragment_count'] = 0
            stats['mean_fragment_size'] = np.nan
            stats['median_fragment_size'] = np.nan
            stats['std_fragment_size'] = np.nan
            stats['min_fragment_size'] = np.nan
            stats['max_fragment_size'] = np.nan
        
    except Exception as e:
        print(f"  Error processing {sample_id}: {e}")
        # Return dict with NaN values
        for key in ['total_reads', 'mapped_reads', 'mapped_pct', 'properly_paired',
                    'properly_paired_pct', 'duplicates', 'duplicate_rate', 'mean_mapq',
                    'fragment_count', 'mean_fragment_size', 'median_fragment_size',
                    'std_fragment_size', 'min_fragment_size', 'max_fragment_size']:
            if key not in stats:
                stats[key] = np.nan
    
    return stats


def check_bisulfite_conversion(bam_path, sample_id, chromosome=CHROMOSOME, 
                                sample_size=QC_SAMPLE_SIZE):
    """
    Check bisulfite conversion efficiency by examining CHH methylation.
    
    Parameters
    ----------
    bam_path : str
        Path to BAM file
    sample_id : str
        Sample identifier
    chromosome : str
        Chromosome to analyze
    sample_size : int
        Number of reads to sample for conversion check
        
    Returns
    -------
    dict
        Dictionary with conversion metrics
    """
    metrics = {'sample_id': sample_id}
    
    try:
        bam = pysam.AlignmentFile(bam_path, 'rb')
        
        # Counters for methylation calls
        chh_methylated = 0  # H in XM tag
        chh_unmethylated = 0  # h in XM tag
        cpg_methylated = 0  # Z in XM tag
        cpg_unmethylated = 0  # z in XM tag
        reads_with_xm = 0
        reads_sampled = 0
        
        # Sample reads
        for read in bam.fetch(chromosome):
            if reads_sampled >= sample_size:
                break
            
            # Check if read has XM tag (methylation string)
            if read.has_tag('XM'):
                reads_sampled += 1
                reads_with_xm += 1
                xm_tag = read.get_tag('XM')
                
                # Count methylation calls
                chh_methylated += xm_tag.count('H')
                chh_unmethylated += xm_tag.count('h')
                cpg_methylated += xm_tag.count('Z')
                cpg_unmethylated += xm_tag.count('z')
        
        bam.close()
        
        # Calculate metrics
        total_chh = chh_methylated + chh_unmethylated
        total_cpg = cpg_methylated + cpg_unmethylated
        
        metrics['reads_sampled'] = reads_sampled
        metrics['reads_with_xm'] = reads_with_xm
        metrics['total_chh_sites'] = total_chh
        metrics['total_cpg_sites'] = total_cpg
        
        # CHH methylation rate (should be ~0% for good conversion)
        metrics['chh_methylation_rate'] = (chh_methylated / total_chh) if total_chh > 0 else np.nan
        
        # Conversion efficiency (inverse of CHH methylation)
        metrics['conversion_efficiency'] = 1 - metrics['chh_methylation_rate'] if not np.isnan(metrics['chh_methylation_rate']) else np.nan
        
        # CpG methylation rate (for reference)
        metrics['cpg_methylation_rate'] = (cpg_methylated / total_cpg) if total_cpg > 0 else np.nan
        
        # Flag if conversion is poor
        if metrics['conversion_efficiency'] < BISULFITE_CONVERSION_THRESHOLD:
            metrics['conversion_flag'] = 'POOR'
        else:
            metrics['conversion_flag'] = 'PASS'
        
    except Exception as e:
        print(f"  Error checking conversion for {sample_id}: {e}")
        for key in ['reads_sampled', 'reads_with_xm', 'total_chh_sites', 'total_cpg_sites',
                    'chh_methylation_rate', 'conversion_efficiency', 'cpg_methylation_rate',
                    'conversion_flag']:
            if key not in metrics:
                metrics[key] = np.nan if key != 'conversion_flag' else 'ERROR'
    
    return metrics


def run_qc_on_all_samples(manifest):
    """
    Run QC on all samples in the manifest.
    
    Parameters
    ----------
    manifest : pd.DataFrame
        Sample manifest from Module 0
        
    Returns
    -------
    pd.DataFrame
        QC metrics for all samples
    """
    print("\nCalculating QC metrics for all samples...")
    print("=" * 70)
    
    qc_results = []
    
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing samples"):
        sample_id = row['sample_id']
        bam_path = row['bam_path']
        
        # Calculate BAM stats
        bam_stats = calculate_bam_stats(bam_path, sample_id)
        
        # Check bisulfite conversion
        conv_metrics = check_bisulfite_conversion(bam_path, sample_id)
        
        # Combine metrics
        combined = {**bam_stats, **conv_metrics}
        
        # Add metadata
        combined['disease_status'] = row['disease_status']
        combined['batch'] = row['batch']
        combined['age'] = row['age']
        
        qc_results.append(combined)
    
    # Convert to DataFrame
    qc_df = pd.DataFrame(qc_results)
    
    # Reorder columns for clarity
    cols_order = ['sample_id', 'disease_status', 'batch', 'age',
                  'total_reads', 'mapped_reads', 'mapped_pct',
                  'properly_paired', 'properly_paired_pct',
                  'duplicates', 'duplicate_rate', 'mean_mapq',
                  'fragment_count', 'mean_fragment_size', 'median_fragment_size',
                  'std_fragment_size', 'min_fragment_size', 'max_fragment_size',
                  'reads_sampled', 'reads_with_xm', 'total_chh_sites', 'total_cpg_sites',
                  'chh_methylation_rate', 'conversion_efficiency', 'cpg_methylation_rate',
                  'conversion_flag']
    
    # Only reorder columns that exist
    cols_order = [c for c in cols_order if c in qc_df.columns]
    qc_df = qc_df[cols_order]
    
    print(f"\n✓ QC metrics calculated for {len(qc_df)} samples")
    
    return qc_df


def assess_batch_effects(qc_df):
    """
    Assess batch effects on QC metrics.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        QC metrics dataframe
        
    Returns
    -------
    pd.DataFrame
        Statistical comparison of batches
    """
    print("\nAssessing batch effects on QC metrics...")
    print("=" * 70)
    
    # Metrics to test
    metrics_to_test = [
        'total_reads', 'mapped_pct', 'properly_paired_pct',
        'duplicate_rate', 'mean_mapq', 'mean_fragment_size',
        'conversion_efficiency', 'cpg_methylation_rate'
    ]
    
    batch_comparisons = []
    
    for metric in metrics_to_test:
        if metric not in qc_df.columns:
            continue
        
        # Split by batch
        discovery = qc_df[qc_df['batch'] == 'discovery'][metric].dropna()
        validation = qc_df[qc_df['batch'] == 'validation'][metric].dropna()
        
        if len(discovery) == 0 or len(validation) == 0:
            continue
        
        # Mann-Whitney U test
        stat, p_value = mannwhitneyu(discovery, validation, alternative='two-sided')
        
        # Calculate means
        disc_mean = discovery.mean()
        val_mean = validation.mean()
        
        batch_comparisons.append({
            'metric': metric,
            'discovery_mean': disc_mean,
            'validation_mean': val_mean,
            'difference': val_mean - disc_mean,
            'p_value': p_value,
            'significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    batch_df = pd.DataFrame(batch_comparisons)
    
    print("\nBatch Effect Assessment:")
    print(batch_df.to_string(index=False))
    
    # Count significant batch effects
    n_significant = (batch_df['p_value'] < 0.05).sum()
    print(f"\n{n_significant}/{len(batch_df)} metrics show significant batch effects (p < 0.05)")
    
    return batch_df


def generate_qc_plots(qc_df, output_dir):
    """
    Generate QC visualization plots.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        QC metrics dataframe
    output_dir : Path
        Directory to save plots
    """
    print("\nGenerating QC plots...")
    print("=" * 70)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Plot 1: QC metrics by batch
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QC Metrics by Batch', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('mean_mapq', 'Mean MAPQ'),
        ('mapped_pct', 'Mapped Reads (%)'),
        ('properly_paired_pct', 'Properly Paired (%)'),
        ('duplicate_rate', 'Duplicate Rate (%)'),
        ('mean_fragment_size', 'Mean Fragment Size (bp)'),
        ('conversion_efficiency', 'Conversion Efficiency')
    ]
    
    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        if metric in qc_df.columns:
            qc_df.boxplot(column=metric, by='batch', ax=ax)
            ax.set_title(label)
            ax.set_xlabel('Batch')
            ax.set_ylabel(label)
            plt.sca(ax)
            plt.xticks(rotation=0)
        else:
            ax.text(0.5, 0.5, f'{metric}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'qc_metrics_by_batch.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: qc_metrics_by_batch.png")
    
    # Plot 2: Conversion efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'conversion_efficiency' in qc_df.columns:
        qc_df['conversion_pct'] = qc_df['conversion_efficiency'] * 100
        
        sns.barplot(data=qc_df, x='sample_id', y='conversion_pct', 
                   hue='batch', ax=ax)
        
        # Add threshold line
        ax.axhline(y=BISULFITE_CONVERSION_THRESHOLD * 100, 
                  color='red', linestyle='--', linewidth=2,
                  label=f'Threshold ({BISULFITE_CONVERSION_THRESHOLD*100}%)')
        
        ax.set_title('Bisulfite Conversion Efficiency by Sample', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample ID', fontsize=12)
        ax.set_ylabel('Conversion Efficiency (%)', fontsize=12)
        ax.legend()
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
    
    plt.savefig(output_dir / 'conversion_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: conversion_efficiency.png")
    
    # Plot 3: Fragment size distribution by disease status
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'mean_fragment_size' in qc_df.columns:
        qc_df.boxplot(column='mean_fragment_size', by='disease_status', ax=ax)
        ax.set_title('Fragment Size by Disease Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('Disease Status', fontsize=12)
        ax.set_ylabel('Mean Fragment Size (bp)', fontsize=12)
        plt.sca(ax)
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fragment_size_by_disease.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fragment_size_by_disease.png")
    
    print(f"\n✓ All QC plots saved to: {output_dir}")


def generate_qc_report(qc_df, batch_df, output_file):
    """
    Generate text report summarizing QC results.
    
    Parameters
    ----------
    qc_df : pd.DataFrame
        QC metrics
    batch_df : pd.DataFrame
        Batch effect assessment
    output_file : Path
        Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("QUALITY CONTROL REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {len(qc_df)}\n")
        f.write(f"  ALS: {(qc_df['disease_status'] == 'als').sum()}\n")
        f.write(f"  Control: {(qc_df['disease_status'] == 'ctrl').sum()}\n")
        f.write(f"  Discovery batch: {(qc_df['batch'] == 'discovery').sum()}\n")
        f.write(f"  Validation batch: {(qc_df['batch'] == 'validation').sum()}\n\n")
        
        # Conversion efficiency summary
        f.write("BISULFITE CONVERSION EFFICIENCY\n")
        f.write("-" * 70 + "\n")
        if 'conversion_efficiency' in qc_df.columns:
            mean_conv = qc_df['conversion_efficiency'].mean()
            min_conv = qc_df['conversion_efficiency'].min()
            poor_samples = qc_df[qc_df['conversion_efficiency'] < BISULFITE_CONVERSION_THRESHOLD]
            
            f.write(f"Mean conversion efficiency: {mean_conv:.4f} ({mean_conv*100:.2f}%)\n")
            f.write(f"Minimum conversion efficiency: {min_conv:.4f} ({min_conv*100:.2f}%)\n")
            f.write(f"Threshold: {BISULFITE_CONVERSION_THRESHOLD:.4f} ({BISULFITE_CONVERSION_THRESHOLD*100:.2f}%)\n")
            f.write(f"Samples below threshold: {len(poor_samples)}\n")
            
            if len(poor_samples) > 0:
                f.write("\nSamples with poor conversion:\n")
                for idx, row in poor_samples.iterrows():
                    f.write(f"  - {row['sample_id']}: {row['conversion_efficiency']:.4f}\n")
        f.write("\n")
        
        # Batch effects summary
        f.write("BATCH EFFECT ASSESSMENT\n")
        f.write("-" * 70 + "\n")
        n_sig = (batch_df['p_value'] < 0.05).sum()
        f.write(f"Metrics with significant batch effects (p < 0.05): {n_sig}/{len(batch_df)}\n\n")
        
        if n_sig > 0:
            f.write("Significant batch effects:\n")
            sig_metrics = batch_df[batch_df['p_value'] < 0.05]
            for idx, row in sig_metrics.iterrows():
                f.write(f"  - {row['metric']}: p = {row['p_value']:.4f}\n")
        else:
            f.write("No significant batch effects detected.\n")
        f.write("\n")
        
        # QC metrics summary statistics
        f.write("QC METRICS SUMMARY STATISTICS\n")
        f.write("-" * 70 + "\n")
        
        summary_metrics = ['mean_mapq', 'mapped_pct', 'properly_paired_pct',
                          'duplicate_rate', 'mean_fragment_size', 'conversion_efficiency']
        
        for metric in summary_metrics:
            if metric in qc_df.columns:
                f.write(f"\n{metric}:\n")
                f.write(f"  Mean: {qc_df[metric].mean():.2f}\n")
                f.write(f"  Std: {qc_df[metric].std():.2f}\n")
                f.write(f"  Min: {qc_df[metric].min():.2f}\n")
                f.write(f"  Max: {qc_df[metric].max():.2f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("RECOMMENDED FILTERING CRITERIA\n")
        f.write("=" * 70 + "\n")
        f.write(f"MAPQ >= {MIN_MAPQ}\n")
        f.write(f"Fragment size: {MIN_FRAGMENT_SIZE}-{MAX_FRAGMENT_SIZE} bp\n")
        f.write("Proper pairs only\n")
        f.write("Exclude duplicates\n")
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ QC report saved to: {output_file}")


def run_module_1():
    """
    Run complete Module 1: Quality Control & Filtering pipeline.
    
    Returns
    -------
    tuple
        (qc_metrics DataFrame, batch_effects DataFrame)
    """
    print("\n" + "=" * 70)
    print("MODULE 1: Quality Control & Filtering")
    print("=" * 70)
    
    # Ensure output directories exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    QC_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load sample manifest from Module 0
    print(f"\nLoading sample manifest from: {SAMPLE_MANIFEST}")
    
    if not SAMPLE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Sample manifest not found: {SAMPLE_MANIFEST}\n"
            f"Please run Module 0 first."
        )
    
    manifest = pd.read_csv(SAMPLE_MANIFEST)
    print(f"✓ Loaded manifest: {len(manifest)} samples")
    
    # Run QC on all samples
    qc_df = run_qc_on_all_samples(manifest)
    
    # Assess batch effects
    batch_df = assess_batch_effects(qc_df)
    
    # Generate plots
    generate_qc_plots(qc_df, QC_FIGURES_DIR)
    
    # Generate report
    generate_qc_report(qc_df, batch_df, QC_REPORT)
    
    # Save QC metrics
    qc_df.to_csv(QC_METRICS, index=False)
    print(f"\n✓ QC metrics saved to: {QC_METRICS}")
    
    print("\n" + "=" * 70)
    print("MODULE 1 COMPLETE")
    print("=" * 70 + "\n")
    
    return qc_df, batch_df


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 1 as a standalone script
    qc_metrics, batch_effects = run_module_1()
    
    # Display summary
    print("\nQC Metrics Summary:")
    print(qc_metrics.describe())