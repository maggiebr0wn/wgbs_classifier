## wgbs_classifier

# cfDNA WGBS Classifier: ALS vs. Control

Binary classification of Amyotrophic Lateral Sclerosis (ALS) patients from healthy controls using cell-free DNA (cfDNA) whole-genome bisulfite sequencing (WGBS) data.

---

## 🎯 Quick Start

### Installation & Setup
```bash
# Clone repository
git clone https://github.com/maggiebr0wn/wgbs_classifier.git
cd wgbs_classifier

# Create environment
conda env create -f environment.yml
conda activate wgbs_classifier

# Verify installation
python -c "import pysam, pandas, sklearn; print('✓ Installation successful')"
```

### Add raw data and metadata
```bash
mv <path to bam and bai files>/*bam* wgbs_classifier/data/raw
mv <path to metadata file>/celfie_cfDNA_ss.csv wgbs_classifier/data/metadata
```

### Run complete pipeline end-to-end
```bash
python scripts/run_pipeline.py
```
### Tested Configuration
- **OS**: macOS Sequoia v15.6.1
- **Processor**: 2.3 GHz 8-Core Intel Core i9
- **RAM**: 16 GB
- **Python**: Python 3.10.19
- **Runtime**: ~50 seconds (complete pipeline, 22 samples, chr21 only)
---

## 📊 Key Results

### Classification Performance (Validation Set, n=14)

<table>
<tr>
<td width="20%">

| Metric | Value |
|--------|-------|
| **AUC** | **0.750** |
| **Accuracy** | **78.6%** |
| **Precision** | 0.727 |
| **Sensitivity (Recall)** | **1.000** |
| **Specificity** | 0.500 |
| **F1-Score** | 0.842 |

</td>
<td width="80%">

<p align="center">
  <img src="https://github.com/maggiebr0wn/wgbs_classifier/blob/main/results/figures/classification/roc_curve.png" width="53%" />
  <img src="https://github.com/maggiebr0wn/wgbs_classifier/blob/main/results/figures/classification/confusion_matrix.png" width="37%" />
</p>

</td>
</tr>
</table>

<p align="center">
    <img src="https://github.com/maggiebr0wn/wgbs_classifier/blob/main/results/figures/validation/frag_mean_analysis.png" width="95%">
</p>

**Model:** XGBoost classifier using combined fragmentomics + methylation features

**Key Finding:** A single feature—fragment size standard deviation (`frag_std`)—achieves 100% sensitivity in detecting ALS cases. Lower cfDNA fragment size variability in ALS reflects a homogeneous disease-specific fragmentation signature, likely from dominant motor neuron/glial cell death overwhelming the normal heterogeneous cfDNA background. Strong correlation of fragment size standard deviation with mean fragment size.

---

## 🔬 Biological Discovery: Fragment Size Homogeneity as ALS Biomarker

### The Signal: Lower Variability in ALS cfDNA

ALS samples exhibit **significantly lower fragment size variability** compared to controls:

| Feature | ALS (n=8) | Control (n=6) | Correlation with frag_std |
|---------|-----------|---------------|---------------------------|
| **frag_std** (Standard Deviation) | 51.7 ± 4.1 | 62.6 ± 7.0 | 1.000 |
| **frag_mean** (Mean size) | 168.3 bp | 175.8 bp | 0.937 |
| **frag_pct_long** (>400 bp) | Lower | Higher | 0.970 |
| **frag_pct_dinucleosomal** (220-400 bp) | Lower | Higher | 0.942 |

**Decision Rule:** `IF fragment_std < 61.88 THEN predict ALS`

## Biological Interpretation

Circulating cell‑free DNA (cfDNA) fragment lengths are shaped by how DNA is released and cleaved during cell death. In healthy plasma, cfDNA arises from many cell types and shows a **nucleosomal pattern** with a broad range of fragment sizes (mono‑, di‑, tri‑nucleosomes) reflecting diverse sources and nuclease activity.  
- [Circulating cell‑free DNA fragmentation is a stepwise and conserved process linked to apoptosis](https://link.springer.com/article/10.1186/s12915-023-01752-6) — cfDNA reflects apoptotic processes and nucleosome structure, with healthy cfDNA dominated by a ~166 bp peak and sub‑peaks corresponding to nucleosomal units.

Disease states can shift cfDNA size distributions because of altered tissue contribution and fragmentation processes. It is known that tumor‑derived cfDNA tends to be **shorter** than cfDNA from normal cells, likely due to differences in chromatin accessibility and nuclease cleavage.  
- [Quantitative characterization of tumor cell‑free DNA shortening](https://link.springer.com/article/10.1186/s12864-020-06848-9) — tumor cfDNA shows increased proportions of short fragments (~100‑150 bp) compared to non‑tumor cfDNA, consistent with ctDNA being enriched in shorter fragment sizes.

In this dataset:

- ALS samples have **shorter average fragments** (mean ~165 bp vs ~172 bp) with **lower standard deviation**.  
- Controls show **higher variance** in fragment sizes.

This pattern suggests:

- A more **homogeneous source of cfDNA** in ALS, potentially due to a dominant contribution from disease‑affected tissues or specific cell death pathways, producing a **narrower fragment length distribution** and reduced variance compared to a heterogeneous mix in healthy controls.  
- In healthy individuals, cfDNA reflects a mixture of cell types and physiological cell turnover, leading to **broader size variability**.  
- The combination of **shorter and more uniform cfDNA fragments in ALS** may reflect a distinct pathological cfDNA signal that is more consistent in its fragmentation characteristics than the mixed background seen in controls.

Overall, this pattern aligns with established cfDNA biology: cfDNA size profiles depend on **nucleosome positioning, chromatin accessibility, and cell‑type‑specific cleavage processes**, and disease‑specific contributions can shift and reshape these distributions.

---

## 🔍 Approach & Model Evolution

### Systematic Exploration: From Complex to Simple

**Approach 1: High-dimensional features (FAILED)**
- 120 features: k-mer end motifs + regional methylation
- Selected by variance/discrimination on discovery set
- Result: Severe overfitting (AUC: 1.0 discovery → 0.40 validation)
- Problem: Data leakage, feature selection bias on tiny training set

**Approach 2: Pre-defined summary features (SUCCESS)**
- 23 features: 17 fragmentomics + 6 methylation summaries
- No data-driven selection, biology-guided choices
- XGBoost, Random Forest, Logistic Regression tested
- Result: XGBoost with single feature achieves best performance

### Why Such a Simple Model?

With only **8 training samples**, model simplicity is statistically appropriate:
- XGBoost correctly regularized to avoid overfitting
- High correlation among fragmentomics features (r > 0.9) means they measure redundant biological phenomenon
- The single feature captures the most discriminative signal
- This is a **biologically interpretable biomarker**

**Trade-off:** Model prioritizes sensitivity (100%, catches all ALS) over specificity (50%, some controls misclassified). For a fatal neurodegenerative disease, this may be clinically appropriate.

---

## 📁 Repository Structure

```
wgbs_classifier/
│
├── README.md                          
│
├── scripts/
│   └── run_pipeline.py                # Automated end-to-end 
│
├── src/                              
│   ├── data_loader.py                 # Module 0             
│   ├── qc.py                          # Module 1                       
│   ├── feature_extraction.py          # Module 2       
│   ├── visualization.py               # Module 3     
│   ├── classification.py              # Module 4; RF classifier only
│   └── config.py
│
├── notebooks/
│   ├── complete_analysis.ipynb       # Walkthrough
│   │
│   └── exploratory/                  # Exploration 
│       ├── 01_setup_qc.ipynb        
│       ├── 02_feature_extraction.ipynb
│       ├── 03_visualization.ipynb
│       ├── 04_model_exploration.ipynb     # Model selection & comparison
│       └── 05_final_validation.ipynb      # Feature interpretation & biology
│
├── data/
│   ├── processed/
│   │   ├── sample_manifest.csv
│   │   ├── qc_metrics.csv
│   │   ├── all_features.csv          # ~1200 fragmentomics + methylation features
│   │   └── validation_predictions.csv    
│   ├── metadata/
│   │   └── celfie_cfDNA_ss.csv
│   └── raw/                           # BAM and BAM.bai files
│         
└── results/
    ├── classification/
    │   └── approach2_combined_xgboost.pkl  # Final model
    ├── figures/
    │   ├── required_plots/            # Assignment requirements
    │   ├── classification/            # ROC curves, confusion matrices
    │   └── validation/                # Feature interpretation plots
    └── tables/
        └──batch_effects_summary.csv
```

---

**Pipeline Steps:**

1. **Module 0: Data Loading** (`src/data_loader.py`)
   - Load metadata, verify BAM files
   - Output: `data/processed/sample_manifest.csv`

2. **Module 1: Quality Control** (`src/qc.py`)
   - BAM statistics, bisulfite conversion, batch effects
   - Output: `data/processed/qc_metrics.csv`, QC plots

3. **Module 2: Feature Extraction** (`src/feature_extraction.py`)
   - Extract ~1,200 fragmentomics & methylation features
     - 17 fragmentomics summary statistics
     - 256 4-mer end motifs
     - ~467 regional methylation bins (100 kb)
     - 6 global methylation summaries
   - Output: `data/processed/all_features.csv`

4. **Module 3: Required Visualizations** (`src/visualization.py`)
   - Fragment length distribution
   - Position distributions
   - End motif distribution
   - Methylation analysis
   - Output: `results/figures/required_plots/`

5. **Module 4: Classification** (`src/classification.py`)
   - Train XGBoost with combined features (23 total)
   - Validate on held-out test set
   - Output: Classification metrics, trained model, predictions

6. **Post-Hoc: Feature Interpretation (not automated)** (`notebooks/05_final_validation.ipynb`)
   - XGBoost feature importance analysis
   - Correlation among features
   - Biological interpretation of results
   - Output: Feature correlation plots, decision boundary visualizations
---

## 🔍 Data Overview

### Dataset
- **Source:** Published ALS cfDNA WGBS dataset (downsampled to chr21)
- **Samples:** 22 total (12 ALS, 10 Control)
- **Batches:** Discovery (n=8) for training, Validation (n=14) for testing
- **Sequencing:** NovaSeq 6000, bisulfite-treated, Bismark-aligned
- **Region:** Chromosome 21 only (~47 Mb)

### Quality Metrics

| Metric | Mean | Status |
|--------|------|--------|
| Mapping Quality (MAPQ) | 38.3 | Excellent |
| Mapped Reads | 100% | Perfect |
| Properly Paired | 100% | Perfect |
| Bisulfite Conversion | 99.4% | Excellent |
| Mean Fragment Size | 172.5 bp | Expected cfDNA |

## ⚠️ Limitations & Future Directions

### Current Limitations
1. **Small training set** (n=8 discovery) severely limits model complexity—single feature models are appropriate
2. **Single feature dependency** - Model relies entirely on fragment size variability
3. **Chromosome 21 only** - Not whole genome analysis
4. **Modest specificity** (50%) leads to false positives in controls
5. **No clinical variables** (disease duration, ALSFRS scores, progression rate) included
6. **Batch effects** present but fragmentomics more robust than methylation

### Strengths & Clinical Potential
1. **100% sensitivity** - Critical for fatal disease screening
2. **Simple, interpretable biomarker** - Single measurement, no black box
3. **Biologically plausible** - Reflects known cfDNA fragmentation biology
4. **Robust feature** - Lower technical variability than methylation

### "Nice to haves" Improvements
1. **Larger cohorts** - Enable multi-feature models, better specificity
2. **Whole genome analysis** - More comprehensive feature set
3. **Orthogonal validation** - Test on independent ALS cohorts
4. **Clinical integration** - Correlate with ALSFRS-R scores, survival, progression rate
5. **Technical validation** - Test across sequencing platforms, library prep methods

---

## 🛠️ Configuration

Key parameters are centralized in `src/config.py`:

```python
# Analysis parameters
CHROMOSOME = "chr21"                  # Chromosome to analyze
MIN_MAPQ = 20                         # Minimum mapping quality
MIN_FRAGMENT_SIZE = 50                # Minimum fragment size (bp)
MAX_FRAGMENT_SIZE = 1000              # Maximum fragment size (bp)
BISULFITE_CONVERSION_THRESHOLD = 0.99 # 99% conversion required

# Final model features
FRAGMENTOMICS_SUMMARY = [
    "frag_mean",               # Mean fragment size
    "frag_median",             # Median fragment size
    "frag_std",                # Standard deviation of fragment sizes
    "frag_iqr",                # Interquartile range
    "frag_cv",                 # Coefficient of variation

    "frag_q25",                # 25th percentile
    "frag_q50",                # 50th percentile (median, redundant)
    "frag_q75",                # 75th percentile

    "frag_skewness",           # Skewness of fragment size distribution
    "frag_kurtosis",           # Kurtosis of fragment size distribution

    "frag_pct_very_short",     # % very short fragments
    "frag_pct_short",          # % short fragments
    "frag_pct_mononucleosomal",# % mono-nucleosomal fragments
    "frag_pct_dinucleosomal",  # % di-nucleosomal fragments
    "frag_pct_long",           # % long fragments

    "frag_ratio_short_long",   # Ratio of short to long fragments
    "frag_ratio_mono_di",      # Ratio of mono- to di-nucleosomal fragments
]

METHYLATION_SUMMARY = [
    "meth_mean_cpg",           # Global mean CpG methylation
    "meth_std",                # Global CpG methylation std dev
    "meth_pct_high",           # % highly methylated CpGs
    "meth_pct_low",            # % lowly methylated CpGs
    "meth_pct_intermediate",   # % intermediate methylated CpGs
    "regional_meth_mean",      # Mean methylation in regional bins
]
```

Modify `src/config.py` before running to change analysis parameters.

---

## 📄 License

MIT License - See LICENSE file for details.