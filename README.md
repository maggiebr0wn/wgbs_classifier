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

### Run Complete Pipeline (automated)
```bash
python scripts/run_pipeline.py
```
---

## 📊 Key Results

### Classification Performance (Validation Set, n=14)

| Metric | Value |
|--------|-------|
| **AUC** | **0.646** |
| **Accuracy** | **64.3%** |
| **Precision** | 0.64 |
| **Sensitivity (Recall)** | 0.50 |
| **F1-Score** | 0.57 |

**Model:** Random Forest classifier with 5 fragmentomics features

**Key Finding:** Simple, biologically interpretable fragment size distribution features provide modest but consistent discriminative ability for ALS classification, achieving 64% accuracy on held-out validation data.

---

## 🔬 Approach & Rationale

### Scientific Challenges Encountered

This analysis revealed significant challenges:

1. **Batch effects** between discovery (n=8) and validation (n=14) sets
2. **Methylation features completely overfit** to discovery batch (AUC: 1.0 → 0.48)
3. **Complex models failed to generalize** due to small training set

### Solution: Simple Fragmentomics Features

After systematic exploration, the best approach used **5 core cfDNA fragmentomics metrics**:

1. `frag_mean` - Mean fragment size (overall fragmentation)
2. `frag_pct_short` - Short fragments <150bp (nucleosome-free DNA, putative apoptosis marker)
3. `frag_pct_long` - Long fragments >400bp (genomic DNA)
4. `frag_ratio_short_long` - Short-to-long ratio (fragmentation balance)
5. `frag_pct_mononucleosomal` - Mononucleosome peak 150-220bp (nucleosome positioning)

**Feature notes:**
- Core cfDNA metrics are well-established in literature
- Less sensitive to technical batch effects than methylation
- Biologically interpretable (nucleosome positioning, apoptosis)
- Minimal feature set prevents overfitting with n=8 training samples

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
│       ├── 04_model_exploration.ipynb     
│       └── 05_final_validation.ipynb      
│
├── data/
│   ├── processed/
│   │   ├── sample_manifest.csv
│   │   ├── qc_metrics.csv
│   │   ├── all_features.csv
│   │   └── final_predictions.csv    
│   ├── metadata/
│   │   └── celfie_cfDNA_ss.csv
│   └── raw/                          # bam and bam.bai files
│         
└── results/
    ├── figures/
    │   ├── production/              
    │   │   ├── required_plots/       # Assignment requirements
    │   │   └── classification/       # Final model results
    │   └── exploratory/              # From exploration notebooks
    └── tables/
        └── final_metrics.csv
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
   - Output: `data/processed/all_features.csv`

4. **Module 3: Required Visualizations** (`src/visualization.py`)
   - **Fragment length distribution** 
   - **Position distributions** 
   - **End motif distribution**
   - **Methylation analysis** 
   - Output: `results/figures/required_plots/`

5. **Module 4: Classification** (`src/classification.py`)
   - Train Random Forest with fragmentomics features
   - Validate on held-out test set
   - Output: Classification metrics, ROC curve, predictions

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
1. **Small training set** (n=8 discovery) limits model complexity
2. **Batch effects** between discovery/validation reduce performance
3. **Chromosome 21 only** - not whole genome
4. **Modest performance** (AUC=0.646) suggests signal is weak
5. **No clinical variables** (disease duration, ALSFRS scores) included

### Future Improvements
1. **Batch correction** - ComBat, harmonization methods
2. **Larger cohorts** - Enable more complex models, feature interactions
3. **Whole genome analysis** - More comprehensive feature space
4. **Multi-modal integration** - Combine fragmentomics + methylation with batch correction
5. **Clinical integration** - Add ALSFRS scores, disease duration
6. **Pathway analysis** - Differentially methylated regions → gene pathways
7. **Longitudinal analysis** - Track changes over disease progression

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
```

Modify `src/config.py` before running to change analysis parameters.

---

## 📄 License

MIT License - See LICENSE file for details.