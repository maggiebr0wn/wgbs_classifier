# wgbs_classifier

A modular pipeline for binary classification of ALS vs. Control samples using cell-free DNA (cfDNA) whole-genome bisulfite sequencing (WGBS) data.

## 📋 Overview

Analyze cfDNA WGBS samples to identify features that distinguish ALS patients from healthy controls. The analysis focuses on chromosome 21 data from a published dataset and extracts multi-modal features including:

- **Fragment size distributions** (fragmentomics)
- **End motif patterns** (4-mer frequencies from fragment ends)
- **Methylation profiles** (CpG methylation from bisulfite-treated DNA)
- **Positional features** (fragment start/end distributions)

**Goal:** Build a binary classifier to distinguish ALS from Control samples and identify discriminative biomarkers.

---

## 🗂️ Repository Structure
```
wgbs_classifier/
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies (pip)
├── environment.yml              # Conda environment specification
├── .gitignore
│
├── src/                         # Source code (modular Python scripts)
│   ├── __init__.py
│   ├── config.py                # Configuration: paths, parameters, constants
│   ├── data_loader.py           # Module 0: Load metadata, verify BAM files
│   ├── qc.py                    # Module 1: Quality control & batch assessment
│   ├── feature_extraction.py    # Module 2: Extract features from BAM files
│   ├── visualization.py         # Module 3: Generate plots & EDA (IN PROGRESS)
│   ├── feature_preparation.py   # Module 4: Feature selection & scaling (TODO)
│   └── classification.py        # Module 5: Train classifiers & evaluate (TODO)
│
├── notebooks/                   # Jupyter notebooks (interactive analysis)
│   ├── 01_setup_and_qc.ipynb           # Module 0 & 1: Setup, QC, batch effects
│   ├── 02_qc_analysis.ipynb            # Module 1: Detailed QC analysis
│   ├── 03_feature_extraction.ipynb     # Module 2: Feature extraction ✓
│   └── 04_exploratory_analysis.ipynb   # Module 3: EDA & required plots (TODO)
│
├── scripts/
│   └── run_pipeline.py          # Automated pipeline execution (TODO)
│
├── data/
│   ├── metadata/
│   │   └── sample_metadata.csv         # Sample information (user-provided)
│   ├── raw/                            # BAM files (not tracked in git)
│   └── processed/                      # Generated data files
│       ├── sample_manifest.csv         # Module 0 output
│       ├── qc_metrics.csv              # Module 1 output
│       └── all_features.csv            # Module 2 output (~370 features)
│
├── results/                     # Analysis outputs (not tracked in git)
│   ├── figures/
│   │   └── qc/                         # QC plots from Module 1
│   └── tables/
│
└── models/                      # Saved model objects (not tracked in git)
```

---

## 🚀 Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Git

### Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/YOUR_USERNAME/wgbs_classifier.git
   cd wgbs_classifier
```

2. **Create conda environment:**
```bash
   conda env create -f environment.yml
```

3. **Activate the environment:**
```bash
   conda activate wgbs_classifier
```

4. **Register Jupyter kernel (for notebooks):**
```bash
   python -m ipykernel install --user --name wgbs_classifier --display-name "Python (wgbs_classifier)"
```

5. **Verify installation:**
```bash
   python -c "import pysam, pandas, sklearn; print('✓ Installation successful!')"
```

---

## 📊 Data Preparation

### Required Input Data

1. **Sample metadata:** Place `sample_metadata.csv` in `data/metadata/`
   - Required columns: `Run`, `disease_status`, `batch`, `AGE`
   - Example provided in repository

2. **BAM files:** Place BAM files and their index files in `data/raw/`
```
   data/raw/
   ├── SRR13404367.*.bam
   ├── SRR13404367.*.bam.bai
   └── ...
```
   - Files should match the `Run` IDs in metadata
   - Both `.bam` and `.bam.bai` (index) files required

---

## 📖 Analysis Pipeline

### Module Overview

The pipeline consists of 6 modules that can be run interactively (Jupyter notebooks) or as automated scripts:

| Module | Description | Status | Notebook | Output |
|--------|-------------|--------|----------|--------|
| **0** | Setup & Data Loading | ✅ Complete | `01_setup_and_qc.ipynb` | `sample_manifest.csv` |
| **1** | Quality Control & Filtering | ✅ Complete | `01_setup_and_qc.ipynb`<br>`02_qc_analysis.ipynb` | `qc_metrics.csv`<br>QC plots & report |
| **2** | Feature Extraction | ✅ Complete | `03_feature_extraction.ipynb` | `all_features.csv` |
| **3** | Exploratory Analysis | 🔄 In Progress | `04_exploratory_analysis.ipynb` | Required plots |
| **4** | Feature Preparation | 📝 TODO | TBD | Selected features, scaler |
| **5** | Classification | 📝 TODO | TBD | Models, metrics, results |

---

### Running the Analysis

#### Option 1: Interactive (Jupyter Notebooks) - Recommended
```bash
# Start Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_setup_and_qc.ipynb       # Modules 0 & 1
# 2. notebooks/02_qc_analysis.ipynb        # Module 1 (detailed)
# 3. notebooks/03_feature_extraction.ipynb # Module 2
# 4. notebooks/04_exploratory_analysis.ipynb # Module 3 (in progress)
```

Make sure to select the **"Python (wgbs_classifier)"** kernel when opening notebooks.

#### Option 2: Run Individual Modules (Scripts)
```bash
# Module 0: Setup & Data Loading
python src/data_loader.py

# Module 1: Quality Control
python src/qc.py

# Module 2: Feature Extraction
python src/feature_extraction.py

# Module 3+: Coming soon
```

#### Option 3: Automated Pipeline (Coming Soon)
```bash
python scripts/run_pipeline.py
```

---

## 📈 Current Progress

### ✅ Completed Modules

#### **Module 0: Setup & Data Loading**
- Loads sample metadata (22 samples: 12 ALS, 10 Control)
- Verifies BAM file availability
- Creates sample manifest with metadata
- **Output:** `data/processed/sample_manifest.csv`

#### **Module 1: Quality Control & Filtering**
- Calculates BAM-level statistics (mapping quality, paired reads, duplicates)
- Assesses bisulfite conversion efficiency (mean: 99.41%, 2 samples flagged)
- Evaluates batch effects on QC metrics
  - **Significant batch effects found:** Fragment size (p=0.0064), CpG methylation (p=0.024)
- Generates QC plots stratified by disease and batch
- **Outputs:** 
  - `data/processed/qc_metrics.csv`
  - `results/qc_report.txt`
  - QC plots in `results/figures/qc/`

**Key Findings:**
- Batch effect on fragment size (11.6 bp) > Disease effect (5.8 bp)
- Strategy: Use Discovery/Validation split to control batch effects
- All samples have acceptable quality (conversion ≥ 97%)

#### **Module 2: Feature Extraction**
- Extracts ~370 features from BAM files:
  - **Fragment size features** (~20): summary stats, size bins, distribution shape
  - **Position features** (~104): coverage bins across chr21, start/end positions
  - **End motif features** (~258): 4-mer frequencies, diversity, GC content
  - **Methylation features** (~10): CpG rates, variance, per-read statistics
- Applies consistent filtering: MAPQ≥20, proper pairs, 50-1000bp fragments
- **Output:** `data/processed/all_features.csv` (22 samples × 374 columns)

---

### 🔄 In Progress

#### **Module 3: Exploratory Analysis & Visualization**
- Generate required plots for assignment:
  - Fragment length distribution
  - Start/end position distributions
  - End motif distribution
  - Methylation analysis
- PCA and clustering analysis
- Batch effect visualization on features
- Feature correlation analysis

---

### 📝 TODO

#### **Module 4: Feature Preparation**
- Train/test split (Discovery vs. Validation batch)
- Feature selection (identify top discriminative features)
- Feature scaling (StandardScaler)
- Handle batch effects and confounders

#### **Module 5: Classification**
- Train binary classifiers (Logistic Regression, Random Forest)
- Evaluate on test set
- **Report required metrics:**
  - Precision
  - Sensitivity (Recall)
  - F1-Score
- Feature importance analysis
- Model interpretation

#### **Module 6: Documentation & Summary**
- Results summary
- Biological interpretation
- Limitations and future directions

---

## 🔬 Analysis Details

### Data Characteristics

- **Dataset:** 22 cfDNA WGBS samples (chr21 only, downsampled)
  - 12 ALS patients
  - 10 Healthy controls
  - Split into Discovery (n=8) and Validation (n=14) batches
- **Sequencing:** NovaSeq 6000, bisulfite-treated, Bismark-aligned
- **Average fragment size:** 172.5 bp (typical cfDNA mono-nucleosome)
- **Conversion efficiency:** 99.41% average

### Quality Control Summary

| Metric | Mean | Status |
|--------|------|--------|
| Mapping Quality (MAPQ) | 38.3 | ✅ Excellent |
| Mapped Reads | 100% | ✅ Perfect |
| Properly Paired | 100% | ✅ Perfect |
| Duplicate Rate | 0% | ✅ Pre-filtered |
| Bisulfite Conversion | 99.41% | ✅ Excellent |
| Mean Fragment Size | 172.5 bp | ✅ Expected cfDNA |

### Batch Effects

**Identified batch effects:**
- Fragment size: Discovery (179.9 bp) vs. Validation (168.3 bp), p=0.0064
- CpG methylation: p=0.024

**Strategy:** Use Discovery batch for training, Validation batch for testing to naturally control batch effects and test generalization.

---

## 🛠️ Configuration

Key parameters are centralized in `src/config.py`:
```python
# Analysis parameters
CHROMOSOME = "chr21"                           # Chromosome to analyze
MIN_MAPQ = 20                                  # Minimum mapping quality
MIN_FRAGMENT_SIZE = 50                         # Minimum fragment size (bp)
MAX_FRAGMENT_SIZE = 1000                       # Maximum fragment size (bp)
BISULFITE_CONVERSION_THRESHOLD = 0.99          # 99% conversion required
KMER_SIZE = 4                                  # K-mer size for motif analysis
```

To modify parameters, edit `src/config.py` before running the pipeline.

---

## 📚 Dependencies

### Core Libraries

- **pysam** (≥0.22.0) - BAM file processing
- **pandas** (≥2.0.0) - Data manipulation
- **numpy** (≥1.24.0) - Numerical computing
- **scipy** (≥1.10.0) - Statistical analysis
- **scikit-learn** (≥1.3.0) - Machine learning
- **matplotlib** (≥3.7.0), **seaborn** (≥0.12.0) - Visualization
- **jupyter** - Interactive analysis

See `environment.yml` or `requirements.txt` for complete list.

---

## 📖 Tutorials & Documentation

### Understanding WGBS and cfDNA

For background on the methods used in this pipeline, see:
- `docs/WGBS_cfDNA_Study_Guide.pdf` (comprehensive guide created during development)

### Key Concepts

- **WGBS (Whole Genome Bisulfite Sequencing):** Bisulfite converts unmethylated C→T, preserves methylated C
- **cfDNA (Cell-Free DNA):** Short DNA fragments in blood, released by dying cells
- **Fragmentomics:** Analysis of cfDNA fragment patterns (size, ends, positions)
- **XM tag:** Bismark methylation string (Z/z for CpG, X/x for CHG, H/h for CHH)

### Module-Specific Documentation

Each module has detailed docstrings explaining:
- Purpose and inputs/outputs
- Function parameters
- Feature definitions
- Usage examples

Example:
```python
from src.feature_extraction import extract_fragment_features
help(extract_fragment_features)
```

---

## 📄 License

MIT License - see `LICENSE` file for details.

---