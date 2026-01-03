import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Since you are running from the project root, we point to data/processed/
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / 'data' / 'processed' / 'all_features.csv'
RESULTS_DIR = BASE_DIR / 'results'
PCA_DIR = RESULTS_DIR / 'pca'

# Ensure directories exist
PCA_DIR.mkdir(parents=True, exist_ok=True)

def find_significant_features(df, feature_cols, p_thresh=0.1, top_n_fallback=20):
    """Identifies features with the most separation in the Discovery set."""
    als = df[df['disease_status'] == 'als']
    ctrl = df[df['disease_status'] == 'ctrl']
    stats = []
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        a, c = als[col].dropna(), ctrl[col].dropna()
        if len(a) >= 3 and len(c) >= 3:
            try:
                u, p = mannwhitneyu(a, c, alternative='two-sided')
                stats.append({'feature': col, 'p_value': p, 'als_mean': a.mean(), 'ctrl_mean': c.mean()})
            except: continue
            
    stats_df = pd.DataFrame(stats).sort_values('p_value')
    sig_features = stats_df[stats_df['p_value'] <= p_thresh]['feature'].tolist()
    
    # Fallback if few features are significant at small N
    if len(sig_features) < 5:
        sig_features = stats_df.head(top_n_fallback)['feature'].tolist()
    return sig_features, stats_df

def run_supervised_pca_pipeline(df_disc, df_val, features, block_name, n_comp=2):
    """Fits PCA on Discovery and projects Validation samples."""
    X_d, X_v = df_disc[features].copy(), df_val[features].copy()
    medians = X_d.median()
    X_d, X_v = X_d.fillna(medians).fillna(0), X_v.fillna(medians).fillna(0)
    
    scaler = StandardScaler()
    X_d_scaled = scaler.fit_transform(X_d)
    X_v_scaled = scaler.transform(X_v)
    
    pca = PCA(n_components=n_comp)
    pcs_d = pca.fit_transform(X_d_scaled)
    pcs_v = pca.transform(X_v_scaled)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pcs_d[:, 0], y=pcs_d[:, 1], hue=df_disc['disease_status'], 
                    palette='Set1', s=150, edgecolors='black', alpha=0.9)
    plt.title(f"Discovery: Supervised {block_name} PCA\n({len(features)} Significant Features)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(PCA_DIR / f"pca_supervised_{block_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    cols = [f'{block_name.replace(" ", "_")}_PC{i+1}' for i in range(n_comp)]
    return pd.DataFrame(pcs_d, columns=cols, index=df_disc.index), \
           pd.DataFrame(pcs_v, columns=cols, index=df_val.index)

def run_module_4_supervised():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find input file at {INPUT_FILE}. "
                                f"Current working directory is {os.getcwd()}")

    df = pd.read_csv(INPUT_FILE)
    df_disc = df[df['batch'] == 'discovery'].copy()
    df_val = df[df['batch'] == 'validation'].copy()
    
    frag_cols = [c for c in df.columns if c.startswith('frag_')]
    meth_cols = [c for c in df.columns if c.startswith('regional_meth_bin_')]
    
    # 1. Feature Selection (Discovery ONLY)
    sig_frag, frag_stats = find_significant_features(df_disc, frag_cols)
    sig_meth, meth_stats = find_significant_features(df_disc, meth_cols)
    
    # 2. Perform 3 PCAs
    # PCA 1: Fragmentation Only
    pca1_d, pca1_v = run_supervised_pca_pipeline(df_disc, df_val, sig_frag, "Fragmentation")
    
    # PCA 2: Methylation Only
    pca2_d, pca2_v = run_supervised_pca_pipeline(df_disc, df_val, sig_meth, "Methylation")
    
    # PCA 3: Combined
    combined_features = list(set(sig_frag + sig_meth))
    pca3_d, pca3_v = run_supervised_pca_pipeline(df_disc, df_val, combined_features, "Combined")
    
    # 3. Assemble and Save
    meta = [c for c in ['sample_id', 'disease_status', 'batch'] if c in df.columns]
    final_disc = pd.concat([df_disc[meta], pca1_d, pca2_d, pca3_d], axis=1)
    final_val = pd.concat([df_val[meta], pca1_v, pca2_v, pca3_v], axis=1)
    
    final_disc.to_csv(RESULTS_DIR / 'supervised_features_discovery.csv', index=False)
    final_val.to_csv(RESULTS_DIR / 'supervised_features_validation.csv', index=False)
    
    return final_disc, final_val

# Run the pipeline
selected_disc, selected_val = run_module_4_supervised()