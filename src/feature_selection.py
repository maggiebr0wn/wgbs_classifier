"""
Module 4: Feature Selection, Training & Validation (Consolidated)

ONE SCRIPT that does everything:
1. Feature selection (choose LASSO or Random Forest)
2. Train model on discovery set
3. LOO cross-validation on discovery
4. Test on validation set
5. Generate all visualizations
6. Save results

Usage:
    python feature_selection_consolidated.py --model lasso
    python feature_selection_consolidated.py --model rf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, 
    precision_score, recall_score, f1_score, confusion_matrix
)

# Import configuration
from src.config import (
    ALL_FEATURES,
    DISCOVERY_BATCH,
    VALIDATION_BATCH,
    METHYLATION_AGGREGATION_SIZE,
    FEATURE_SELECTION_DIR,
    FIGURES_DIR,
    RESULTS_DIR
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature selection parameters
MIN_SAMPLE_COVERAGE = 0.25
MIN_VARIANCE = 0.0001
N_TOP_FEATURES = 20  # Total features to select

# LASSO parameters
LASSO_C_VALUES = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 3,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'class_weight': 'balanced',
    'n_jobs': -1
}


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load and split data into discovery/validation."""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(ALL_FEATURES)
    discovery_df = df[df['batch'] == DISCOVERY_BATCH].copy()
    validation_df = df[df['batch'] == VALIDATION_BATCH].copy()
    
    print(f"\nDiscovery: {len(discovery_df)} samples "
          f"({(discovery_df['disease_status']=='als').sum()} ALS, "
          f"{(discovery_df['disease_status']=='ctrl').sum()} Control)")
    print(f"Validation: {len(validation_df)} samples "
          f"({(validation_df['disease_status']=='als').sum()} ALS, "
          f"{(validation_df['disease_status']=='ctrl').sum()} Control)")
    
    return discovery_df, validation_df


# ============================================================================
# STEP 2: EXTRACT & FILTER FEATURES
# ============================================================================

def extract_features(discovery_df):
    """Extract and filter all features."""
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)
    
    # Fragmentomics features
    frag_patterns = ['frag_mean', 'frag_median', 'frag_std', 'frag_iqr', 
                     'frag_cv', 'frag_q25', 'frag_q50', 'frag_q75',
                     'frag_skewness', 'frag_kurtosis',
                     'frag_pct_', 'frag_ratio_']
    
    frag_cols = []
    for pattern in frag_patterns:
        frag_cols.extend([c for c in discovery_df.columns if pattern in c])
    frag_cols = sorted(set(frag_cols))
    
    # Methylation features (aggregate to 500kb)
    print(f"\nAggregating methylation to 500kb bins...")
    meth_cols_100kb = [c for c in discovery_df.columns 
                       if c.startswith('regional_meth_bin_') 
                       and c.replace('regional_meth_bin_', '').isdigit()]
    
    aggregation_factor = METHYLATION_AGGREGATION_SIZE // 100_000
    aggregated_features = {}
    
    for sample_idx in discovery_df.index:
        sample_data = {}
        for meth_col in meth_cols_100kb:
            bin_num = int(meth_col.replace('regional_meth_bin_', ''))
            agg_bin = bin_num // aggregation_factor
            agg_col_name = f'meth_agg_{agg_bin}'
            
            if agg_col_name not in sample_data:
                sample_data[agg_col_name] = []
            
            value = discovery_df.loc[sample_idx, meth_col]
            if pd.notna(value):
                sample_data[agg_col_name].append(value)
        
        for agg_col, values in sample_data.items():
            if len(values) >= 1:
                aggregated_features.setdefault(agg_col, {})[sample_idx] = np.mean(values)
    
    # Create aggregated methylation dataframe
    meth_agg_df = pd.DataFrame(aggregated_features)
    meth_cols = list(meth_agg_df.columns)
    
    # Combine features
    feature_df = discovery_df[['sample_id', 'disease_status', 'batch', 'age']].copy()
    for col in frag_cols:
        feature_df[col] = discovery_df[col]
    for col in meth_cols:
        if col in meth_agg_df.columns:
            feature_df[col] = meth_agg_df[col]
    
    # Filter by coverage
    n_samples = len(discovery_df)
    min_samples = int(np.ceil(n_samples * MIN_SAMPLE_COVERAGE))
    
    all_feature_cols = frag_cols + meth_cols
    coverage = feature_df[all_feature_cols].notna().sum()
    passing_coverage = coverage[coverage >= min_samples].index.tolist()
    
    # Filter by variance
    variances = feature_df[passing_coverage].var()
    passing_variance = variances[variances >= MIN_VARIANCE].index.tolist()
    
    print(f"\nFragmentomics: {len(frag_cols)} → {len([f for f in passing_variance if f in frag_cols])}")
    print(f"Methylation: {len(meth_cols)} → {len([f for f in passing_variance if f in meth_cols])}")
    print(f"Total features: {len(passing_variance)}")
    
    return feature_df, passing_variance, meth_agg_df


# ============================================================================
# STEP 3: FEATURE SELECTION
# ============================================================================

def select_features_lasso(feature_df, feature_cols):
    """Select features using LASSO regularization."""
    print("\n" + "="*70)
    print("FEATURE SELECTION: LASSO")
    print("="*70)
    
    X = feature_df[feature_cols].fillna(feature_df[feature_cols].median()).values
    y = (feature_df['disease_status'] == 'als').astype(int).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different C values
    best_n_features = 0
    best_C = None
    
    for C in LASSO_C_VALUES:
        model = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=5000, random_state=42)
        model.fit(X_scaled, y)
        n_features = np.sum(model.coef_[0] != 0)
        
        if n_features > 0 and n_features <= N_TOP_FEATURES:
            best_n_features = n_features
            best_C = C
    
    # Train with best C
    if best_C is None:
        best_C = 1.0
    
    model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, max_iter=5000, random_state=42)
    model.fit(X_scaled, y)
    
    # Get selected features
    coefs = model.coef_[0]
    selected_idx = np.where(coefs != 0)[0]
    selected_features = [feature_cols[i] for i in selected_idx]
    
    # If too few selected, add by absolute coefficient
    if len(selected_features) < 5:
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        selected_features = [feature_cols[i] for i in sorted_idx[:N_TOP_FEATURES]]
    
    print(f"\nSelected {len(selected_features)} features (C={best_C})")
    for feat in selected_features:
        print(f"  - {feat}")
    
    return selected_features


def select_features_rf(feature_df, feature_cols):
    """Select features using Random Forest importance."""
    print("\n" + "="*70)
    print("FEATURE SELECTION: RANDOM FOREST")
    print("="*70)
    
    X = feature_df[feature_cols].fillna(feature_df[feature_cols].median()).values
    y = (feature_df['disease_status'] == 'als').astype(int).values
    
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    selected_features = [feature_cols[i] for i in sorted_idx[:N_TOP_FEATURES]]
    
    print(f"\nSelected {len(selected_features)} features by importance")
    for feat, idx in zip(selected_features, sorted_idx[:N_TOP_FEATURES]):
        print(f"  - {feat}: {importances[idx]:.4f}")
    
    return selected_features


# ============================================================================
# STEP 4: TRAIN FINAL MODEL WITH LOO-CV
# ============================================================================

def train_with_loo(feature_df, selected_features, model_type='lasso'):
    """Train final model with LOO cross-validation."""
    print("\n" + "="*70)
    print(f"TRAINING {model_type.upper()} WITH LOO-CV")
    print("="*70)
    
    X = feature_df[selected_features].fillna(feature_df[selected_features].median()).values
    y = (feature_df['disease_status'] == 'als').astype(int).values
    
    print(f"\nTraining: {X.shape[0]} samples × {X.shape[1]} features")
    
    # LOO Cross-validation
    loo = LeaveOneOut()
    loo_predictions = []
    loo_true = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if model_type == 'lasso':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=5000, random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[0, 1]
        else:  # rf
            model = RandomForestClassifier(**RF_PARAMS)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[0, 1]
        
        loo_predictions.append(y_pred_proba)
        loo_true.append(y_test[0])
    
    loo_predictions = np.array(loo_predictions)
    loo_true = np.array(loo_true)
    
    try:
        loo_auc = roc_auc_score(loo_true, loo_predictions)
    except:
        loo_auc = 0.5
    
    print(f"\nLOO-CV AUC: {loo_auc:.3f}")
    
    if loo_auc >= 0.95 and len(X) <= 10:
        print("\n⚠️  WARNING: Very high LOO-CV AUC with small n!")
        print("    This may indicate overfitting. Validation will tell.")
    
    # Train final model on all data
    if model_type == 'lasso':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        final_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=5000, random_state=42)
        final_model.fit(X_scaled, y)
        model_data = {'model': final_model, 'scaler': scaler}
    else:
        final_model = RandomForestClassifier(**RF_PARAMS)
        final_model.fit(X, y)
        model_data = {'model': final_model}
    
    model_data.update({
        'selected_features': selected_features,
        'loo_auc': loo_auc,
        'model_type': model_type
    })
    
    return model_data, loo_predictions, loo_true


# ============================================================================
# STEP 5: VALIDATE
# ============================================================================

def validate(validation_df, meth_agg_template_df, model_data):
    """Test on validation set."""
    print("\n" + "="*70)
    print("VALIDATION TESTING")
    print("="*70)
    
    selected_features = model_data['selected_features']
    
    # Prepare validation methylation aggregation
    meth_cols_100kb = [c for c in validation_df.columns 
                       if c.startswith('regional_meth_bin_') 
                       and c.replace('regional_meth_bin_', '').isdigit()]
    
    aggregation_factor = METHYLATION_AGGREGATION_SIZE // 100_000
    aggregated_features = {}
    
    for sample_idx in validation_df.index:
        sample_data = {}
        for meth_col in meth_cols_100kb:
            bin_num = int(meth_col.replace('regional_meth_bin_', ''))
            agg_bin = bin_num // aggregation_factor
            agg_col_name = f'meth_agg_{agg_bin}'
            
            if agg_col_name not in sample_data:
                sample_data[agg_col_name] = []
            
            value = validation_df.loc[sample_idx, meth_col]
            if pd.notna(value):
                sample_data[agg_col_name].append(value)
        
        for agg_col, values in sample_data.items():
            if len(values) >= 1:
                aggregated_features.setdefault(agg_col, {})[sample_idx] = np.mean(values)
    
    val_meth_df = pd.DataFrame(aggregated_features)
    
    # Build validation feature dataframe
    val_feature_df = validation_df[['sample_id', 'disease_status', 'batch', 'age']].copy()
    for feat in selected_features:
        if feat in validation_df.columns:
            val_feature_df[feat] = validation_df[feat]
        elif feat in val_meth_df.columns:
            val_feature_df[feat] = val_meth_df[feat]
        else:
            val_feature_df[feat] = np.nan
    
    X_val = val_feature_df[selected_features].fillna(val_feature_df[selected_features].median()).values
    y_val = (validation_df['disease_status'] == 'als').astype(int).values
    
    # Predict
    if model_data['model_type'] == 'lasso':
        X_val_scaled = model_data['scaler'].transform(X_val)
        y_pred_proba = model_data['model'].predict_proba(X_val_scaled)[:, 1]
    else:
        y_pred_proba = model_data['model'].predict_proba(X_val)[:, 1]
    
    try:
        val_auc = roc_auc_score(y_val, y_pred_proba)
    except:
        val_auc = 0.5
    
    val_acc = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
    
    print(f"\nValidation AUC: {val_auc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    print(f"\nDiscovery LOO-CV: {model_data['loo_auc']:.3f}")
    print(f"Difference: {val_auc - model_data['loo_auc']:+.3f}")
    
    # Interpretation
    if val_auc >= 0.75:
        print("\n✓ GOOD: Model generalizes!")
    elif val_auc >= 0.60:
        print("\n⚠ MODERATE: Weak generalization")
    else:
        print("\n✗ POOR: Does not generalize")
    
    predictions_df = validation_df[['sample_id', 'disease_status', 'age']].copy()
    predictions_df['pred_proba'] = y_pred_proba
    predictions_df['pred_label'] = (y_pred_proba >= 0.5).astype(int)
    predictions_df['true_label'] = y_val
    
    return predictions_df, val_auc, val_acc


# ============================================================================
# STEP 6: VISUALIZE
# ============================================================================

def visualize_results(model_data, loo_true, loo_pred, val_predictions, val_auc):
    """Create summary visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    output_dir = FIGURES_DIR / f"{model_data['model_type']}_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure: Discovery vs Validation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Discovery LOO-CV ROC
    fpr_loo, tpr_loo, _ = roc_curve(loo_true, loo_pred)
    axes[0].plot(fpr_loo, tpr_loo, linewidth=2, label=f'AUC = {model_data["loo_auc"]:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Discovery LOO-CV')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation ROC
    y_true_val = val_predictions['true_label'].values
    y_pred_val = val_predictions['pred_proba'].values
    fpr_val, tpr_val, _ = roc_curve(y_true_val, y_pred_val)
    axes[1].plot(fpr_val, tpr_val, linewidth=2, label=f'AUC = {val_auc:.3f}')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Validation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'roc_curves.png'}")
    plt.close()
    
    # Performance comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Discovery\n(LOO-CV)', 'Validation'], 
                   [model_data['loo_auc'], val_auc],
                   color=['steelblue', 'coral'], alpha=0.7)
    ax.set_ylabel('AUC')
    ax.set_title(f'{model_data["model_type"].upper()}: Discovery vs Validation')
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_comparison.png'}")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(model_type='lasso'):
    """Run complete pipeline."""
    print("\n" + "="*70)
    print(f"MODULE 4: {model_type.upper()} CLASSIFIER")
    print("="*70)
    
    # Step 1: Load data
    discovery_df, validation_df = load_data()
    
    # Step 2: Extract features
    feature_df, all_features, meth_agg_df = extract_features(discovery_df)
    
    # Step 3: Select features
    if model_type == 'lasso':
        selected_features = select_features_lasso(feature_df, all_features)
    else:
        selected_features = select_features_rf(feature_df, all_features)
    
    # Step 4: Train with LOO-CV
    model_data, loo_pred, loo_true = train_with_loo(feature_df, selected_features, model_type)
    
    # Step 5: Validate
    val_predictions, val_auc, val_acc = validate(validation_df, meth_agg_df, model_data)
    
    # Step 6: Visualize
    visualize_results(model_data, loo_true, loo_pred, val_predictions, val_auc)
    
    # Save everything
    output_dir = RESULTS_DIR / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    val_predictions.to_csv(output_dir / 'validation_predictions.csv', index=False)
    
    # Summary
    summary = pd.DataFrame([{
        'model': model_type,
        'n_features': len(selected_features),
        'discovery_loo_auc': model_data['loo_auc'],
        'validation_auc': val_auc,
        'validation_accuracy': val_acc,
        'performance_drop': model_data['loo_auc'] - val_auc
    }])
    summary.to_csv(output_dir / 'summary.csv', index=False)
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Model: {model_type.upper()}")
    print(f"  Features: {len(selected_features)}")
    print(f"  Discovery LOO-CV AUC: {model_data['loo_auc']:.3f}")
    print(f"  Validation AUC: {val_auc:.3f}")
    print(f"  Drop: {model_data['loo_auc'] - val_auc:.3f}")
    
    return {
        'model_data': model_data,
        'val_predictions': val_predictions,
        'val_auc': val_auc
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', choices=['lasso', 'rf'],
                        help='Model type: lasso or rf')
    args = parser.parse_args()
    
    results = run_pipeline(model_type=args.model)