"""
Module 4: Classification (Production)

Purpose:
    Binary classification using the final validated approach.
    
    Approach (determined through exploration):
    - 23 combined features (17 fragmentomics + 6 methylation summaries)
    - XGBoost classifier
    - Train on discovery (n=8)
    - Validate on held-out validation set (n=14)
    
Input:
    - data/processed/all_features.csv from Module 2

Output:
    - results/classification/classification_metrics.csv
    - results/classification/validation_predictions.csv
    - results/classification/trained_xgb_model.pkl
    - results/figures/classification/roc_curve.png
    - results/figures/classification/confusion_matrix.png

Usage:
    As a script:
        python src/classification.py
    
    In a notebook:
        from src.classification import run_module_4
        results = run_module_4()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    ALL_FEATURES,
    DISCOVERY_BATCH,
    VALIDATION_BATCH,
    FINAL_FEATURES,
    FINAL_MODEL_PARAMS,
    CLASSIFICATION_DIR,
    CLASSIFICATION_METRICS_FILE,
    VALIDATION_PREDICTIONS_FILE,
    TRAINED_MODEL_FILE,
    ROC_CURVE_FILE,
    CONFUSION_MATRIX_FILE,
    FIGURES_DIR
)


# ============================================================================
# TRAIN CLASSIFIER
# ============================================================================

def train_classifier(df):
    """
    Train final XGBoost classifier with combined features.
    
    Parameters
    ----------
    df : pd.DataFrame
        All features dataframe
        
    Returns
    -------
    dict
        Trained model and metadata
    """
    print("\n" + "="*70)
    print("TRAINING CLASSIFIER")
    print("="*70)
    
    # Split data
    discovery_df = df[df['batch'] == DISCOVERY_BATCH].copy()
    validation_df = df[df['batch'] == VALIDATION_BATCH].copy()
    
    print(f"\nDiscovery set: {len(discovery_df)} samples "
          f"({(discovery_df['disease_status']=='als').sum()} ALS, "
          f"{(discovery_df['disease_status']=='ctrl').sum()} Control)")
    print(f"Validation set: {len(validation_df)} samples "
          f"({(validation_df['disease_status']=='als').sum()} ALS, "
          f"{(validation_df['disease_status']=='ctrl').sum()} Control)")
    
    # Prepare training data
    X_train = discovery_df[FINAL_FEATURES].fillna(discovery_df[FINAL_FEATURES].median()).values
    y_train = (discovery_df['disease_status'] == 'als').astype(int).values
    
    # Prepare validation data
    X_val = validation_df[FINAL_FEATURES].fillna(validation_df[FINAL_FEATURES].median()).values
    y_val = (validation_df['disease_status'] == 'als').astype(int).values
    
    print(f"\nFeatures used ({len(FINAL_FEATURES)}):")
    print(f"  Fragmentomics summary: 17 features")
    print(f"  Methylation summary: {len(FINAL_FEATURES) - 17} features")
    print(f"\nFirst 10 features:")
    for feat in FINAL_FEATURES[:10]:
        print(f"  - {feat}")
    if len(FINAL_FEATURES) > 10:
        print(f"  ... and {len(FINAL_FEATURES)-10} more")
    
    # Train XGBoost
    print(f"\nTraining XGBoost...")
    print(f"  Parameters: {FINAL_MODEL_PARAMS}")
    
    model = xgb.XGBClassifier(**FINAL_MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'auc': roc_auc_score(y_val, y_pred_proba),
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0)
    }
    
    print(f"\n{'='*70}")
    print("VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  AUC:        {metrics['auc']:.3f}")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1-Score:   {metrics['f1']:.3f}")
    
    # Create predictions dataframe
    predictions_df = validation_df[['sample_id', 'disease_status', 'age']].copy()
    predictions_df['true_label'] = y_val
    predictions_df['pred_proba'] = y_pred_proba
    predictions_df['pred_label'] = y_pred
    predictions_df['correct'] = (y_val == y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Ctrl    ALS")
    print(f"  Actual Ctrl   {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"         ALS    {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    return {
        'model': model,
        'metrics': metrics,
        'predictions': predictions_df,
        'confusion_matrix': cm,
        'y_val': y_val,
        'y_pred_proba': y_pred_proba,
        'feature_names': FINAL_FEATURES
    }


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results):
    """
    Save classification results and generate plots.
    
    Parameters
    ----------
    results : dict
        Results from train_classifier()
    """
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Create directories
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / 'classification').mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(CLASSIFICATION_METRICS_FILE, index=False)
    print(f"\n✓ Saved metrics: {CLASSIFICATION_METRICS_FILE}")
    
    # Save predictions
    results['predictions'].to_csv(VALIDATION_PREDICTIONS_FILE, index=False)
    print(f"✓ Saved predictions: {VALIDATION_PREDICTIONS_FILE}")
    
    # Save model
    with open(TRAINED_MODEL_FILE, 'wb') as f:
        pickle.dump(results['model'], f)
    print(f"✓ Saved model: {TRAINED_MODEL_FILE}")
    
    # Generate plots
    plot_roc_curve(results['y_val'], results['y_pred_proba'], results['metrics']['auc'])
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Save feature importances
    save_feature_importances(results['model'], results['feature_names'])
    
    print(f"\n✓ All results saved to: {CLASSIFICATION_DIR}")


def plot_roc_curve(y_true, y_pred_proba, auc):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Validation Set', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ROC_CURVE_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved ROC curve: {ROC_CURVE_FILE}")


def plot_confusion_matrix(cm):
    """Generate and save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Control', 'ALS'],
                yticklabels=['Control', 'ALS'])
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved confusion matrix: {CONFUSION_MATRIX_FILE}")


def save_feature_importances(model, feature_names):
    """Save feature importances to CSV."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    output_file = CLASSIFICATION_DIR / 'feature_importances.csv'
    importance_df.to_csv(output_file, index=False)
    print(f"✓ Saved feature importances: {output_file}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_module_4():
    """
    Run complete Module 4: Classification pipeline.
    
    Returns
    -------
    dict
        Classification results
    """
    print("\n" + "=" * 70)
    print("MODULE 4: Classification")
    print("=" * 70)
    
    # Load features
    print(f"\nLoading features from: {ALL_FEATURES}")
    
    if not ALL_FEATURES.exists():
        raise FileNotFoundError(
            f"Feature file not found: {ALL_FEATURES}\n"
            f"Please run Module 2 first."
        )
    
    df = pd.read_csv(ALL_FEATURES)
    print(f"✓ Loaded: {df.shape[0]} samples × {df.shape[1]} features")
    
    # Train classifier
    results = train_classifier(df)
    
    # Save results
    save_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("MODULE 4 COMPLETE")
    print("=" * 70)
    print(f"\nFinal Model: XGBoost with {len(FINAL_FEATURES)} features")
    print(f"  - Fragmentomics summary: 17 features")
    print(f"  - Methylation summary: 6 features")
    print(f"\nValidation Metrics:")
    print(f"  AUC:        {results['metrics']['auc']:.3f}")
    print(f"  Accuracy:   {results['metrics']['accuracy']:.3f}")
    print(f"  Precision:  {results['metrics']['precision']:.3f}")
    print(f"  Recall:     {results['metrics']['recall']:.3f}")
    print(f"  F1-Score:   {results['metrics']['f1']:.3f}")
    print("=" * 70 + "\n")
    
    return results


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run Module 4 as a standalone script
    results = run_module_4()
