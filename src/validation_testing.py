"""
Module 5: Validation Testing

Purpose:
    Test the trained LASSO model on the held-out validation set.
    This is the ultimate test of whether the discovery model generalizes.

Strategy:
    - Load trained model from Module 4
    - Load validation set features
    - Aggregate validation methylation to match training (500kb bins)
    - Make predictions on validation set
    - Calculate and report performance metrics
    - Compare to discovery LOO-CV performance

Critical Questions:
    1. Does validation AUC match LOO-CV AUC (~1.0)?
       → If yes: Signal is real and generalizable!
       → If no: Model overfit to discovery batch
    
    2. How do validation predictions look?
       → Perfect separation or mixed?
       → Any patterns by disease severity (ALSFRS)?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, 
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Import configuration
from src.config import (
    ALL_FEATURES,
    VALIDATION_BATCH,
    TRAINED_LASSO_MODEL,
    FEATURE_SELECTION_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    METHYLATION_AGGREGATION_SIZE
)

# ============================================================================
# STEP 1: LOAD TRAINED MODEL
# ============================================================================

def load_trained_model():
    """
    Load the trained LASSO model from Module 4.
    
    Returns
    -------
    dict
        Model data including model, scaler, feature columns, etc.
    """
    print("\n" + "="*70)
    print("STEP 1: Loading trained model")
    print("="*70)
    
    with open(TRAINED_LASSO_MODEL, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n✓ Loaded trained model from: {TRAINED_LASSO_MODEL}")
    print(f"  Training LOO-CV AUC: {model_data['loo_auc']:.3f}")
    print(f"  Regularization (C): {model_data['best_C']}")
    print(f"  Features selected: {len(model_data['selected_features'])}")
    print(f"\nSelected features:")
    for feat in model_data['selected_features']:
        print(f"  - {feat}")
    
    return model_data


# ============================================================================
# STEP 2: PREPARE VALIDATION DATA
# ============================================================================

def prepare_validation_features(all_features_path, model_feature_cols):
    """
    Load validation set and prepare features to match training data.
    
    This includes:
    1. Loading validation samples
    2. Extracting same fragmentomics features as training
    3. Aggregating methylation to 500kb bins (same as training)
    
    Parameters
    ----------
    all_features_path : Path
        Path to all_features.csv
    model_feature_cols : list
        Feature columns used in training
        
    Returns
    -------
    tuple
        (validation_df, X_validation)
    """
    print("\n" + "="*70)
    print("STEP 2: Preparing validation data")
    print("="*70)
    
    # Load all features
    df = pd.read_csv(all_features_path)
    validation_df = df[df['batch'] == VALIDATION_BATCH].copy()
    
    print(f"\nValidation set: {len(validation_df)} samples")
    print(f"  ALS: {(validation_df['disease_status']=='als').sum()}")
    print(f"  Control: {(validation_df['disease_status']=='ctrl').sum()}")
    
    # Extract features that match training
    # Separate fragmentomics and methylation
    frag_features = [f for f in model_feature_cols if not f.startswith('meth_agg_')]
    meth_features = [f for f in model_feature_cols if f.startswith('meth_agg_')]
    
    print(f"\nFeatures needed:")
    print(f"  Fragmentomics: {len(frag_features)}")
    print(f"  Methylation (500kb aggregated): {len(meth_features)}")
    
    # Fragmentomics features should already exist
    validation_features_df = validation_df[['sample_id', 'disease_status', 'batch', 'age']].copy()
    
    for feat in frag_features:
        if feat in validation_df.columns:
            validation_features_df[feat] = validation_df[feat]
        else:
            print(f"  WARNING: Missing fragmentomics feature: {feat}")
            validation_features_df[feat] = np.nan
    
    # Aggregate methylation bins for validation
    print(f"\nAggregating validation methylation to 500kb bins...")
    meth_cols_100kb = [c for c in validation_df.columns if c.startswith('regional_meth_bin_')]
    
    # Filter to numeric bins only
    meth_cols_100kb = []
    for c in validation_df.columns:
        if c.startswith('regional_meth_bin_'):
            suffix = c.replace('regional_meth_bin_', '')
            if suffix.isdigit():
                meth_cols_100kb.append(c)
    
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
    
    # Add aggregated methylation features
    for feat in meth_features:
        if feat in aggregated_features:
            feature_values = pd.Series(aggregated_features[feat])
            validation_features_df[feat] = validation_features_df.index.map(
                lambda idx: feature_values.get(idx, np.nan)
            )
        else:
            print(f"  WARNING: Missing methylation feature: {feat}")
            validation_features_df[feat] = np.nan
    
    # Extract feature matrix
    X_validation = validation_features_df[model_feature_cols].fillna(
        validation_features_df[model_feature_cols].median()
    ).values
    
    print(f"\n✓ Validation feature matrix: {X_validation.shape}")
    
    # Check for missing features
    missing_count = validation_features_df[model_feature_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"  ⚠ Warning: {missing_count} missing values (filled with median)")
    
    return validation_features_df, X_validation


# ============================================================================
# STEP 3: MAKE PREDICTIONS
# ============================================================================

def predict_validation(model_data, validation_df, X_validation):
    """
    Use trained model to predict on validation set.
    
    Parameters
    ----------
    model_data : dict
        Trained model and scaler
    validation_df : pd.DataFrame
        Validation metadata
    X_validation : np.ndarray
        Validation feature matrix
        
    Returns
    -------
    pd.DataFrame
        Predictions with probabilities and binary predictions
    """
    print("\n" + "="*70)
    print("STEP 3: Making predictions on validation set")
    print("="*70)
    
    # Scale features using training scaler
    X_val_scaled = model_data['scaler'].transform(X_validation)
    
    # Predict probabilities
    y_pred_proba = model_data['model'].predict_proba(X_val_scaled)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    
    # Create results dataframe
    predictions_df = validation_df[['sample_id', 'disease_status', 'age']].copy()
    predictions_df['true_label'] = (validation_df['disease_status'] == 'als').astype(int)
    predictions_df['pred_proba'] = y_pred_proba
    predictions_df['pred_label'] = y_pred_binary
    predictions_df['pred_class'] = predictions_df['pred_label'].map({0: 'ctrl', 1: 'als'})
    predictions_df['correct'] = (predictions_df['true_label'] == predictions_df['pred_label'])
    
    print(f"\n✓ Generated predictions for {len(predictions_df)} validation samples")
    print(f"\nPrediction distribution:")
    print(f"  Predicted ALS: {y_pred_binary.sum()}")
    print(f"  Predicted Control: {len(y_pred_binary) - y_pred_binary.sum()}")
    
    return predictions_df


# ============================================================================
# STEP 4: EVALUATE PERFORMANCE
# ============================================================================

def evaluate_performance(predictions_df, model_data):
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with true and predicted labels
    model_data : dict
        Model data including LOO-CV performance
        
    Returns
    -------
    dict
        Performance metrics
    """
    print("\n" + "="*70)
    print("STEP 4: Evaluating performance")
    print("="*70)
    
    y_true = predictions_df['true_label'].values
    y_pred_proba = predictions_df['pred_proba'].values
    y_pred = predictions_df['pred_label'].values
    
    # Calculate metrics
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc = np.nan
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle cases where precision/recall might fail
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except:
        precision = recall = f1 = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'validation_auc': auc,
        'validation_accuracy': accuracy,
        'validation_precision': precision,
        'validation_recall': recall,
        'validation_f1': f1,
        'confusion_matrix': cm,
        'discovery_loo_auc': model_data['loo_auc']
    }
    
    # Print results
    print("\n" + "="*70)
    print("VALIDATION PERFORMANCE")
    print("="*70)
    
    print(f"\nMetrics:")
    print(f"  AUC:       {auc:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"               Ctrl    ALS")
    print(f"  Actual Ctrl   {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"         ALS    {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # Compare to discovery
    print(f"\n" + "="*70)
    print("DISCOVERY vs VALIDATION COMPARISON")
    print("="*70)
    print(f"\n  Discovery LOO-CV AUC:  {model_data['loo_auc']:.3f}")
    print(f"  Validation AUC:        {auc:.3f}")
    print(f"  Difference:            {auc - model_data['loo_auc']:+.3f}")
    
    # Interpretation
    print(f"\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)
    
    if auc >= 0.9:
        print("  ✓ EXCELLENT: Model generalizes extremely well!")
        print("    The methylation signal appears to be real and robust.")
    elif auc >= 0.75:
        print("  ✓ GOOD: Model shows decent generalization.")
        print("    Some signal is real, though not as strong as in discovery.")
    elif auc >= 0.60:
        print("  ⚠ MODERATE: Model shows weak generalization.")
        print("    Signal exists but is likely confounded by batch effects.")
    else:
        print("  ✗ POOR: Model does not generalize.")
        print("    Discovery performance was likely due to overfitting/batch effects.")
    
    if abs(auc - model_data['loo_auc']) > 0.2:
        print("\n  ⚠ Large drop from discovery suggests:")
        print("    - Batch effects between discovery and validation")
        print("    - Overfitting in discovery despite regularization")
        print("    - Different disease characteristics in validation")
    
    return metrics


# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

def plot_validation_results(predictions_df, metrics):
    """
    Create comprehensive visualization of validation results.
    """
    print("\n" + "="*70)
    print("STEP 5: Creating visualizations")
    print("="*70)
    
    output_dir = FIGURES_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC curve
    y_true = predictions_df['true_label'].values
    y_pred_proba = predictions_df['pred_proba'].values
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUC = {metrics["validation_auc"]:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('Validation ROC Curve', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prediction distribution
    als_probs = predictions_df[predictions_df['disease_status']=='als']['pred_proba']
    ctrl_probs = predictions_df[predictions_df['disease_status']=='ctrl']['pred_proba']
    
    axes[1].hist(ctrl_probs, bins=10, alpha=0.6, label='Control', color='blue')
    axes[1].hist(als_probs, bins=10, alpha=0.6, label='ALS', color='red')
    axes[1].axvline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
    axes[1].set_xlabel('Predicted Probability (ALS)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Prediction Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_performance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'validation_performance.png'}")
    plt.close()
    
    # Figure 2: Sample-level predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    predictions_sorted = predictions_df.sort_values('pred_proba')
    colors = ['red' if status == 'als' else 'blue' 
              for status in predictions_sorted['disease_status']]
    
    ax.barh(range(len(predictions_sorted)), predictions_sorted['pred_proba'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(predictions_sorted)))
    ax.set_yticklabels(predictions_sorted['sample_id'], fontsize=8)
    ax.set_xlabel('Predicted Probability (ALS)', fontsize=12)
    ax.set_title('Sample-Level Predictions (Validation Set)', fontsize=13, fontweight='bold')
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='ALS (true)'),
        Patch(facecolor='blue', alpha=0.7, label='Control (true)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'sample_predictions.png'}")
    plt.close()
    
    # Figure 3: Discovery vs Validation comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    comparison_data = {
        'Set': ['Discovery\n(LOO-CV)', 'Validation'],
        'AUC': [metrics['discovery_loo_auc'], metrics['validation_auc']]
    }
    
    bars = ax.bar(comparison_data['Set'], comparison_data['AUC'], 
                   color=['steelblue', 'coral'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Model Performance: Discovery vs Validation', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random')
    ax.axhline(0.7, color='orange', linestyle='--', linewidth=1, label='Acceptable')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'discovery_vs_validation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'discovery_vs_validation.png'}")
    plt.close()


def save_results(predictions_df, metrics):
    """
    Save results to CSV files.
    """
    output_dir = RESULTS_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_df.to_csv(output_dir / 'validation_predictions.csv', index=False)
    print(f"\n✓ Saved predictions: {output_dir / 'validation_predictions.csv'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([{
        'metric': k,
        'value': v
    } for k, v in metrics.items() if k != 'confusion_matrix'])
    metrics_df.to_csv(output_dir / 'validation_metrics.csv', index=False)
    print(f"✓ Saved metrics: {output_dir / 'validation_metrics.csv'}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_module_5():
    """
    Run complete validation testing pipeline.
    
    Returns
    -------
    dict
        All results including predictions and metrics
    """
    print("\n" + "="*70)
    print("MODULE 5: VALIDATION TESTING")
    print("="*70)
    print("\nThis is the moment of truth!")
    print("Will the discovery model generalize to validation?")
    
    # Step 1: Load model
    model_data = load_trained_model()
    
    # Step 2: Prepare validation data
    validation_df, X_validation = prepare_validation_features(
        ALL_FEATURES, 
        model_data['feature_cols']
    )
    
    # Step 3: Make predictions
    predictions_df = predict_validation(model_data, validation_df, X_validation)
    
    # Step 4: Evaluate
    metrics = evaluate_performance(predictions_df, model_data)
    
    # Step 5: Visualize
    plot_validation_results(predictions_df, metrics)
    
    # Save results
    save_results(predictions_df, metrics)
    
    print("\n" + "="*70)
    print("MODULE 5 COMPLETE")
    print("="*70)
    
    return {
        'model_data': model_data,
        'validation_df': validation_df,
        'predictions': predictions_df,
        'metrics': metrics
    }


if __name__ == "__main__":
    results = run_module_5()



