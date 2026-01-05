"""
Run Complete wgbs_classifier Pipeline

Purpose:
    Execute all modules in sequence from data loading through classification.
    
Pipeline Steps:
    Module 0: Data Loading & Validation
    Module 1: Quality Control & Filtering
    Module 2: Feature Extraction (fragmentomics + methylation)
    Module 3: Required Visualizations & Batch Analysis
    Module 4: Classification (XGBoost)

Usage:
    python scripts/run_pipeline.py
    
    Optional arguments:
        --skip-qc           Skip Module 1 (QC)
        --skip-features     Skip Module 2 (Feature Extraction)
        --skip-viz          Skip Module 3 (Visualizations)
        --skip-class        Skip Module 4 (Classification)
        --modules           Run specific modules only (e.g., --modules 0,1,2)

Examples:
    # Run complete pipeline
    python scripts/run_pipeline.py
    
    # Run only classification (assumes previous modules complete)
    python scripts/run_pipeline.py --modules 4
    
    # Run modules 2-4 (skip data loading and QC)
    python scripts/run_pipeline.py --modules 2,3,4
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all modules
from src.data_loader import run_module_0
from src.qc import run_module_1
from src.feature_extraction import run_module_2
from src.visualization import run_module_3
from src.classification import run_module_4


def print_header(module_name, module_num):
    """Print formatted module header."""
    print("\n" + "=" * 80)
    print(f"MODULE {module_num}: {module_name}".center(80))
    print("=" * 80 + "\n")


def print_pipeline_header():
    """Print pipeline start header."""
    print("\n" + "=" * 80)
    print("WGBS CLASSIFIER PIPELINE".center(80))
    print("=" * 80)
    print("\nALS vs Control Classification using cfDNA WGBS")
    print("Pipeline: Data Loading → QC → Features → Visualization → Classification")
    print("\n" + "=" * 80 + "\n")


def print_pipeline_summary(timings):
    """Print final pipeline summary."""
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE".center(80))
    print("=" * 80 + "\n")
    
    total_time = sum(timings.values())
    
    print("Module Execution Times:")
    print("-" * 80)
    for module, elapsed in timings.items():
        pct = (elapsed / total_time) * 100 if total_time > 0 else 0
        print(f"  {module:<40s} {elapsed:>8.1f}s ({pct:>5.1f}%)")
    print("-" * 80)
    print(f"  {'TOTAL':<40s} {total_time:>8.1f}s (100.0%)")
    
    print("\n" + "=" * 80)
    print("\nOutput Files:")
    print("-" * 80)
    print("  data/processed/sample_manifest.csv")
    print("  data/processed/qc_metrics.csv")
    print("  data/processed/all_features.csv")
    print("  results/classification/classification_metrics.csv")
    print("  results/classification/validation_predictions.csv")
    print("  results/classification/trained_xgb_model.pkl")
    print("  results/figures/qc/")
    print("  results/figures/required_plots/")
    print("  results/figures/classification/")
    
    print("\n" + "=" * 80 + "\n")


def run_pipeline(modules_to_run=None):
    """
    Run the complete wgbs_classifier pipeline.
    
    Parameters
    ----------
    modules_to_run : list of int, optional
        Specific modules to run (0-4). If None, runs all modules.
    """
    # Default: run all modules
    if modules_to_run is None:
        modules_to_run = [0, 1, 2, 3, 4]
    
    print_pipeline_header()
    
    timings = {}
    results = {}
    
    # ========================================================================
    # MODULE 0: Data Loading
    # ========================================================================
    if 0 in modules_to_run:
        print_header("Data Loading & Validation", 0)
        start = time.time()
        try:
            manifest = run_module_0()
            results['module_0'] = manifest
            timings['Module 0: Data Loading'] = time.time() - start
            print(f"\n✓ Module 0 completed in {timings['Module 0: Data Loading']:.1f}s")
        except Exception as e:
            print(f"\n✗ Module 0 FAILED: {e}")
            return
    
    # ========================================================================
    # MODULE 1: Quality Control
    # ========================================================================
    if 1 in modules_to_run:
        print_header("Quality Control & Filtering", 1)
        start = time.time()
        try:
            qc_metrics, batch_effects = run_module_1()
            results['module_1'] = {'qc_metrics': qc_metrics, 'batch_effects': batch_effects}
            timings['Module 1: Quality Control'] = time.time() - start
            print(f"\n✓ Module 1 completed in {timings['Module 1: Quality Control']:.1f}s")
        except Exception as e:
            print(f"\n✗ Module 1 FAILED: {e}")
            return
    
    # ========================================================================
    # MODULE 2: Feature Extraction
    # ========================================================================
    if 2 in modules_to_run:
        print_header("Feature Extraction", 2)
        start = time.time()
        try:
            fragmentomics_df, methylation_df, all_features_df = run_module_2()
            results['module_2'] = {
                'fragmentomics': fragmentomics_df,
                'methylation': methylation_df,
                'all_features': all_features_df
            }
            timings['Module 2: Feature Extraction'] = time.time() - start
            print(f"\n✓ Module 2 completed in {timings['Module 2: Feature Extraction']:.1f}s")
        except Exception as e:
            print(f"\n✗ Module 2 FAILED: {e}")
            return
    
    # ========================================================================
    # MODULE 3: Visualization & Batch Analysis
    # ========================================================================
    if 3 in modules_to_run:
        print_header("Required Visualizations & Batch Analysis", 3)
        start = time.time()
        try:
            run_module_3()
            results['module_3'] = 'completed'
            timings['Module 3: Visualization'] = time.time() - start
            print(f"\n✓ Module 3 completed in {timings['Module 3: Visualization']:.1f}s")
        except Exception as e:
            print(f"\n✗ Module 3 FAILED: {e}")
            return
    
    # ========================================================================
    # MODULE 4: Classification
    # ========================================================================
    if 4 in modules_to_run:
        print_header("Classification (XGBoost)", 4)
        start = time.time()
        try:
            classification_results = run_module_4()
            results['module_4'] = classification_results
            timings['Module 4: Classification'] = time.time() - start
            print(f"\n✓ Module 4 completed in {timings['Module 4: Classification']:.1f}s")
        except Exception as e:
            print(f"\n✗ Module 4 FAILED: {e}")
            return
    
    # ========================================================================
    # Print Summary
    # ========================================================================
    print_pipeline_summary(timings)
    
    return results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run wgbs_classifier pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python scripts/run_pipeline.py
  
  # Run only classification (assumes previous steps complete)
  python scripts/run_pipeline.py --modules 4
  
  # Run modules 2-4 (skip data loading and QC)
  python scripts/run_pipeline.py --modules 2,3,4
  
  # Skip specific modules
  python scripts/run_pipeline.py --skip-viz
        """
    )
    
    parser.add_argument(
        '--modules',
        type=str,
        help='Comma-separated list of modules to run (e.g., "0,1,2" or "4")'
    )
    parser.add_argument(
        '--skip-qc',
        action='store_true',
        help='Skip Module 1 (Quality Control)'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip Module 2 (Feature Extraction)'
    )
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='Skip Module 3 (Visualizations)'
    )
    parser.add_argument(
        '--skip-class',
        action='store_true',
        help='Skip Module 4 (Classification)'
    )
    
    args = parser.parse_args()
    
    # Determine which modules to run
    if args.modules:
        # Parse comma-separated module numbers
        modules_to_run = [int(m.strip()) for m in args.modules.split(',')]
    else:
        # Start with all modules
        modules_to_run = [0, 1, 2, 3, 4]
        
        # Remove skipped modules
        if args.skip_qc:
            modules_to_run.remove(1)
        if args.skip_features:
            modules_to_run.remove(2)
        if args.skip_viz:
            modules_to_run.remove(3)
        if args.skip_class:
            modules_to_run.remove(4)
    
    # Validate module numbers
    for m in modules_to_run:
        if m not in [0, 1, 2, 3, 4]:
            print(f"Error: Invalid module number '{m}'. Must be 0-4.")
            return
    
    # Sort modules to ensure proper execution order
    modules_to_run = sorted(modules_to_run)
    
    # Run pipeline
    print(f"\nRunning modules: {modules_to_run}")
    run_pipeline(modules_to_run=modules_to_run)

# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    main()
