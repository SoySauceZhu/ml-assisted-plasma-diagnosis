"""
Phase 1 Main Entry Point
=========================
Command-line interface for the baseline ML pipeline. Orchestrates the full
Phase 1 workflow: data loading -> PCA analysis -> LOOCV evaluation -> plotting.

Usage:
    python -m phase1.main              # run full pipeline
    python -m phase1.main --pca-only   # run only PCA analysis (no model evaluation)
    python -m phase1.main --eval-only  # skip PCA, run only model evaluation
    python -m phase1.main --eval-only --pca-k 11  # specify PCA k manually
"""
import argparse
import numpy as np
import torch
import warnings

from .config import RANDOM_SEED, RESULTS_DIR, FIGURES_DIR, TABLES_DIR
from .data_loader import prepare_data
from .pca_analysis import run_pca_analysis
from .evaluation import run_all_evaluations
from .plotting import generate_all_plots

# Suppress sklearn and torch UserWarnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """Main pipeline function for Phase 1.

    Workflow:
        1. Parse command-line arguments (--pca-only, --eval-only, --pca-k)
        2. Set random seeds for reproducibility (numpy seed=42, torch seed=42)
        3. Load and preprocess data via prepare_data()
        4. Run PCA analysis to determine optimal k (unless --eval-only)
        5. Run LOOCV for all 7 models x 3 configs (unless --pca-only)
        6. Generate all result plots
        7. Print summary table to stdout
    """
    parser = argparse.ArgumentParser(description="Phase 1 ML Pipeline")
    parser.add_argument("--pca-only", action="store_true", help="Run only PCA analysis")
    parser.add_argument("--eval-only", action="store_true", help="Run only model evaluation")
    parser.add_argument("--pca-k", type=int, default=None, help="Override PCA k (for --eval-only)")
    args = parser.parse_args()

    # Create output directories
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Set random seeds for reproducibility across numpy, torch, and sklearn
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Step 1: Load and preprocess the dataset
    print("Loading and preprocessing data...")
    data = prepare_data()
    print(f"  Samples: {len(data['target'])}")
    print(f"  OES features: {data['oes_raw'].shape[1]}")
    print(f"  Discharge features: {data['discharge_raw'].shape[1]}")
    print(f"  Target range: {data['target'].min():.2f} – {data['target'].max():.2f}")

    # Step 2: PCA analysis (determines how many components to keep)
    if args.eval_only:
        k = args.pca_k or 5
        print(f"\nSkipping PCA analysis, using k={k}")
    else:
        print("\nRunning PCA analysis...")
        _, k = run_pca_analysis(data)

    if args.pca_only:
        print("\nPCA analysis complete. Exiting (--pca-only).")
        return

    # Step 3: Run LOOCV for all model x config combinations
    print(f"\nRunning LOOCV with {k} PCA components...")
    all_results, results_df = run_all_evaluations(data, k)

    # Step 4: Generate all visualisations
    print("\nGenerating plots...")
    generate_all_plots(all_results, results_df)

    # Step 5: Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
