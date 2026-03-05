"""
Phase 1 ML Pipeline: OES-based H2O2 yield rate prediction.

Usage:
    python -m phase1.main              # run everything
    python -m phase1.main --pca-only   # run only PCA analysis
    python -m phase1.main --eval-only  # run only model evaluation
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

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ML Pipeline")
    parser.add_argument("--pca-only", action="store_true", help="Run only PCA analysis")
    parser.add_argument("--eval-only", action="store_true", help="Run only model evaluation")
    parser.add_argument("--pca-k", type=int, default=None, help="Override PCA k (for --eval-only)")
    args = parser.parse_args()

    # Setup
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data
    print("Loading and preprocessing data...")
    data = prepare_data()
    print(f"  Samples: {len(data['target'])}")
    print(f"  OES features: {data['oes_raw'].shape[1]}")
    print(f"  Discharge features: {data['discharge_raw'].shape[1]}")
    print(f"  Target range: {data['target'].min():.2f} – {data['target'].max():.2f}")

    if args.eval_only:
        k = args.pca_k or 5
        print(f"\nSkipping PCA analysis, using k={k}")
    else:
        # PCA analysis
        print("\nRunning PCA analysis...")
        pca, k = run_pca_analysis(data)

    if args.pca_only:
        print("\nPCA analysis complete. Exiting (--pca-only).")
        return

    # Model evaluation
    print(f"\nRunning LOOCV with {k} PCA components...")
    all_results, results_df = run_all_evaluations(data, k)

    # Plots
    print("\nGenerating plots...")
    generate_all_plots(all_results, results_df)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
