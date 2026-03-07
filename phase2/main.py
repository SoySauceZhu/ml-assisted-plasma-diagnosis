"""
Phase 2: Hyperparameter Tuning for RF, CNN, MLP.

Usage:
    python -m phase2.main                  # Run everything
    python -m phase2.main --tune-only      # Only run tuning (no final eval)
    python -m phase2.main --eval-only      # Only run eval with saved params
    python -m phase2.main --models RF MLP  # Tune only specific models
"""
import argparse
import json
import numpy as np
import torch
import warnings

from phase1.data_loader import prepare_data
from .config import (
    RANDOM_SEED, PCA_K, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODEL_CONFIGS
)
from .tuner_rf import tune_rf
from .tuner_mlp import tune_mlp
from .tuner_cnn import tune_cnn
from .evaluation import run_all_tuned_evaluations, build_comparison_table
from .plotting import generate_all_phase2_plots

warnings.filterwarnings("ignore", category=UserWarning)


def _serialize_params(tuned_params):
    """Convert tuned_params dict to JSON-serializable format."""
    out = {}
    for (model, config), params in tuned_params.items():
        key = f"{model}_{config}"
        serializable = {}
        for k, v in params.items():
            if isinstance(v, list):
                serializable[k] = v
            elif v is None:
                serializable[k] = None
            else:
                serializable[k] = v
        out[key] = serializable
    return out


def _deserialize_params(raw):
    """Convert JSON-loaded dict back to tuned_params format."""
    tuned = {}
    for key, params in raw.items():
        parts = key.split("_", 1)
        model, config = parts[0], parts[1]
        tuned[(model, config)] = params
    return tuned


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Hyperparameter Tuning")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run tuning, skip final evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation with saved tuned params")
    parser.add_argument("--models", nargs="+", default=["RF", "MLP", "CNN"],
                        choices=["RF", "MLP", "CNN"],
                        help="Models to tune (default: all three)")
    parser.add_argument("--pca-k", type=int, default=PCA_K,
                        help=f"Number of PCA components (default: {PCA_K})")
    args = parser.parse_args()

    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Loading data...")
    data = prepare_data()
    print(f"  Samples: {len(data['target'])}, PCA k={args.pca_k}")

    tuned_params = {}
    studies = {}
    params_path = TABLES_DIR / "tuned_hyperparameters.json"

    if not args.eval_only:
        tuner_map = {"RF": tune_rf, "MLP": tune_mlp, "CNN": tune_cnn}

        for model_name in args.models:
            configs = MODEL_CONFIGS[model_name]
            for config_name in configs:
                print(f"\n{'=' * 50}")
                print(f"Tuning {model_name} Config {config_name}...")
                print(f"{'=' * 50}")

                tune_fn = tuner_map[model_name]
                best_params, study = tune_fn(data, config_name, pca_k=args.pca_k)

                tuned_params[(model_name, config_name)] = best_params
                studies[(model_name, config_name)] = study

        with open(params_path, "w") as f:
            json.dump(_serialize_params(tuned_params), f, indent=2)
        print(f"\nTuned params saved to {params_path}")

    if args.tune_only:
        print("\nTuning complete. Exiting (--tune-only).")
        return

    if args.eval_only:
        print(f"\nLoading tuned params from {params_path}...")
        with open(params_path) as f:
            raw = json.load(f)
        tuned_params = _deserialize_params(raw)

    print(f"\nRunning outer LOOCV with tuned parameters...")
    all_results, results_df = run_all_tuned_evaluations(
        data, tuned_params, pca_k=args.pca_k
    )

    comparison_df = build_comparison_table(results_df)

    print("\nGenerating plots...")
    generate_all_phase2_plots(all_results, results_df, studies, comparison_df)

    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nPhase 1 vs Phase 2 Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
