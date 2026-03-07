"""
Phase 3: Domain-Knowledge OES Feature Engineering Pipeline.

Usage:
    python -m phase3.main                          # Full pipeline (initial eval + tuning + final eval)
    python -m phase3.main --initial-only           # Only run with default params (no tuning)
    python -m phase3.main --tune-only              # Only run tuning (no final eval)
    python -m phase3.main --eval-only              # Only run eval with saved tuned params
    python -m phase3.main --models RF MLP          # Tune only specific models
"""
import argparse
import json
import numpy as np
import torch
import warnings

from phase1.data_loader import prepare_data
from .config import (
    RANDOM_SEED, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODEL_CONFIGS, MODEL_NAMES,
)
from .feature_engineer import extract_oes_features
from .tuner_rf import tune_rf
from .tuner_mlp import tune_mlp
from .evaluation import run_all_evaluations, build_comparison_table
from .plotting import generate_all_phase3_plots

warnings.filterwarnings("ignore", category=UserWarning)


def _serialize_params(tuned_params):
    """Convert tuned_params dict to JSON-serializable format."""
    out = {}
    for (model, config), params in tuned_params.items():
        key = f"{model}_{config}"
        out[key] = {k: v for k, v in params.items()}
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
    parser = argparse.ArgumentParser(description="Phase 3: Feature Engineering Pipeline")
    parser.add_argument("--initial-only", action="store_true",
                        help="Only run evaluation with default params (no tuning)")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run tuning, skip final evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation with saved tuned params")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES,
                        choices=MODEL_NAMES,
                        help=f"Models to process (default: all)")
    args = parser.parse_args()

    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load data
    print("Loading data...")
    data = prepare_data()
    print(f"  Samples: {len(data['target'])}")
    print(f"  OES raw features: {data['oes_raw'].shape[1]}")

    # Extract engineered features
    print("\nExtracting 13 domain-knowledge OES features...")
    oes_features, feature_names = extract_oes_features(
        data["oes_raw"], data["wavelengths"]
    )
    print(f"  Engineered features shape: {oes_features.shape}")
    print(f"  Features: {feature_names}")

    # Initial evaluation with default params
    if args.initial_only:
        print(f"\nRunning LOOCV with default parameters...")
        all_results, results_df = run_all_evaluations(oes_features, data)

        comparison_df = build_comparison_table(results_df)

        print("\nGenerating plots...")
        generate_all_phase3_plots(all_results, results_df, {}, comparison_df)

        print("\n" + "=" * 60)
        print("PHASE 3 RESULTS (Default Params)")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print(f"\nResults saved to {RESULTS_DIR}")
        return

    tuned_params = {}
    studies = {}
    params_path = TABLES_DIR / "tuned_hyperparameters.json"

    if not args.eval_only:
        tuner_map = {"RF": tune_rf, "MLP": tune_mlp}
        tunable_models = [m for m in args.models if m in tuner_map]

        for model_name in tunable_models:
            configs = MODEL_CONFIGS[model_name]
            for config_name in configs:
                print(f"\n{'=' * 50}")
                print(f"Tuning {model_name} Config {config_name}...")
                print(f"{'=' * 50}")

                tune_fn = tuner_map[model_name]
                best_params, study = tune_fn(oes_features, data, config_name)

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

    # Final evaluation with tuned params
    print(f"\nRunning LOOCV with tuned parameters...")
    all_results, results_df = run_all_evaluations(
        oes_features, data, tuned_params=tuned_params
    )

    comparison_df = build_comparison_table(results_df)

    print("\nGenerating plots...")
    generate_all_phase3_plots(all_results, results_df, studies, comparison_df)

    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nPhase Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
