import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES_DIR


def plot_optimization_history(study, model_name, config_name, save_dir=None):
    """Plot Optuna optimization history (best value vs trial number)."""
    save_dir = save_dir or FIGURES_DIR
    trials = study.trials
    trial_numbers = [t.number for t in trials if t.value is not None]
    values = [t.value for t in trials if t.value is not None]

    best_values = []
    current_best = -np.inf
    for v in values:
        current_best = max(current_best, v)
        best_values.append(current_best)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trial_numbers, values, alpha=0.4, s=15, label="Trial R2")
    ax.plot(trial_numbers, best_values, color="red", linewidth=2, label="Best R2")
    ax.set_xlabel("Trial")
    ax.set_ylabel("R2")
    ax.set_title(f"Optimization History: {model_name} Config {config_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"optuna_history_{model_name}_{config_name}.png", dpi=150)
    plt.close(fig)


def plot_param_importances(study, model_name, config_name, save_dir=None):
    """Plot hyperparameter importance from Optuna study."""
    save_dir = save_dir or FIGURES_DIR
    try:
        importances = _compute_param_importances(study)
    except Exception:
        print(f"    Skipping param importance plot for {model_name} Config {config_name}")
        return

    names = list(importances.keys())
    values = list(importances.values())
    sorted_idx = np.argsort(values)
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values)
    ax.set_xlabel("Importance")
    ax.set_title(f"Param Importance: {model_name} Config {config_name}")
    fig.tight_layout()
    fig.savefig(save_dir / f"optuna_importance_{model_name}_{config_name}.png", dpi=150)
    plt.close(fig)


def _compute_param_importances(study):
    """Compute parameter importances using Optuna's fANOVA."""
    from optuna.importance import get_param_importances
    return get_param_importances(study)


def plot_comparison_bar(comparison_df, save_path=None):
    """Side-by-side bar chart: Phase 1 vs Phase 2 R2 for each model x config."""
    save_path = save_path or FIGURES_DIR / "phase1_vs_phase2_comparison.png"

    df = comparison_df.dropna(subset=["R2_P1", "R2_P2"])
    labels = [f"{row['Model']}\nConfig {row['Config']}" for _, row in df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bars1 = ax.bar(x - width / 2, df["R2_P1"].values, width, label="Phase 1", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, df["R2_P2"].values, width, label="Phase 2 (Tuned)", color="#DD8452")

    ax.set_ylabel("R2")
    ax.set_title("Phase 1 vs Phase 2: R2 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_phase2_predicted_vs_actual(all_results, save_dir=None):
    """Predicted vs actual scatter plots for tuned models."""
    save_dir = save_dir or FIGURES_DIR
    for r in all_results:
        fig, ax = plt.subplots(figsize=(5, 5))
        y_true = r["y_true"]
        y_pred = r["y_pred"]

        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
        mn = min(y_true.min(), y_pred.min()) - 0.05
        mx = max(y_true.max(), y_pred.max()) + 0.05
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Ideal")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel("Actual H2O2 Rate")
        ax.set_ylabel("Predicted H2O2 Rate")
        ax.set_title(f"{r['model']} Config {r['config']} (Tuned)\n"
                      f"R2={r['R2']:.3f}  RMSE={r['RMSE']:.3f}")
        ax.legend()
        ax.set_aspect("equal")
        fig.tight_layout()
        fname = f"predicted_vs_actual_{r['model']}_{r['config']}_tuned.png"
        fig.savefig(save_dir / fname, dpi=150)
        plt.close(fig)


def generate_all_phase2_plots(all_results, results_df, studies_dict, comparison_df):
    """Master plotting function for Phase 2."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for (model_name, config_name), study in studies_dict.items():
        plot_optimization_history(study, model_name, config_name)
        plot_param_importances(study, model_name, config_name)

    plot_phase2_predicted_vs_actual(all_results)
    plot_comparison_bar(comparison_df)

    print(f"  Plots saved to {FIGURES_DIR}")
