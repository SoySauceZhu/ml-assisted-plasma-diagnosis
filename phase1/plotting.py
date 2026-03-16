"""
Plotting Module
================
Generates all Phase 1 result visualisations:
  - Individual predicted-vs-actual scatter plots per model-config
  - Grid view of all scatter plots (7 models x 3 configs)
  - Heatmaps of R2/RMSE/MAE (models on rows, configs on columns)
  - Grouped bar chart comparing R2 across models and configs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .config import FIGURES_DIR, TABLES_DIR


def plot_predicted_vs_actual(result, save_path=None):
    """Plot predicted vs actual H2O2 yield for one model-config combination.

    Creates a scatter plot with a red dashed 1:1 reference line (ideal predictions
    fall on this line). Displays R2 and RMSE in the title. Equal aspect ratio
    ensures visual accuracy.

    Args:
        result: Dict from run_loocv_for_model() with keys "y_true", "y_pred",
                "model", "config", "R2", "RMSE".
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(result["y_true"], result["y_pred"], s=50, edgecolors="k", zorder=3)
    lims = [
        min(result["y_true"].min(), result["y_pred"].min()) - 0.05,
        max(result["y_true"].max(), result["y_pred"].max()) + 0.05,
    ]
    ax.plot(lims, lims, "r--", lw=1, label="Ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual H2O2 rate")
    ax.set_ylabel("Predicted H2O2 rate")
    ax.set_title(f"{result['model']} Config {result['config']}\n"
                 f"R2={result['R2']:.3f}  RMSE={result['RMSE']:.3f}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_all_predicted_vs_actual(all_results, save_dir=None):
    """Grid of predicted-vs-actual scatter plots for all model-config combinations.

    Creates a (n_models x n_configs) subplot grid. Each cell shows one scatter
    plot. Missing combinations (e.g., CNN Config B) are marked "N/A".

    Args:
        all_results: List of result dicts from run_all_evaluations().
        save_dir: Directory to save the grid plot (as predicted_vs_actual_grid.png).
    """
    models = list(dict.fromkeys(r["model"] for r in all_results))
    configs = list(dict.fromkeys(r["config"] for r in all_results))
    result_map = {(r["model"], r["config"]): r for r in all_results}

    n_rows = len(models)
    n_cols = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, model in enumerate(models):
        for j, config in enumerate(configs):
            ax = axes[i, j]
            key = (model, config)
            if key not in result_map:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14)
                ax.set_title(f"{model} Config {config}")
                continue
            r = result_map[key]
            ax.scatter(r["y_true"], r["y_pred"], s=30, edgecolors="k", zorder=3)
            lims = [
                min(r["y_true"].min(), r["y_pred"].min()) - 0.05,
                max(r["y_true"].max(), r["y_pred"].max()) + 0.05,
            ]
            ax.plot(lims, lims, "r--", lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"{model} Config {config}\nR2={r['R2']:.3f}", fontsize=9)
            ax.set_aspect("equal")
            if i == n_rows - 1:
                ax.set_xlabel("Actual")
            if j == 0:
                ax.set_ylabel("Predicted")

    fig.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "predicted_vs_actual_grid.png", dpi=150)
    plt.close(fig)


def plot_summary_heatmap(results_df, metric, save_path=None):
    """Heatmap of a metric (R2, RMSE, or MAE) with models on rows and configs on columns.

    Uses RdYlGn colourmap for R2 (green=good, red=bad) and reversed for error
    metrics (RMSE, MAE). Cell values are annotated with 3 decimal places.

    Args:
        results_df: Summary DataFrame with columns Model, Config, R2, RMSE, MAE.
        metric: Which metric to display ("R2", "RMSE", or "MAE").
        save_path: File path to save the heatmap.
    """
    pivot = results_df.pivot(index="Model", columns="Config", values=metric)
    model_order = ["Ridge", "PLS", "SVR", "XGBoost", "RF", "MLP", "CNN"]
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = "RdYlGn" if metric == "R2" else "RdYlGn_r"
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax, linewidths=0.5)
    ax.set_title(f"LOOCV {metric} by Model and Input Config")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bar(results_df, save_path=None):
    """Grouped bar chart comparing R2 across all models, with separate bars per config.

    Each model gets a cluster of 3 bars (Config A, B, C). A horizontal line at
    R2=0 marks the baseline (predicting the mean). This plot makes the Config B
    dominance pattern immediately visible.

    Args:
        results_df: Summary DataFrame.
        save_path: File path to save the bar chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    model_order = ["Ridge", "PLS", "SVR", "XGBoost", "RF", "MLP", "CNN"]
    df = results_df.copy()
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values("Model")

    configs = df["Config"].unique()
    x = np.arange(len(model_order))
    width = 0.25

    for i, cfg in enumerate(sorted(configs)):
        subset = df[df["Config"] == cfg]
        vals = []
        for m in model_order:
            row = subset[subset["Model"] == m]
            vals.append(row["R2"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=f"Config {cfg}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_order)
    ax.set_ylabel("R2")
    ax.set_title("LOOCV R2 Comparison")
    ax.legend()
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_plots(all_results, results_df):
    """Master plotting function — generates all Phase 1 visualisations.

    Produces:
      - Individual predicted-vs-actual scatter for each model-config
      - Combined grid of all scatter plots
      - Heatmaps for R2, RMSE, and MAE
      - Grouped bar chart of R2 comparison

    Args:
        all_results: List of result dicts from run_all_evaluations().
        results_df: Summary DataFrame from run_all_evaluations().
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Individual scatter plots per model-config
    for r in all_results:
        fname = f"predicted_vs_actual_{r['model']}_{r['config']}.png"
        plot_predicted_vs_actual(r, FIGURES_DIR / fname)

    # Combined grid of all scatter plots
    plot_all_predicted_vs_actual(all_results, FIGURES_DIR)

    # Heatmaps for each metric
    for metric in ["R2", "RMSE", "MAE"]:
        plot_summary_heatmap(results_df, metric, FIGURES_DIR / f"heatmap_{metric}.png")

    # Grouped bar chart
    plot_model_comparison_bar(results_df, FIGURES_DIR / "model_comparison_bar.png")

    print(f"All plots saved to {FIGURES_DIR}")
