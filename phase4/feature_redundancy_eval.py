"""
Phase 4 Supplementary: Ablation Study Visualisation & Statistical Testing.

This module generates:
- fig9: Backward elimination trajectory (Ridge + MLP overlay)
- fig10: Category ablation bar chart
- fig11: VIF bar chart (enhanced)
- fig12: Permutation test for pruned model

Also produces:
- ablation_results_mlp.csv: MLP backward elimination
- permutation_test_pruned_ridge.csv: Permutation test results
- ablation_summary_article.csv: Combined summary table

Usage:
    python -m phase4.feature_redundancy_eval
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy import stats
import torch

from phase1.data_loader import prepare_data
from phase3.feature_engineer import extract_oes_features
from phase1.models.mlp import MLPModel
from phase1.evaluation import compute_metrics
from .config import (
    RANDOM_SEED, RESULTS_DIR, TABLES_DIR, FIGURES_DIR,
    OES_FEATURE_NAMES, ALL_FEATURE_NAMES_C,
    PHASE3_TUNED_PARAMS_PATH,
)


# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def _load_data():
    """Load data and OES features."""
    data = prepare_data()
    oes_features, _ = extract_oes_features(data["oes_raw"], data["wavelengths"])
    return data, oes_features


def _run_mlp_loocv_subset(oes_subset, data, mlp_cfg):
    """Run MLP LOOCV on Config C with a subset of OES features.

    Args:
        oes_subset: (20, k) where k <= 13, the OES feature subset
        data: dict from prepare_data()
        mlp_cfg: MLP configuration dict (with hidden_sizes, dropout, lr, etc.)

    Returns:
        dict with R2, RMSE, MAE
    """
    discharge = data["discharge_raw"]
    target = data["target"]
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(oes_subset):
        # Scale OES subset
        oes_scaler = StandardScaler()
        oes_tr_s = oes_scaler.fit_transform(oes_subset[train_idx])
        oes_te_s = oes_scaler.transform(oes_subset[test_idx])

        # Scale discharge
        dis_scaler = StandardScaler()
        dis_tr_s = dis_scaler.fit_transform(discharge[train_idx])
        dis_te_s = dis_scaler.transform(discharge[test_idx])

        # Config C: hstack
        X_train = np.hstack([oes_tr_s, dis_tr_s])
        X_test = np.hstack([oes_te_s, dis_te_s])

        # MLP with fixed seed for reproducibility
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # MLPModel takes a config dict directly
        model = MLPModel(config=mlp_cfg)
        model.fit(X_train, target[train_idx])
        pred = model.predict(X_test)

        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    return compute_metrics(np.array(y_true_all), np.array(y_pred_all))


def _run_mlp_ablation_backward_elimination(oes_features, data, feature_names, mlp_cfg):
    """MLP backward elimination ablation study.

    Args:
        oes_features: (20, 13) OES feature matrix
        data: dict from prepare_data()
        feature_names: list of 13 OES feature names
        mlp_cfg: MLP configuration dict

    Returns:
        DataFrame: n_oes_features, removed_feature, remaining_features, R2, RMSE, MAE
    """
    remaining_idx = list(range(len(feature_names)))
    results = []

    # Baseline with all 13
    print("  Running MLP ablation (this may take a few minutes)...")
    baseline = _run_mlp_loocv_subset(oes_features, data, mlp_cfg)
    results.append({
        "n_oes_features": len(remaining_idx),
        "removed_feature": None,
        "remaining_features": ",".join([feature_names[i] for i in remaining_idx]),
        "R2": baseline["R2"],
        "RMSE": baseline["RMSE"],
        "MAE": baseline["MAE"],
    })

    # Use Ridge coefficients to determine removal order (same as original ablation)
    from phase1.models.ridge import RidgeModel
    from .config import RIDGE_ALPHAS

    while len(remaining_idx) > 3:
        oes_subset = oes_features[:, remaining_idx]
        discharge = data["discharge_raw"]
        target = data["target"]

        # Fit Ridge on full dataset to get coefficient magnitudes for removal order
        oes_scaler = StandardScaler()
        dis_scaler = StandardScaler()
        oes_s = oes_scaler.fit_transform(oes_subset)
        dis_s = dis_scaler.fit_transform(discharge)
        X_full = np.hstack([oes_s, dis_s])

        model = RidgeModel(alphas=RIDGE_ALPHAS)
        model.fit(X_full, target)

        # OES coefficients are the first k entries
        n_oes = len(remaining_idx)
        oes_coefs = np.abs(model.model.coef_[:n_oes])

        # Remove the least important OES feature (same order as Ridge ablation)
        least_idx_local = np.argmin(oes_coefs)
        removed_global = remaining_idx[least_idx_local]
        remaining_idx.remove(removed_global)

        # Evaluate with reduced subset using MLP
        oes_subset_new = oes_features[:, remaining_idx]
        metrics = _run_mlp_loocv_subset(oes_subset_new, data, mlp_cfg)

        results.append({
            "n_oes_features": len(remaining_idx),
            "removed_feature": feature_names[removed_global],
            "remaining_features": ",".join([feature_names[i] for i in remaining_idx]),
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
        })

        print(f"    MLP {len(remaining_idx)} OES: R²={metrics['R2']:.4f}")

    return pd.DataFrame(results)


def run_permutation_test(oes_features, data, n_permutations=2000):
    """Permutation test for statistical significance of pruned model.

    Uses the optimal 3-ratio + 4-discharge = 7 feature subset.

    Args:
        oes_features: (20, 13) OES feature matrix
        data: dict from prepare_data()
        n_permutations: number of permutation iterations

    Returns:
        (observed_r2, null_r2_array, p_value)
    """
    from phase1.models.ridge import RidgeModel
    from .config import RIDGE_ALPHAS

    # Optimal feature indices: 3 ratios = indices 10, 11, 12
    ratio_indices = [10, 11, 12]  # ratio_309_656, ratio_777_309, ratio_656_486
    oes_subset = oes_features[:, ratio_indices]
    discharge = data["discharge_raw"]
    target = data["target"]

    # Build X = ratios + discharge
    oes_scaler = StandardScaler()
    dis_scaler = StandardScaler()
    oes_s = oes_scaler.fit_transform(oes_subset)
    dis_s = dis_scaler.fit_transform(discharge)
    X = np.hstack([oes_s, dis_s])

    # Observed R² with LOOCV
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in loo.split(X):
        model = RidgeModel(alphas=RIDGE_ALPHAS)
        model.fit(X[train_idx], target[train_idx])
        pred = model.predict(X[test_idx])
        y_true_all.append(target[test_idx][0])
        y_pred_all.append(pred[0])

    observed_metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
    observed_r2 = observed_metrics["R2"]

    # Permutation test
    null_r2_values = []
    print(f"  Running {n_permutations} permutations...")
    for i in range(n_permutations):
        target_perm = target.copy()
        np.random.shuffle(target_perm)

        y_true_perm, y_pred_perm = [], []
        for train_idx, test_idx in loo.split(X):
            model = RidgeModel(alphas=RIDGE_ALPHAS)
            model.fit(X[train_idx], target_perm[train_idx])
            pred = model.predict(X[test_idx])
            y_true_perm.append(target_perm[test_idx][0])
            y_pred_perm.append(pred[0])

        null_metrics = compute_metrics(np.array(y_true_perm), np.array(y_pred_perm))
        null_r2_values.append(null_metrics["R2"])

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n_permutations} permutations done")

    null_r2_array = np.array(null_r2_values)
    p_value = np.mean(null_r2_array >= observed_r2)

    return observed_r2, null_r2_array, p_value


def fig9_ablation_trajectory(ablation_ridge, ablation_mlp=None):
    """Backward elimination trajectory plot (Ridge + MLP overlay).

    Args:
        ablation_ridge: DataFrame with backward elimination results (Ridge)
        ablation_mlp: DataFrame with backward elimination results (MLP), optional
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ridge trajectory
    ridge_data = ablation_ridge.sort_values("n_oes_features", ascending=True)
    ax.plot(
        ridge_data["n_oes_features"], ridge_data["R2"],
        "o-", color="steelblue", linewidth=2, markersize=8,
        label="Ridge",
    )

    # MLP trajectory (if provided)
    if ablation_mlp is not None:
        mlp_data = ablation_mlp.sort_values("n_oes_features", ascending=True)
        ax.plot(
            mlp_data["n_oes_features"], mlp_data["R2"],
            "s--", color="coral", linewidth=2, markersize=8,
            label="MLP",
        )

    # Reference lines
    ax.axhline(y=0.798, color="gray", linestyle="--", linewidth=1.2,
               label="Full 13 OES (R²=0.798)")
    ax.axhline(y=0.904, color="green", linestyle="-.", linewidth=1.2,
               label="Config B (R²=0.904)")

    # Highlight optimal point (4 OES features)
    optimal = ridge_data[ridge_data["n_oes_features"] == 4]
    if len(optimal) > 0:
        ax.scatter(
            4, optimal["R2"].values[0],
            marker="*", s=300, color="gold", edgecolor="black",
            linewidth=1.5, zorder=5,
        )
        ax.annotate(
            f"Optimal: 4 OES\nR²={optimal['R2'].values[0]:.3f}",
            (4, optimal["R2"].values[0]),
            textcoords="offset points", xytext=(15, 10),
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )

    # Shade region where pruned model outperforms full model
    ax.axvspan(3, 4.5, alpha=0.15, color="green",
               label="Pruned > Full")

    ax.set_xlabel("Number of OES Features", fontsize=11)
    ax.set_ylabel("LOOCV R²", fontsize=11)
    ax.set_title("Backward Elimination: Performance Improves with Feature Pruning",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(2.5, 14)
    ax.set_ylim(0.75, 0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save PDF and PNG
    path_pdf = FIGURES_DIR / "fig9_ablation_trajectory.pdf"
    path_png = FIGURES_DIR / "fig9_ablation_trajectory.png"
    fig.savefig(path_pdf, dpi=300)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"  Saved {path_pdf.name} and {path_png.name}")


def fig10_category_ablation(ablation_ridge):
    """Category ablation bar chart.

    Args:
        ablation_ridge: DataFrame with ablation results including category rows
    """
    # Extract category results
    cat_data = ablation_ridge[ablation_ridge["type"] == "category"].copy()
    cat_data = cat_data.sort_values("R2", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = cat_data["removed_feature"].values  # This contains category names
    r2_values = cat_data["R2"].values
    n_oes = cat_data["n_oes_features"].values
    n_total = cat_data["n_total_features"].values

    x = np.arange(len(categories))
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]

    bars = ax.bar(x, r2_values, color=colors, edgecolor="navy", linewidth=1.2)

    # Annotate bars with R² and feature count
    for i, (bar, r2, n_o, n_t) in enumerate(zip(bars, r2_values, n_oes, n_total)):
        height = bar.get_height()
        ax.annotate(
            f"R²={r2:.3f}\n(n={int(n_o)} OES + 4)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    # Reference line
    ax.axhline(y=0.904, color="green", linestyle="-.", linewidth=1.5,
               label="Config B (R²=0.904)")

    # Format x-tick labels
    labels = [
        "Ratios\n(3 OES + 4)",
        "Band Integrals\n(3 OES + 4)",
        "Single-wavelength\n(7 OES + 4)",
        "All 13 OES\n(+ 4 discharge)",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylabel("LOOCV R²", fontsize=11)
    ax.set_title("Category Ablation: Ratios and Band Integrals Outperform Single Wavelengths",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.75, 0.95)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    path_pdf = FIGURES_DIR / "fig10_category_ablation.pdf"
    path_png = FIGURES_DIR / "fig10_category_ablation.png"
    fig.savefig(path_pdf, dpi=300)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"  Saved {path_pdf.name} and {path_png.name}")


def fig11_vif_barchart(vif_df, optimal_features=None):
    """Enhanced VIF bar chart with log scale and optimal feature markers.

    Args:
        vif_df: DataFrame with VIF values
        optimal_features: list of feature names that are in the optimal 4-feature subset
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    vif_sorted = vif_df.sort_values("VIF", ascending=True)

    # Colors: red for high VIF, green for acceptable
    colors = ["#70AD47" if v <= 10 else "#C00000" for v in vif_sorted["VIF"]]

    bars = ax.barh(vif_sorted["feature"], vif_sorted["VIF"],
                   color=colors, edgecolor="navy", linewidth=0.5)

    # Mark optimal features
    if optimal_features is not None:
        for i, (feat, vif) in enumerate(zip(vif_sorted["feature"], vif_sorted["VIF"])):
            if feat in optimal_features:
                bars[i].set_edgecolor("gold")
                bars[i].set_linewidth(3)

    # Log scale for better visualization
    ax.set_xscale("log")

    # Threshold line
    ax.axvline(x=10, color="red", linewidth=2, linestyle="--",
               label="VIF = 10 (threshold)")

    # Annotate extreme values
    for feat in ["I_309_OH", "band_OH_306_312"]:
        row = vif_sorted[vif_sorted["feature"] == feat]
        if len(row) > 0:
            ax.annotate(
                f"{row['VIF'].values[0]:.1f}",
                (row["VIF"].values[0], feat),
                textcoords="offset points", xytext=(5, 0),
                fontsize=9, fontweight="bold", color="red",
            )

    ax.set_xlabel("VIF (log scale)", fontsize=11)
    ax.set_title("Variance Inflation Factor: Severe Multicollinearity Among OES Features",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    # Add annotation for optimal features
    if optimal_features is not None:
        ax.annotate(
            f"★ = optimal 4-feature subset\n({', '.join(optimal_features)})",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    path_pdf = FIGURES_DIR / "fig11_vif_barchart.pdf"
    path_png = FIGURES_DIR / "fig11_vif_barchart.png"
    fig.savefig(path_pdf, dpi=300)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"  Saved {path_pdf.name} and {path_png.name}")


def fig12_permutation_test(observed_r2, null_r2_array, p_value):
    """Permutation test visualization.

    Args:
        observed_r2: Observed R² value
        null_r2_array: Array of null R² values from permutations
        p_value: p-value
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of null distribution
    ax.hist(null_r2_array, bins=40, density=True, alpha=0.6,
            color="steelblue", edgecolor="navy", linewidth=0.5)

    # Observed R² line
    ax.axvline(x=observed_r2, color="red", linewidth=2.5,
               label=f"Observed R² = {observed_r2:.3f}")

    # Add p-value annotation
    ax.annotate(
        f"p-value = {p_value:.4f}\n(n={len(null_r2_array)} permutations)",
        xy=(observed_r2, ax.get_ylim()[1] * 0.8),
        xytext=(observed_r2 + 0.1, ax.get_ylim()[1] * 0.85),
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # Shade p-value region
    x_fill = null_r2_array[null_r2_array >= observed_r2]
    if len(x_fill) > 0:
        ax.axvspan(observed_r2, max(null_r2_array) + 0.05,
                   alpha=0.2, color="red", label="p-value region")

    ax.set_xlabel("R² (null distribution)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Permutation Test: Statistical Significance of Pruned Model (3 Ratios + 4 Discharge)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path_pdf = FIGURES_DIR / "fig12_permutation_test.pdf"
    path_png = FIGURES_DIR / "fig12_permutation_test.png"
    fig.savefig(path_pdf, dpi=300)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"  Saved {path_pdf.name} and {path_png.name}")


def create_ablation_summary(ablation_ridge, ablation_mlp, category_ablation):
    """Create comprehensive ablation summary table for the article.

    Args:
        ablation_ridge: Ridge backward elimination results
        ablation_mlp: MLP backward elimination results
        category_ablation: Category ablation results

    Returns:
        DataFrame with all ablation results combined
    """
    rows = []

    # Backward elimination (Ridge)
    for _, row in ablation_ridge[ablation_ridge["type"] == "backward_elimination"].iterrows():
        rows.append({
            "step": int(row["n_oes_features"]),
            "type": "backward_elimination",
            "model": "Ridge",
            "n_oes_features": int(row["n_oes_features"]),
            "n_total_features": int(row["n_oes_features"]) + 4,
            "oes_features_kept": row["remaining_features"],
            "removed_feature": row["removed_feature"] if pd.notna(row["removed_feature"]) else "None",
            "R2": row["R2"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "note": f"R²={row['R2']:.4f}, removed: {row['removed_feature']}" if pd.notna(row["removed_feature"]) else "Full 13 OES",
        })

    # Backward elimination (MLP) - add R2_mlp column
    if ablation_mlp is not None:
        mlp_dict = {row["n_oes_features"]: row["R2"] for _, row in ablation_mlp.iterrows()}
        for row in rows:
            if row["type"] == "backward_elimination":
                row["R2_mlp"] = mlp_dict.get(row["n_oes_features"], None)

    # Category ablation - category name is in removed_feature column
    for _, row in category_ablation.iterrows():
        cat_name = row["removed_feature"]  # This is the category name
        rows.append({
            "step": int(row["n_oes_features"]),
            "type": "category",
            "model": "Ridge",
            "n_oes_features": int(row["n_oes_features"]),
            "n_total_features": int(row["n_total_features"]),
            "oes_features_kept": row["features"],
            "removed_feature": cat_name,
            "R2": row["R2"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "note": cat_name,
        })

    # Add Config B reference
    rows.append({
        "step": 0,
        "type": "reference",
        "model": "Ridge",
        "n_oes_features": 0,
        "n_total_features": 4,
        "oes_features_kept": "None (discharge only)",
        "removed_feature": "Config B",
        "R2": 0.904,
        "RMSE": 0.071,
        "MAE": None,
        "note": "Discharge parameters only (4 features)",
    })

    return pd.DataFrame(rows)


def main():
    """Run all supplementary analyses."""
    print("\n" + "=" * 60)
    print("Phase 4 Supplementary: Ablation Study Visualisation")
    print("=" * 60)

    # Ensure output directories exist
    for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data, oes_features = _load_data()

    # Load tuned hyperparameters
    with open(PHASE3_TUNED_PARAMS_PATH) as f:
        tuned = json.load(f)
    mlp_cfg = tuned["MLP_C"]
    if isinstance(mlp_cfg.get("hidden_sizes"), str):
        mlp_cfg["hidden_sizes"] = [int(x) for x in mlp_cfg["hidden_sizes"].split("_")]

    # Load existing ablation results
    ablation_ridge = pd.read_csv(TABLES_DIR / "ablation_results.csv")
    vif_df = pd.read_csv(TABLES_DIR / "feature_correlation_vif.csv")

    # Check if MLP ablation already exists
    mlp_ablation_path = TABLES_DIR / "ablation_results_mlp.csv"
    if mlp_ablation_path.exists():
        print("\nLoading existing MLP ablation results...")
        ablation_mlp = pd.read_csv(mlp_ablation_path)
    else:
        print("\nRunning MLP backward elimination (this takes ~3-5 minutes)...")
        ablation_mlp = _run_mlp_ablation_backward_elimination(
            oes_features, data, OES_FEATURE_NAMES, mlp_cfg
        )
        ablation_mlp.to_csv(mlp_ablation_path, index=False)
        print(f"  Saved MLP ablation results")

    # Extract category results
    category_ablation = ablation_ridge[ablation_ridge["type"] == "category"].copy()

    # Generate figures
    print("\nGenerating supplementary figures...")

    # Fig 9: Ablation trajectory
    print("  Fig 9: Backward elimination trajectory...")
    fig9_ablation_trajectory(ablation_ridge, ablation_mlp)

    # Fig 10: Category ablation  Fig 10
    print(": Category ablation bar chart...")
    fig10_category_ablation(ablation_ridge)

    # Fig 11: VIF bar chart (enhanced)
    print("  Fig 11: VIF bar chart...")
    optimal_4_features = ["I_486_Hb", "I_516_C2", "band_CO2p_398_412", "ratio_656_486"]
    fig11_vif_barchart(vif_df, optimal_4_features)

    # Check if permutation test already exists
    perm_test_path = TABLES_DIR / "permutation_test_pruned_ridge.csv"
    if perm_test_path.exists():
        print("\nLoading existing permutation test results...")
        perm_df = pd.read_csv(perm_test_path)
        observed_r2 = perm_df[perm_df["type"] == "observed"]["r2"].values[0]
        null_r2_array = perm_df[perm_df["type"] == "null"]["r2"].values
        p_value = np.mean(null_r2_array >= observed_r2)
    else:
        print("\nRunning permutation test (2000 iterations)...")
        observed_r2, null_r2_array, p_value = run_permutation_test(oes_features, data)
        # Save permutation test results
        perm_df = pd.DataFrame({
            "type": ["observed"] + ["null"] * len(null_r2_array),
            "r2": [observed_r2] + list(null_r2_array),
        })
        perm_df.to_csv(perm_test_path, index=False)
        # Also save summary
        perm_summary = pd.DataFrame({
            "observed_r2": [observed_r2],
            "p_value": [p_value],
            "n_permutations": [len(null_r2_array)],
        })
        perm_summary.to_csv(
            TABLES_DIR / "permutation_test_summary.csv",
            index=False
        )
        print(f"  Permutation test: R²={observed_r2:.4f}, p={p_value:.4f}")

    # Fig 12: Permutation test
    print("  Fig 12: Permutation test visualization...")
    fig12_permutation_test(observed_r2, null_r2_array, p_value)

    # Create summary table
    print("\nCreating ablation summary table...")
    summary_df = create_ablation_summary(ablation_ridge, ablation_mlp, category_ablation)
    summary_df.to_csv(TABLES_DIR / "ablation_summary_article.csv", index=False)
    print(f"  Saved ablation_summary_article.csv")

    print("\n" + "=" * 60)
    print("Supplementary Analysis Complete!")
    print("=" * 60)
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
