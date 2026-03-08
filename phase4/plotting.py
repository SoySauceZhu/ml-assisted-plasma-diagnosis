import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from .config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT, ALL_FEATURE_NAMES_C

plt.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": FIGURE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def _savefig(fig, name):
    path = FIGURES_DIR / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"    Saved {path.name}")


def fig1_importance_heatmap(importance_df):
    """Multi-model feature importance heatmap (4 models x 17 features).

    Features ordered by consensus rank. Cells show normalised importance.
    """
    df = importance_df.sort_values("consensus_rank")
    features = df["feature"].values

    # Build heatmap matrix (4 rows x 17 cols)
    matrix = np.array([
        df["ridge_importance"].values,
        df["pls_vip"].values,
        df["rf_perm_importance"].values,
        df["mlp_shap"].values,
    ])
    model_labels = ["Ridge", "PLS", "RF", "MLP"]

    # Build annotation matrix with ranks
    annot = np.array([
        df["ridge_rank"].values,
        df["pls_rank"].values,
        df["rf_rank"].values,
        df["mlp_rank"].values,
    ]).astype(str)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.heatmap(
        matrix, ax=ax, xticklabels=features, yticklabels=model_labels,
        cmap="YlOrRd", annot=annot, fmt="s", annot_kws={"size": 7},
        linewidths=0.5, cbar_kws={"label": "Normalised importance", "shrink": 0.8},
    )
    ax.set_title("Multi-Model Feature Importance (Config C)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, "fig1_feature_importance_heatmap")


def fig2_shap_beeswarm(shap_values, X_test_all, feature_names):
    """SHAP beeswarm plot for MLP Config C."""
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X_test_all,
        feature_names=feature_names,
        show=False, plot_size=None,
    )
    plt.title("SHAP Summary — MLP Config C", fontsize=12)
    plt.tight_layout()
    _savefig(plt.gcf(), "fig2_shap_beeswarm_mlp_C")


def fig3_shap_dependence(shap_values, X_test_all, feature_names, top_n=4):
    """SHAP dependence plots for top features by mean |SHAP|."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, feat_idx in enumerate(top_indices):
        ax = axes[i]
        plt.sca(ax)
        shap.dependence_plot(
            feat_idx, shap_values, X_test_all,
            feature_names=feature_names,
            show=False, ax=ax,
        )
        ax.set_title(feature_names[feat_idx], fontsize=10)

    fig.suptitle("SHAP Dependence — Top 4 Features (MLP Config C)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "fig3_shap_dependence_top4")


def fig4_stability_errorbars(stability_df):
    """Feature importance stability: bar chart with error bars from 20 LOOCV folds.

    2x2 grid, one panel per model.
    """
    models = ["Ridge", "PLS", "RF", "MLP"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for i, model_name in enumerate(models):
        ax = axes[i]
        sub = stability_df[stability_df["model"] == model_name].copy()
        sub = sub.sort_values("mean_importance", ascending=True)

        ax.barh(
            sub["feature"], sub["mean_importance"],
            xerr=sub["std_importance"],
            capsize=2, color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5,
        )
        ax.set_xlabel("Importance (mean ± std)")
        ax.set_title(f"{model_name}", fontsize=11)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Feature Importance Stability Across 20 LOOCV Folds (Config C)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "fig4_importance_stability_errorbars")


def fig5_bootstrap_distributions(bootstrap_df):
    """Bootstrap R2 distribution plots.

    2x2 grid: rows = Ridge/MLP, cols = Config B/Config C.
    """
    distributions = bootstrap_df.attrs.get("distributions", {})
    pairs = [
        ("Ridge", "B"), ("Ridge", "C"),
        ("MLP", "B"), ("MLP", "C"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, (model, config) in enumerate(pairs):
        ax = axes[i]
        key = (model, config)
        dist = distributions.get(key, [])

        if len(dist) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{model} Config {config}")
            continue

        dist = np.array(dist)
        ax.hist(dist, bins=30, density=True, alpha=0.6, color="steelblue",
                edgecolor="navy", linewidth=0.5)
        ax.axvline(np.mean(dist), color="red", linewidth=1.5, label="Mean")

        lo = np.percentile(dist, 2.5)
        hi = np.percentile(dist, 97.5)
        ax.axvline(lo, color="orange", linewidth=1.2, linestyle="--", label=f"95% CI")
        ax.axvline(hi, color="orange", linewidth=1.2, linestyle="--")

        # Get point estimate from the dataframe
        row = bootstrap_df[
            (bootstrap_df["Model"] == model) & (bootstrap_df["Config"] == config)
        ]
        if len(row) > 0:
            point = row["R2_point"].values[0]
            ax.axvline(point, color="green", linewidth=1.5, linestyle=":",
                       label=f"Point R²={point:.3f}")

        ax.set_xlabel("R²")
        ax.set_ylabel("Density")
        ax.set_title(f"{model} Config {config}")
        ax.legend(fontsize=7)

    fig.suptitle("Bootstrap R² Distributions (500 iterations)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "fig5_bootstrap_r2_distributions")


def fig6_residual_pred_vs_actual(residual_df, data):
    """Predicted vs actual scatter for Ridge C and MLP C.

    1x2 subplot. Colour by experimental condition. 1:1 line + outlier annotations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, model_name in enumerate(["Ridge", "MLP"]):
        ax = axes[i]
        sub = residual_df[residual_df["Model"] == model_name].copy()

        conditions = sub["condition"].unique()
        cmap = plt.cm.tab10
        color_map = {c: cmap(j / max(len(conditions) - 1, 1))
                     for j, c in enumerate(conditions)}

        for cond in conditions:
            mask = sub["condition"] == cond
            ax.scatter(
                sub.loc[mask, "y_true"], sub.loc[mask, "y_pred"],
                c=[color_map[cond]], label=cond, edgecolors="k",
                linewidth=0.5, s=40, alpha=0.8,
            )

        # 1:1 line
        all_vals = np.concatenate([sub["y_true"].values, sub["y_pred"].values])
        mn, mx = all_vals.min() - 0.05, all_vals.max() + 0.05
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="1:1")

        # +/- 1 sigma bands
        sigma = sub["residual"].std()
        ax.fill_between(
            [mn, mx], [mn - sigma, mx - sigma], [mn + sigma, mx + sigma],
            alpha=0.1, color="red",
        )

        # Annotate outliers
        outliers = sub[sub["is_outlier"]]
        for _, row in outliers.iterrows():
            ax.annotate(
                f"S{int(row['Sample'])}",
                (row["y_true"], row["y_pred"]),
                fontsize=7, color="red",
                xytext=(5, 5), textcoords="offset points",
            )

        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_xlabel("Actual H₂O₂ Rate")
        ax.set_ylabel("Predicted H₂O₂ Rate")
        ax.set_title(f"{model_name} Config C")
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Predicted vs Actual — Config C", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, "fig6_residual_pred_vs_actual")


def fig7_correlation_vif(corr_df, vif_df):
    """Feature correlation matrix + VIF bar chart.

    1x2 subplot. Left: 13x13 heatmap. Right: horizontal VIF bars with threshold.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: correlation heatmap
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    sns.heatmap(
        corr_df, ax=ax1, mask=mask, cmap="RdBu_r", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 6},
        linewidths=0.5, vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax1.set_title("OES Feature Correlation Matrix", fontsize=11)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax1.get_yticklabels(), fontsize=7)

    # Right: VIF bar chart
    vif_sorted = vif_df.sort_values("VIF", ascending=True)
    colors = ["red" if v else "steelblue" for v in vif_sorted["is_high_vif"]]
    ax2.barh(vif_sorted["feature"], vif_sorted["VIF"], color=colors,
             edgecolor="navy", linewidth=0.5)
    ax2.axvline(x=10, color="red", linewidth=1.5, linestyle="--", label="VIF = 10")
    ax2.set_xlabel("VIF")
    ax2.set_title("Variance Inflation Factor", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _savefig(fig, "fig7_feature_correlation_vif")


def fig8_chemistry_mapping(importance_df=None):
    """Chemistry mapping table: feature -> species -> H2O2 pathway."""
    mapping = [
        ["I_309_OH", "OH (A²Σ⁺→X²Π, 0-0)", "·OH + ·OH → H₂O₂", "Direct precursor"],
        ["I_777_O", "Atomic O (⁵S°→⁵P)", "O + H₂O → 2·OH", "Indirect (OH generation)"],
        ["I_656_Ha", "Hα (Balmer n=3→2)", "H₂O → ·OH + H·", "Dissociation indicator"],
        ["I_486_Hb", "Hβ (Balmer n=4→2)", "Balmer decrement pair", "Electron temp. proxy"],
        ["I_337_N2", "N₂ SPS (C³Πᵤ→B³Πg)", "e⁻ energy marker", "High energy → dissociation"],
        ["I_406_CO2p", "CO₂⁺ FDB (A²Πᵤ→X²Πg)", "CO₂ → CO₂⁺ + e⁻", "O radical pool"],
        ["I_516_C2", "C₂ Swan (d³Πg→a³Πᵤ)", "Deep CO₂ decomposition", "Carbon chain marker"],
        ["band_OH_306_312", "OH 0-0 band integral", "Full P,Q,R branches", "Noise-robust OH"],
        ["band_CO2p_398_412", "CO₂⁺ FDB band", "Aggregate ionisation", "Robust CO₂⁺"],
        ["band_CO_Hb_460_500", "CO Angstrom + Hβ", "CO abundance + H₂O dissoc.", "Composite"],
        ["ratio_309_656", "OH/Hα ratio", "OH recomb. availability", "Self-normalised"],
        ["ratio_777_309", "O/OH ratio", "Radical pool balance", "Self-normalised"],
        ["ratio_656_486", "Hα/Hβ (Balmer dec.)", "Electron temp./density", "Self-normalised"],
        ["frequency_hz", "Discharge frequency", "Energy deposition rate", "Discharge param"],
        ["pulse_width_ns", "Pulse width", "Energy per pulse", "Discharge param"],
        ["rise_time_ns", "Rise time", "dV/dt, field strength", "Discharge param"],
        ["flow_rate_sccm", "Gas flow rate", "Residence time", "Discharge param"],
    ]

    # Add consensus rank if available
    if importance_df is not None:
        rank_map = dict(zip(importance_df["feature"], importance_df["consensus_rank"]))
        for row in mapping:
            row.insert(1, str(rank_map.get(row[0], "—")))
        col_labels = ["Feature", "Rank", "Species / Parameter", "Pathway / Role",
                       "Mechanism"]
    else:
        col_labels = ["Feature", "Species / Parameter", "Pathway / Role", "Mechanism"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table = ax.table(
        cellText=mapping,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(mapping) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor("#D9E2F3")

    ax.set_title(
        "ML Feature Importance → Plasma Chemistry Mapping",
        fontsize=12, fontweight="bold", pad=20,
    )
    fig.tight_layout()
    _savefig(fig, "fig8_chemistry_mapping")


def generate_all_phase4_plots(importance_df, shap_values, X_test_all,
                               stability_df, bootstrap_df, residual_df,
                               corr_df, vif_df, data):
    """Master function: generate all 8 publication figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Generating Fig 1: Feature importance heatmap...")
    fig1_importance_heatmap(importance_df)

    print("  Generating Fig 2: SHAP beeswarm...")
    fig2_shap_beeswarm(shap_values, X_test_all, ALL_FEATURE_NAMES_C)

    print("  Generating Fig 3: SHAP dependence...")
    fig3_shap_dependence(shap_values, X_test_all, ALL_FEATURE_NAMES_C)

    print("  Generating Fig 4: Importance stability...")
    fig4_stability_errorbars(stability_df)

    print("  Generating Fig 5: Bootstrap distributions...")
    fig5_bootstrap_distributions(bootstrap_df)

    print("  Generating Fig 6: Residual analysis...")
    fig6_residual_pred_vs_actual(residual_df, data)

    print("  Generating Fig 7: Correlation + VIF...")
    fig7_correlation_vif(corr_df, vif_df)

    print("  Generating Fig 8: Chemistry mapping...")
    fig8_chemistry_mapping(importance_df)

    print(f"  All figures saved to {FIGURES_DIR}")
