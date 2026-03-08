"""
Phase 4: Interpretability, Stability & Residual Analysis.

Usage:
    python -m phase4.main
"""
import json
import numpy as np
import pandas as pd
import torch
import warnings

from phase1.data_loader import prepare_data
from phase3.feature_engineer import extract_oes_features
from .config import (
    RANDOM_SEED, RESULTS_DIR, TABLES_DIR, FIGURES_DIR,
    PHASE3_TUNED_PARAMS_PATH, PHASE3_PREDICTIONS_PATH,
    ALL_FEATURE_NAMES_C, OES_FEATURE_NAMES,
)
from . import interpretability, shap_analysis, stability
from . import residual_analysis, feature_redundancy, plotting

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --- 0. Load data + features + tuned hyperparameters ---
    print("Loading data...")
    data = prepare_data()
    oes_features, feature_names = extract_oes_features(
        data["oes_raw"], data["wavelengths"]
    )
    print(f"  Samples: {len(data['target'])}, OES features: {oes_features.shape[1]}")

    with open(PHASE3_TUNED_PARAMS_PATH) as f:
        tuned = json.load(f)
    rf_params = tuned["RF_C"]
    mlp_cfg = tuned["MLP_C"]
    # Parse hidden_sizes if stored as list already (from JSON)
    if isinstance(mlp_cfg.get("hidden_sizes"), list):
        pass  # already a list
    elif isinstance(mlp_cfg.get("hidden_sizes"), str):
        mlp_cfg["hidden_sizes"] = [int(x) for x in mlp_cfg["hidden_sizes"].split("_")]

    # --- 1. Feature importance (Ridge, PLS, RF) ---
    print("\nStep 1: Feature importance extraction (Config C)...")
    print("  Ridge...")
    ridge_imp = interpretability.ridge_importance_loocv(oes_features, data)
    print("  PLS...")
    pls_imp = interpretability.pls_importance_loocv(oes_features, data)
    print("  RF...")
    rf_imp = interpretability.rf_importance_loocv(oes_features, data, rf_params)

    # --- 2. SHAP analysis (MLP Config C) ---
    print("\nStep 2: SHAP analysis for MLP Config C...")
    shap_vals, X_test_arr = shap_analysis.compute_shap_loocv(
        oes_features, data, mlp_cfg
    )
    mlp_imp = shap_analysis.get_shap_importance(shap_vals)

    # Save raw SHAP values
    shap_df = pd.DataFrame(shap_vals, columns=ALL_FEATURE_NAMES_C)
    shap_df.to_csv(TABLES_DIR / "shap_values_mlp_configC.csv", index=False)
    print(f"  SHAP values saved.")

    # --- 3. Consensus ranking ---
    print("\nStep 3: Building consensus ranking...")
    importance_df = interpretability.build_consensus_table(
        ridge_imp, pls_imp, rf_imp, mlp_imp, ALL_FEATURE_NAMES_C
    )
    importance_df.to_csv(TABLES_DIR / "feature_importance_all_models.csv", index=False)

    # Print Spearman correlations
    spearman = importance_df.attrs.get("spearman_correlations", {})
    if spearman:
        print("  Spearman rank correlations:")
        for pair, (rho, pval) in spearman.items():
            print(f"    {pair}: rho={rho:.3f}, p={pval:.3f}")

    # Print OES vs discharge fraction
    oes_frac = importance_df.attrs.get("oes_fraction", {})
    dis_frac = importance_df.attrs.get("discharge_fraction", {})
    if oes_frac:
        print("  OES vs discharge importance fraction:")
        for model in oes_frac:
            print(f"    {model}: OES={oes_frac[model]:.3f}, "
                  f"Discharge={dis_frac[model]:.3f}")

    print(f"  Consensus table saved ({len(importance_df)} features).")

    # --- 4. Bootstrap CI ---
    print("\nStep 4: Bootstrap confidence intervals...")
    bootstrap_df = stability.bootstrap_all_models(PHASE3_PREDICTIONS_PATH)
    bootstrap_df_save = bootstrap_df.drop(columns=[], errors="ignore")
    bootstrap_df_save.to_csv(TABLES_DIR / "bootstrap_ci_summary.csv", index=False)
    print("  Bootstrap CIs:")
    for _, row in bootstrap_df.iterrows():
        print(f"    {row['Model']} Config {row['Config']}: "
              f"R²={row['R2_point']:.3f} [{row['R2_lo95']:.3f}, {row['R2_hi95']:.3f}]")

    # --- 5. LOOCV importance stability ---
    print("\nStep 5: LOOCV fold importance stability...")
    stability_df = stability.fold_importance_stability(
        ridge_imp, pls_imp, rf_imp, shap_vals, ALL_FEATURE_NAMES_C
    )
    stability_df.to_csv(TABLES_DIR / "loocv_fold_importance_stability.csv", index=False)

    unstable = stability_df[~stability_df["is_stable"]]
    if len(unstable) > 0:
        print(f"  {len(unstable)} unstable feature-model pairs (CV > 1.0):")
        for _, row in unstable.iterrows():
            print(f"    {row['model']}/{row['feature']}: CV={row['cv']:.2f}")
    else:
        print("  All feature-model pairs are stable (CV <= 1.0).")

    # --- 6. Residual analysis ---
    print("\nStep 6: Residual analysis (Config C)...")
    residual_df = residual_analysis.analyse_residuals(PHASE3_PREDICTIONS_PATH, data)
    residual_df.to_csv(TABLES_DIR / "residual_detail.csv", index=False)

    resid_feat_corr = residual_analysis.residual_feature_correlation(
        residual_df, oes_features, data["discharge_raw"], ALL_FEATURE_NAMES_C
    )
    resid_feat_corr.to_csv(TABLES_DIR / "residual_feature_correlation.csv", index=False)

    cond_summary = residual_analysis.condition_grouped_summary(residual_df)
    cond_summary.to_csv(TABLES_DIR / "residual_condition_summary.csv", index=False)

    outliers = residual_df[residual_df["is_outlier"]]
    print(f"  Outlier samples (|residual| > 2σ): {len(outliers)}")
    for _, row in outliers.iterrows():
        print(f"    {row['Model']} Sample {int(row['Sample'])}: "
              f"true={row['y_true']:.3f}, pred={row['y_pred']:.3f}, "
              f"resid={row['residual']:.3f}")

    # --- 7. Feature redundancy ---
    print("\nStep 7: Feature redundancy (correlation + VIF + ablation)...")
    corr_df, vif_df = feature_redundancy.compute_correlation_vif(
        oes_features, OES_FEATURE_NAMES
    )
    vif_df.to_csv(TABLES_DIR / "feature_correlation_vif.csv", index=False)

    high_vif = vif_df[vif_df["is_high_vif"]]
    if len(high_vif) > 0:
        print(f"  High VIF features (>10):")
        for _, row in high_vif.iterrows():
            print(f"    {row['feature']}: VIF={row['VIF']:.1f}")

    print("  Running backward elimination ablation...")
    ablation_be = feature_redundancy.ablation_backward_elimination(
        oes_features, data, OES_FEATURE_NAMES
    )
    print("  Running category ablation...")
    ablation_cat = feature_redundancy.ablation_category(
        oes_features, data, OES_FEATURE_NAMES
    )

    # Combine ablation results
    ablation_all = pd.concat([
        ablation_be.assign(type="backward_elimination"),
        ablation_cat.rename(columns={"category": "removed_feature"}).assign(
            type="category"
        ),
    ], ignore_index=True)
    ablation_all.to_csv(TABLES_DIR / "ablation_results.csv", index=False)

    print("  Ablation — backward elimination:")
    for _, row in ablation_be.iterrows():
        print(f"    {int(row['n_oes_features'])} OES features: "
              f"R²={row['R2']:.4f} (removed: {row['removed_feature']})")
    print("  Ablation — category:")
    for _, row in ablation_cat.iterrows():
        print(f"    {row['category']}: R²={row['R2']:.4f}")

    # --- 8. All figures ---
    print("\nStep 8: Generating publication figures...")
    plotting.generate_all_phase4_plots(
        importance_df=importance_df,
        shap_values=shap_vals,
        X_test_all=X_test_arr,
        stability_df=stability_df,
        bootstrap_df=bootstrap_df,
        residual_df=residual_df,
        corr_df=corr_df,
        vif_df=vif_df,
        data=data,
    )

    # --- 9. Summary ---
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"Tables: {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print(f"\nTop 5 features (consensus ranking):")
    for _, row in importance_df.head(5).iterrows():
        print(f"  #{int(row['consensus_rank'])}: {row['feature']} "
              f"(mean rank: {row['mean_rank']:.1f})")


if __name__ == "__main__":
    main()
