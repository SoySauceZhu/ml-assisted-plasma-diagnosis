"""
PCA Analysis
=============
Performs Principal Component Analysis on the 701-wavelength OES spectrum to
reduce dimensionality while retaining 95% of the total variance. Generates
diagnostic plots: scree plot, cumulative variance, loading vectors (annotated
with known spectral lines), and 2D score scatter.

Result: k=11 components are needed, which means 11/20 feature-to-sample ratio
after PCA — still high, contributing to the Phase 1 overfitting problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .config import PCA_VARIANCE_THRESHOLD, FIGURES_DIR


# Known diagnostic emission lines in the OES spectrum (wavelength -> species label).
# Used to annotate PCA loading plots so we can see which spectral lines
# each principal component captures.
DIAGNOSTIC_LINES = {
    308: "OH",           # OH radical A-X transition at ~308-309 nm
    337: r"N$_2$",       # N2 second positive system at 337 nm
    656: r"H$\alpha$",   # Hydrogen Balmer-alpha at 656 nm
    777: "O",            # Atomic oxygen triplet at 777 nm
}


def fit_pca(oes_scaled, n_components=None):
    """Fit PCA to standardised OES data.

    Args:
        oes_scaled: Standardised OES matrix (n_samples, 701), zero mean and unit variance.
        n_components: Number of components to compute. Default: min(n, p) - 1 = 19.

    Returns:
        tuple: (fitted PCA object, transformed score matrix [n_samples, n_components])
    """
    if n_components is None:
        n_components = min(oes_scaled.shape) - 1
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(oes_scaled)
    return pca, scores


def determine_optimal_k(pca, threshold=PCA_VARIANCE_THRESHOLD):
    """Find the minimum number of PCA components to explain >= threshold variance.

    Uses numpy searchsorted on the cumulative variance array to find the first
    index where cumulative variance crosses the threshold (default 95%).

    Args:
        pca: Fitted PCA object (must have explained_variance_ratio_ attribute).
        threshold: Cumulative variance threshold (default: 0.95 = 95%).

    Returns:
        int: Optimal number of components k (result: k=11 for this dataset).
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, len(cumvar))


def plot_scree(pca, save_path=None):
    """Generate a scree plot showing individual explained variance per component.

    The scree plot helps visualise the "elbow" where additional components
    contribute diminishing returns. Each bar shows one component's contribution.

    Args:
        pca: Fitted PCA object.
        save_path: File path to save the plot (PNG). If None, plot is not saved.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(pca.explained_variance_ratio_)
    ax.bar(range(1, n + 1), pca.explained_variance_ratio_ * 100, color="steelblue")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(range(1, n + 1))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_variance(pca, k_optimal, save_path=None):
    """Plot cumulative explained variance with threshold line and optimal k marker.

    Shows how many components are needed to reach the 95% variance threshold.
    The red dashed line marks the threshold; the green dashed line marks k_optimal.

    Args:
        pca: Fitted PCA object.
        k_optimal: Optimal number of components (marked with vertical line).
        save_path: File path to save the plot.
    """
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="steelblue")
    ax.axhline(PCA_VARIANCE_THRESHOLD * 100, ls="--", color="red", label=f"{PCA_VARIANCE_THRESHOLD*100:.0f}% threshold")
    ax.axvline(k_optimal, ls="--", color="green", label=f"k = {k_optimal}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Cumulative Variance")
    ax.legend()
    ax.set_xticks(range(1, len(cumvar) + 1))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loadings(pca, wavelengths, n_components=3, save_path_prefix=None):
    """Plot PCA loading vectors vs wavelength, annotated with known spectral lines.

    Each loading vector shows how much each wavelength contributes to a given
    principal component. Diagnostic spectral lines (OH, N2, Halpha, O) are
    annotated with arrows to help interpret what each PC captures physically.

    Args:
        pca: Fitted PCA object.
        wavelengths: Array of wavelength values (200-900 nm).
        n_components: Number of loading plots to generate (default: 3).
        save_path_prefix: File path prefix; saves as {prefix}_pc1.png, _pc2.png, etc.
    """
    n_components = min(n_components, pca.n_components_)
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wavelengths, pca.components_[i], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Loading")
        ax.set_title(f"PC{i+1} Loadings ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        # Annotate known diagnostic spectral lines on the loading plot
        for wl, label in DIAGNOSTIC_LINES.items():
            if wavelengths.min() <= wl <= wavelengths.max():
                idx = np.argmin(np.abs(wavelengths - wl))
                ax.annotate(
                    f"{label} {wl}nm",
                    xy=(wl, pca.components_[i, idx]),
                    xytext=(0, 15),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
                    color="red",
                )
        fig.tight_layout()
        if save_path_prefix:
            fig.savefig(f"{save_path_prefix}_pc{i+1}.png", dpi=150)
        plt.close(fig)


def plot_scores_2d(scores, target, sample_info, save_path=None):
    """Scatter plot of PC1 vs PC2 scores, colour-coded by H2O2 yield rate.

    Each point represents one experimental sample. Colour indicates the H2O2
    yield rate (viridis colourmap). Points are annotated with their experimental
    condition label. This plot reveals whether PCA separates high- and low-yield
    samples in the reduced 2D space.

    Args:
        scores: PCA score matrix (n_samples, n_components).
        target: H2O2 yield rate array (n_samples,) for colour coding.
        sample_info: DataFrame with "condition" column for point annotations.
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=target, cmap="viridis", s=80, edgecolors="k")
    for i, row in sample_info.iterrows():
        ax.annotate(row["condition"], (scores[i, 0], scores[i, 1]),
                     fontsize=6, ha="center", va="bottom", xytext=(0, 5),
                     textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("H2O2 rate")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scores (PC1 vs PC2)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_pca_analysis(data):
    """Master function: run complete PCA analysis pipeline.

    Standardises the OES data (zero mean, unit variance), fits PCA, determines
    the optimal number of components, prints variance explanation summary,
    and generates all 4 diagnostic plot types (scree, cumulative, loadings, scores).

    Args:
        data: Data dict from prepare_data() containing "oes_raw", "wavelengths",
              "target", "sample_info".

    Returns:
        tuple: (fitted PCA object, optimal k value). k is used downstream by
               the evaluation module for dimensionality reduction in each LOOCV fold.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Standardise OES data before PCA (PCA is sensitive to feature scales)
    scaler = StandardScaler()
    oes_scaled = scaler.fit_transform(data["oes_raw"])

    pca, scores = fit_pca(oes_scaled)
    k_optimal = determine_optimal_k(pca)

    # Print variance explanation summary for the first 10 components
    print(f"PCA: {k_optimal} components explain >= {PCA_VARIANCE_THRESHOLD*100:.0f}% variance")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(10, len(cumvar))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%  (cumulative: {cumvar[i]*100:.1f}%)")

    # Generate all diagnostic plots
    plot_scree(pca, FIGURES_DIR / "pca_scree_plot.png")
    plot_cumulative_variance(pca, k_optimal, FIGURES_DIR / "pca_cumulative_variance.png")
    plot_loadings(pca, data["wavelengths"], n_components=min(5, k_optimal + 1),
                  save_path_prefix=str(FIGURES_DIR / "pca_loading"))
    plot_scores_2d(scores, data["target"], data["sample_info"],
                   FIGURES_DIR / "pca_scores_2d.png")

    return pca, k_optimal
