import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .config import PCA_VARIANCE_THRESHOLD, FIGURES_DIR


DIAGNOSTIC_LINES = {
    308: "OH",
    337: r"N$_2$",
    656: r"H$\alpha$",
    777: "O",
}


def fit_pca(oes_scaled, n_components=None):
    if n_components is None:
        n_components = min(oes_scaled.shape) - 1
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(oes_scaled)
    return pca, scores


def determine_optimal_k(pca, threshold=PCA_VARIANCE_THRESHOLD):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, threshold) + 1)
    return min(k, len(cumvar))


def plot_scree(pca, save_path=None):
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
    n_components = min(n_components, pca.n_components_)
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wavelengths, pca.components_[i], color="steelblue", linewidth=0.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Loading")
        ax.set_title(f"PC{i+1} Loadings ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    oes_scaled = scaler.fit_transform(data["oes_raw"])

    pca, scores = fit_pca(oes_scaled)
    k_optimal = determine_optimal_k(pca)

    print(f"PCA: {k_optimal} components explain >= {PCA_VARIANCE_THRESHOLD*100:.0f}% variance")
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(10, len(cumvar))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%  (cumulative: {cumvar[i]*100:.1f}%)")

    plot_scree(pca, FIGURES_DIR / "pca_scree_plot.png")
    plot_cumulative_variance(pca, k_optimal, FIGURES_DIR / "pca_cumulative_variance.png")
    plot_loadings(pca, data["wavelengths"], n_components=min(5, k_optimal + 1),
                  save_path_prefix=str(FIGURES_DIR / "pca_loading"))
    plot_scores_2d(scores, data["target"], data["sample_info"],
                   FIGURES_DIR / "pca_scores_2d.png")

    return pca, k_optimal
