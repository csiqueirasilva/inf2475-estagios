import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List

def plot_embeddings_latent_space(
    df: pd.DataFrame,
    embed_col: str = 'embedding',
    reduction: str = 'pca',
    labels: Optional[np.ndarray] = None,
    categories: Optional[List[str]] = None,
    output_path: str = 'latent_space.pdf',
    show: bool = False,
    seed: int = 42
):
    """
    Plot 2D projection of embeddings from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column `embed_col` with array-like embeddings.
    embed_col : str, default 'embedding'
        Name of the column in `df` holding embeddings.
    reduction : str, one of {'pca', 'tsne', 'umap'}, default 'pca'
        Dimensionality reduction method.
    labels : np.ndarray, optional
        Numeric labels for each point (shape: (N,)).
    categories : List[str], optional
        Category names corresponding to label values.
    output_path : str, default 'latent_space.pdf'
        Where to save the figure.
    show : bool, default False
        Whether to display the plot interactively.
    seed : int, default 42
        Random seed for reproducibility.
    """
    # Seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Extract embeddings
    embeds = np.vstack(df[embed_col].values)

    # Dimensionality reduction
    r = reduction.lower()
    if r == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif r == "tsne":
        reducer = TSNE(n_components=2, init="pca", random_state=seed,
                       learning_rate="auto", perplexity=50, metric="cosine")
    elif r == "umap":
        try:
            from umap.umap_ import UMAP
        except ImportError:
            raise ImportError("UMAP reduction requires the umap-learn package. Install it with `pip install umap-learn`.")
        reducer = UMAP(n_components=2, n_neighbors=10, min_dist=1.0,
                       metric="cosine", init="pca", random_state=seed)
    else:
        raise ValueError("reduction must be one of 'pca', 'tsne', or 'umap'")

    Z2 = reducer.fit_transform(embeds)  # shape (N, 2)

    # Plot
    if categories and labels is not None:
        C = len(categories)
        ncols = 4
        nrows = math.ceil(C / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 squeeze=False)
        for idx, cat in enumerate(categories):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            mask = labels == idx

            # background points
            ax.scatter(Z2[~mask, 0], Z2[~mask, 1],
                       c="lightgray", s=5, alpha=0.2, rasterized=True)
            # highlighted category
            ax.scatter(Z2[mask, 0], Z2[mask, 1],
                       c=[plt.cm.tab20(idx % 20)], s=20, alpha=0.8,
                       edgecolor="k", linewidth=0.3, rasterized=True)
            ax.set_title(cat, fontsize="small")
            ax.set_xticks([]); ax.set_yticks([])

        # Turn off unused subplots
        for extra in range(C, nrows * ncols):
            r0, c0 = divmod(extra, ncols)
            axes[r0][c0].axis("off")
    else:
        plt.figure(figsize=(6, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=5, alpha=0.6)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{reduction.upper()} projection of embeddings')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Plot saved to {output_path}")