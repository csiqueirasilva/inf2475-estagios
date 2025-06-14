import random
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Counter, Optional, List
from sklearn.feature_extraction.text import CountVectorizer

from internship_matching.data.autolabeler import PT_STOPWORDS

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

word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)

def generate_cluster_names_manual(
    df,
    cluster_col="hdbscan",
    text_col="text",
    top_n=10
):
    # helper to reject any token with digits
    def is_non_numeric(token):
        return not re.search(r"\d", token)

    # 1) Document frequencies (DF)
    total_docs = len(df)
    df_counts = Counter()
    for doc in df[text_col].astype(str):
        tokens = {
            w.lower()
            for w in word_re.findall(doc)
            if w.lower() not in PT_STOPWORDS
            and is_non_numeric(w)
        }
        df_counts.update(tokens)

    # 2) Per‐cluster TF & TF–IDF scoring
    cluster_names = {}
    for cid in sorted(df[cluster_col].unique()):
        if cid == -1:
            cluster_names[cid] = "Noise"
            continue

        texts = df.loc[df[cluster_col] == cid, text_col].astype(str)
        if texts.empty:
            cluster_names[cid] = "Unknown"
            continue

        # term freqs (TF) in this cluster
        tf_counts = Counter()
        for doc in texts:
            tokens = [
                w.lower()
                for w in word_re.findall(doc)
                if w.lower() not in PT_STOPWORDS
                and is_non_numeric(w)
            ]
            tf_counts.update(tokens)

        # compute TF–IDF–style score
        scores = {}
        for term, count in tf_counts.items():
            # DF lookup from global df_counts
            df_count = df_counts.get(term, 0)
            idf = math.log((total_docs) / (1 + df_count))
            scores[term] = (count / len(texts)) * idf

        # pick top_n and join
        top_terms = sorted(scores, key=scores.get, reverse=True)[:top_n]
        cluster_names[cid] = " / ".join(top_terms) if top_terms else "—"

    return cluster_names

def generate_cluster_names(
    df: pd.DataFrame,
    cluster_col: str = "hdbscan_umap",
    name_col:    str = "course_name",
    top_n:       int = 3
) -> dict[int, str]:
    """
    For each cluster ID (excluding noise = -1), find the top_n most
    common words/phrases in the course_name texts, and join them as a label.
    """
    vectorizer = CountVectorizer(
        stop_words=PT_STOPWORDS,
        ngram_range=(1,2),    # allow unigrams & bigrams
        max_features=top_n
    )
    labels = {}
    for cluster in sorted(df[cluster_col].unique()):
        if cluster == -1:
            labels[cluster] = "Noise"
            continue
        texts = df.loc[df[cluster_col] == cluster, name_col].astype(str)
        if texts.empty:
            labels[cluster] = "Unknown"
            continue
        X = vectorizer.fit_transform(texts)
        # get the top words by frequency
        features = vectorizer.get_feature_names_out()
        labels[cluster] = " / ".join(features)
    return labels