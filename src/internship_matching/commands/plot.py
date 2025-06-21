from datetime import datetime
import math
from pathlib import Path
import pickle
import re
from typing import Counter, Dict
import json
import click
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list

from ..data.match import apply_piecewise_power, get_distance_suggestions_pipeline
from .root import cli
from ..templates import PALETTE, THREEJS_TEMPLATE
from ..data.plot import plot_embeddings_latent_space
from ..utils import deprecated
from ..data.clusters import fetch_cluster_vectors
from ..data.job_autoencoder import JobAutoencoder
from ..constants import CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_AUTOENCODER_FILE_PATH, CVS_EMBEDDING_COLUMN_NAME, CVS_TRAIN_SEED, GLOBAL_RANDOM_SEED, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_AUTOENCODER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME, JOBS_TRAIN_SEED
from ..data.autolabeler import PT_STOPWORDS
from ..data.autoencoder import CVAutoencoder
from ..data.cvs      import fetch_embeddings_cv, fetch_embeddings_cv_with_courses, fetch_embeddings_cv_with_courses_filtered_with_experience, fetch_experiences_per_student, reset_database_embeddings_size_cv
from ..data.jobs     import fetch_embeddings_job_with_metadata
from ..data.db import POSTGRES_URL

# ─── PLOT GROUP ──────────────────────────────────────────────────────────────
@cli.group()
def plot():
    """Model plot commands."""
    pass

@plot.command("tsne-clusters")
@click.option(
    "--source",
    type=click.Choice(
        ["job-nomic", "cv-nomic", "job-autoencode", "cv-autoencode"],
        case_sensitive=False,
    ),
    required=True,
    help="Which clustering to visualise.",
)
@click.option("--perplexity", type=float, default=50.0, show_default=True)
@click.option("--threejs/--no-threejs", default=False, help="Generate interactive Three.js viewer.")
@click.option("--output-json", type=click.Path(path_type=Path), default=None, help="Custom JSON path.")
@click.option("--output-html", type=click.Path(path_type=Path), default=None, help="Custom HTML path.")
@click.option("--no-noise-filter", is_flag=True, help="Keep noise points (-1) in dataset (grey).")
def tsne_clusters_3d(
    source: str,
    perplexity: float,
    threejs: bool,
    output_json: Path | None,
    output_html: Path | None,
    no_noise_filter: bool,
):
    """Compute 3‑D t‑SNE and optionally output a Three.js interactive viewer."""

    # ------------------------------------------------------ load clusterer
    cl_paths = {
        "job-nomic": JOB_NOMIC_CLUSTER_FILE_PATH,
        "job-autoencode": JOB_CLUSTER_FILE_PATH,
        "cv-nomic": CV_NOMIC_CLUSTER_FILE_PATH,
        "cv-autoencode": CV_CLUSTER_FILE_PATH,
    }
    with open(cl_paths[source.lower()], "rb") as f:
        clusterer = pickle.load(f)
    labels = clusterer.labels_

    # ------------------------------------------------------ load embeddings
    if source.startswith("job"):
        df = fetch_embeddings_job_with_metadata()
        vec_col = JOBS_EMBEDDING_COLUMN_NAME if source == "job-nomic" else "latent_code"
    else:
        df = fetch_embeddings_cv_with_courses_filtered_with_experience()
        vec_col = CVS_EMBEDDING_COLUMN_NAME if source == "cv-nomic" else "latent_code"
    X = np.vstack(df[vec_col].values).astype(np.float32)

    if not no_noise_filter:
        keep = labels != -1
        labels = labels[keep]
        X = X[keep]

    # ------------------------------------------------------ t‑SNE 3‑D
    tsne = TSNE(n_components=3, perplexity=perplexity, init="pca", metric="cosine", random_state=GLOBAL_RANDOM_SEED)
    X3 = tsne.fit_transform(X)

    # ------------------------------------------------------ build clusters dict + hulls
    clusters: Dict[int, Dict[str, object]] = {}
    for cid in np.unique(labels):
        mask = labels == cid
        pts = X3[mask]
        color = "lightgrey" if cid == -1 else PALETTE[int(cid) % len(PALETTE)]
        entry: Dict[str, object] = {"color": color}
        if cid != -1 and pts.shape[0] >= 4:
            try:
                hull = ConvexHull(pts, qhull_options="QJ Pp")
                entry["vertices"] = pts.tolist()
                entry["faces"] = hull.simplices.tolist()
            except QhullError:
                pass  # skip degenerate clusters
        clusters[int(cid)] = entry

    # ------------------------------------------------------ JSON output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_fname = f"tsne_points_{source.replace('-', '_')}_{ts}.json"
    if output_json is None:
        output_json = Path("data/processed") / output_json_fname 
    output_json.parent.mkdir(parents=True, exist_ok=True)

    points_payload = [
        {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "label": int(lab)}
        for p, lab in zip(X3, labels)
    ]
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump({"points": points_payload, "clusters": clusters}, fp)

    click.echo(f"✅ JSON saved to {output_json}")

    # ------------------------------------------------------ Three.js viewer
    if threejs:
        if output_html is None:
            output_html = Path("data/processed") / f"tsne_viewer_{source.replace('-', '_')}_{ts}.html"
        output_html.parent.mkdir(parents=True, exist_ok=True)
        html = THREEJS_TEMPLATE.format(title=source, json_path=output_json_fname)
        with open(output_html, "w", encoding="utf-8") as fp:
            fp.write(html)
        click.echo(f"✅ HTML viewer saved to {output_html}")

    # ------------------------------------------------------ also dump naive 2‑D static plot for quick look
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X3[:, 0], X3[:, 1], s=6, c=[clusters[int(l)]["color"] for l in labels], alpha=0.6)
    ax.set_title(f"Quick 2‑D proj – {source}")
    ax.set_xlabel("dim1"); ax.set_ylabel("dim2")
    quick_png = output_json.with_suffix(".png")
    plt.tight_layout(); plt.savefig(quick_png, dpi=130); plt.close(fig)
    click.echo(f"(2‑D preview saved to {quick_png})")

# ─── (2a) UNIQUENESS SCORE PER ITEM ──────────────────────────────────────
@plot.command("cluster-uniqueness")
@click.option("--source", type=click.Choice(["cv", "cv-ae", "job", "job-ae"],
                                            case_sensitive=False),
              required=True, help="Which cluster table / embedding to use")
@click.option("--cluster-id", type=int, required=True)
@click.option("--k", default=10, show_default=True,
              help="k-NN count in uniqueness score")
@click.option("--output", default=None,
              help="PNG/PDF path; default into data/processed/")
def cluster_uniqueness(source, cluster_id, k, output):
    """
    Bar-plot of uniqueness score inside one cluster.

    Uniqueness = 1 − mean(cosine_sim to k nearest neighbours).
    """
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    labels, X = fetch_cluster_vectors(source.lower(), cluster_id)
    N, _ = X.shape
    k = min(k, N-1)

    sims = cosine_similarity(X)
    # for each row, take k largest sims (excluding self on diag)
    uniq = []
    for i in range(N):
        s = np.partition(sims[i], -k-1)[-k-1:-1]  # fastest k (skip self)
        uniq.append(1.0 - s.mean())
    uniq = np.array(uniq)

    order = uniq.argsort()[::-1]           # most unique on top
    labels_ord, uniq_ord = np.array(labels)[order], uniq[order]

    plt.figure(figsize=(8, max(4, 0.25*N)))
    plt.barh(range(N), uniq_ord, color="tab:blue")
    plt.yticks(range(N), labels_ord, fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Uniqueness (k={k})  ↑ more unique")
    plt.title(f"Cluster {cluster_id} – {source.upper()} ({N} items)")
    plt.tight_layout()

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("data/processed") / f"uniq_{source}_{cluster_id}_{ts}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    click.echo(f"✅ Uniqueness plot saved to {output}")

# ─── (2b) INTRA-CLUSTER HEAT-MAP ─────────────────────────────────────────
@plot.command("cluster-heatmap")
@click.option("--source", type=click.Choice(["cv", "cv-ae", "job", "job-ae"],
                                            case_sensitive=False),
              required=True)
@click.option("--cluster-id", type=int, required=True)
@click.option("--sample", default=60, show_default=True,
              help="If cluster is larger, random-sample this many items "
                   "(-1 = take all)")
@click.option("--interpolate", default="nearest", show_default=True,
              help='Value for matplotlib.imshow "interpolation" argument; '
                   'e.g. "nearest", "none", "bilinear"')                   
@click.option("--output", default=None)
def cluster_heatmap(source, cluster_id, sample, interpolate, output):
    """
    K×K cosine-similarity heat-map of one cluster (hierarchically ordered).
    """
    from pathlib import Path
    from datetime import datetime

    labels, X = fetch_cluster_vectors(source.lower(), cluster_id)
    N = X.shape[0]
    if sample != -1 and N > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, sample, replace=False)
        X, labels = X[idx], [labels[i] for i in idx]
        N = sample

    sims = cosine_similarity(X)

    # hierarchical ordering
    Z      = linkage(sims, method="average")
    leaves = leaves_list(Z)
    sims   = sims[leaves][:, leaves]
    labels = [labels[i] for i in leaves]

    plt.figure(figsize=(0.25*N + 2, 0.25*N + 2))
    im = plt.imshow(sims, cmap="viridis", vmin=0.0, vmax=1.0, interpolation=interpolate)
    plt.xticks(range(N), labels, rotation=90, fontsize=6)
    plt.yticks(range(N), labels, fontsize=6)
    plt.colorbar(im, fraction=0.046, pad=0.04).set_label("Cosine similarity")
    plt.title(f"Cluster {cluster_id} – {source.upper()}  (N={N})")
    plt.tight_layout()

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("data/processed") / f"heat_{source}_{cluster_id}_{ts}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    click.echo(f"✅ Heat-map saved to {output}")

@plot.command("compare-heatmap")
@click.option(
    "--mode",
    type=click.Choice(
        ["cv-to-cv", "cv-to-job", "job-to-job", "job-to-cv"],
        case_sensitive=False
    ),
    required=True,
    help="""
    cv-to-cv   : highlight 1 CV against its nearest CV neighbours
    cv-to-job  : compare 1 CV against job-cluster centroids
    job-to-job : highlight 1 job against its nearest job neighbours
    job-to-cv  : compare 1 job against CV-cluster centroids
    """
)
@click.option("--id", "item_key", required=True,
              help="(fonte_aluno,matricula) for CV or contract_id for Job")
@click.option("--topk", default=30, show_default=True,
              help="How many neighbours / clusters to show.")
@click.option("--output", default=None,
              help="Path to save PNG (defaults to data/processed/)")
def compare_heatmap(mode, item_key, topk, output):
    """
    Visualise similarity of a single item to peers or centroids.

    Examples
    --------
    # one CV vs 30 nearest CVs
    internship plot compare-heatmap --mode cv-to-cv --id VRADM,0116256

    # that same CV against all job cluster centroids (top 30)
    internship plot compare-heatmap --mode cv-to-job --id VRADM,0116256

    # one job vs jobs in its own cluster
    internship plot compare-heatmap --mode job-to-job --id 181012
    """
    import matplotlib.pyplot as plt, numpy as np, json, psycopg2, os
    from sklearn.metrics.pairwise import cosine_similarity
    from datetime import datetime
    from pathlib import Path

    def pg_vec(v):
        return np.array(json.loads(v), dtype=np.float32) if isinstance(v, str) else np.array(v, dtype=np.float32)

    # ── extract query vector ────────────────────────────────────────────
    conn = psycopg2.connect(POSTGRES_URL if isinstance(POSTGRES_URL, str) else POSTGRES_URL)
    cur  = conn.cursor()

    if mode.startswith("cv"):
        fonte, matric = map(str.strip, item_key.split(","))
        cur.execute("SELECT embedding FROM cv_embeddings WHERE fonte_aluno=%s AND matricula=%s",
                    (fonte, matric))
    else:
        contract_id = int(item_key)
        cur.execute("SELECT embedding FROM job_embeddings WHERE contract_id=%s",
                    (contract_id,))
    row = cur.fetchone()
    if not row:
        click.echo("❌ Item not found.")
        return
    qvec = pg_vec(row[0]).reshape(1, -1)

    # ── prepare comparison matrix ───────────────────────────────────────
    if mode == "cv-to-cv":
        # pull nearest CV neighbours by cosine similarity
        cur.execute("""
            SELECT fonte_aluno, matricula, embedding
            FROM   cv_embeddings
            """)  # naive; for >50k CVs use ANN instead
        peers = cur.fetchall()
        mats  = np.stack([pg_vec(r[2]) for r in peers])
        sims  = cosine_similarity(qvec, mats).flatten()
        top   = sims.argsort()[::-1][1:topk+1]  # skip self
        labels = [f"{peers[i][0]},{peers[i][1]}" for i in top]
        sims   = sims[top]

    elif mode == "job-to-job":
        cur.execute("SELECT contract_id, embedding FROM job_embeddings")
        peers = cur.fetchall()
        mats  = np.stack([pg_vec(r[1]) for r in peers])
        sims  = cosine_similarity(qvec, mats).flatten()
        top   = sims.argsort()[::-1][1:topk+1]
        labels = [str(peers[i][0]) for i in top]
        sims   = sims[top]

    elif mode == "cv-to-job":
        cur.execute("SELECT cluster_id, centroid FROM job_nomic_centroids")
        rows = cur.fetchall()
        mats = np.stack([pg_vec(r[1]) for r in rows])
        sims = cosine_similarity(qvec, mats).flatten()
        top  = sims.argsort()[::-1][:topk]
        labels = [f"jobC{rows[i][0]}" for i in top]
        sims   = sims[top]

    else:  # job-to-cv
        cur.execute("SELECT cluster_id, centroid FROM cv_nomic_centroids")
        rows = cur.fetchall()
        mats = np.stack([pg_vec(r[1]) for r in rows])
        sims = cosine_similarity(qvec, mats).flatten()
        top  = sims.argsort()[::-1][:topk]
        labels = [f"cvC{rows[i][0]}" for i in top]
        sims   = sims[top]

    cur.close(); conn.close()

    # ── plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 0.4*len(sims)))
    im = ax.imshow(sims.reshape(1, -1), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks([]); ax.set_xlabel("Similarity →")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
    cbar.set_label("Cosine similarity")

    title = f"{mode.upper()} – top {len(sims)}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"compare_heatmap_{mode}_{ts}.png"
        output = Path("data/processed") / fname
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    click.echo(f"✅ Heat-map saved to {output}")

# ─── CLUSTER‑DENSITY HISTOGRAM ──────────────────────────────────────────────
@plot.command("cluster-density-hist")
@click.option(
    "--source",
    type=click.Choice(
        ["job", "cv", "job-nomic", "cv-nomic"],
        case_sensitive=False
    ),
    required=True,
    help="Which cluster‑assignment table to plot:\n"
         "  job        -> job_clusters\n"
         "  cv         -> cv_clusters\n"
         "  job-nomic  -> job_nomic_clusters\n"
         "  cv-nomic   -> cv_nomic_clusters",
)
@click.option(
    "--exclude-noise/--include-noise",
    default=True,
    show_default=True,
    help="Exclude HDBSCAN noise points (cluster_id = -1) from the histogram."
)
@click.option(
    "--bins",
    default="auto",
    help="Matplotlib bin specification (e.g. 50, 'auto', 'sturges', …)."
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Optional path to save the figure.  "
         "If omitted, a timestamped PNG goes to data/processed/."
)
@click.option("--logx/--linearx", default=False, help="Log-scale X axis")
@click.option("--logy/--lineary", default=False, help="Log-scale Y axis")
@click.option(
    "--clip-top",
    type=int,
    default=0,
    help="Remove the N largest clusters before plotting"
)
@click.option(
    "--cdf", "plot_cdf",
    is_flag=True,
    help="Plot cumulative distribution instead of histogram"
)
@click.option(
    "--annotate-top",
    type=int,
    default=0,
    help="Annotate the K largest clusters on the plot"
)
def plot_cluster_density_hist(source, exclude_noise, bins, output_path, logx, logy, clip_top, plot_cdf, annotate_top):
    """
    Histogram of cluster sizes (#items per cluster).

    Examples
    --------
    internship plot cluster-density-hist --source job-nomic
    internship plot cluster-density-hist --source cv --bins 40
    internship plot cluster-density-hist --source cv-nomic --include-noise
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import psycopg2
    from datetime import datetime
    from pathlib import Path

    # ── 1) Map human flag → table name & nice title ───────────────────────
    src_map = {
        "job":       ("job_clusters",       "Job (latent‑AE)"),
        "cv":        ("cv_clusters",        "CV (latent‑AE)"),
        "job-nomic": ("job_nomic_clusters", "Job (Nomic)"),
        "cv-nomic":  ("cv_nomic_clusters",  "CV (Nomic)"),
    }
    table, pretty = src_map[source.lower()]

    # ── 2) Query Postgres for cluster sizes ───────────────────────────────
    conn = psycopg2.connect(POSTGRES_URL) if isinstance(POSTGRES_URL, str) \
           else psycopg2.connect(**POSTGRES_URL)

    q = f"""
        SELECT cluster_id, COUNT(*) AS size
        FROM {table}
        GROUP BY cluster_id
    """
    df = pd.read_sql(q, conn)
    conn.close()

    if exclude_noise:
        df = df[df.cluster_id != -1]

    if df.empty:
        click.echo("⚠️  No clusters to plot after filtering; aborting.")
        return

    sizes = df["size"].to_numpy()

    # ── preprocessing: clip top outliers if requested ──────────────
    df = df.sort_values("size", ascending=False)
    if clip_top > 0:
        df = df.iloc[clip_top:]

    sizes = df["size"].to_numpy()
    if sizes.size == 0:
        click.echo("⚠️ Nothing left to plot after clipping; aborting.")
        return

    # ── plotting ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(7, 4))
    if plot_cdf:
        sorted_sizes = np.sort(sizes)
        cdf = np.arange(1, len(sorted_sizes)+1) / len(sorted_sizes)
        plt.step(sorted_sizes, cdf)
        plt.ylabel("Cumulative fraction of clusters")
    else:
        # parse fancy bin spec start:stop:step
        if isinstance(bins, str) and ":" in bins:
            start, stop, step = map(int, bins.split(":"))
            bins = np.arange(start, stop + step, step)
        plt.hist(sizes, bins=bins, edgecolor="black")
        plt.ylabel("Number of clusters")

    plt.xlabel("Cluster size (# items)")
    plt.title(f"Cluster‑Size Distribution – {pretty}")

    # log scales
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    # annotate top clusters
    if annotate_top > 0 and not plot_cdf:
        top = df.head(annotate_top)
        for _, row in top.iterrows():
            plt.annotate(f"{int(row.size)}",
                         xy=(row.size, 0),
                         xytext=(0, 6),
                         textcoords="offset points",
                         ha="center",
                         fontsize=8,
                         rotation=90)

    plt.tight_layout()

    # ── 4) Determine output path & save ───────────────────────────────────
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"cluster_density_{source}_{ts}.png"
        output_path = Path("data/processed") / fname
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150)
    plt.close()

    # ── 5) Console summary ────────────────────────────────────────────────
    click.echo(f"✅ Histogram saved to {output_path}")
    click.echo(f"   Clusters plotted : {len(sizes):,}")
    click.echo(f"   Min / Median / Max size : "
               f"{sizes.min():,} / {np.median(sizes):,.0f} / {sizes.max():,}")

@plot.command("cv-autoencoder")
@deprecated("Esse comando é antigo. O relatório até funciona, mas foram desenvolvidos relatórios melhores.")
def plot_cv_autoencoder():
    model = CVAutoencoder()
    embeds = fetch_embeddings_cv()
    recons = model.get_reconstructions(embeds)
    errors = np.abs(recons - embeds).mean(axis=1)
    CVAutoencoder.plot_latent_space(model, embeds, "pca", labels=errors)
    CVAutoencoder.plot_latent_space(model, embeds, "tsne", labels=errors)
    CVAutoencoder.plot_latent_space(model, embeds, "umap", labels=errors)

@plot.command("job-autoencoder-faceted-highlight-principal")
def job_autoencoder_by_principal_facet():
    """Facet latent-space by atividade_principal"""
    embedding_col = JOBS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching job metadata and embeddings...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df)} records.")

    topN = 40

    top_principals = df['atividade_principal_code'].value_counts().nlargest(topN).index.tolist()
    df = df[df['atividade_principal_code'].isin(top_principals)]

    click.echo(f"Filtered to {len(df)} records.")

    labels, principals = pd.factorize(df['atividade_principal_code'])
    X = np.vstack(df[embedding_col].values).astype(np.float32)

    click.echo("Loading JobAutoencoder model...")
    model = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)
    model.eval()
    click.echo("Finished loading JobAutoencoder")

    for red in ("tsne", "umap", "pca"):
        output_path = f"data/processed/job_latent_space_by_principal_facet_{red}_{embedding_col}.pdf"
        click.echo(f"Generating latent space facet plot for principal '{red}'...")
        JobAutoencoder.plot_latent_space_with_categories_facet(
            model,
            X,
            reduction=red,
            labels=labels,
            categories=list(principals),
            seed=JOBS_TRAIN_SEED,
            output_path=output_path
        )
        click.echo(f"Saved plot to {output_path}")

@plot.command("job-autoencoder-faceted-highlight-courses")
def job_autoencoder_by_course_facet():
    """Facet latent-space by course_name"""
    embedding_col = JOBS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching job metadata and embeddings...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df)} records.")

    topN = 80

    top_principals = df['course_name'].value_counts().nlargest(topN).index.tolist()
    df = df[df['course_name'].isin(top_principals)]

    click.echo(f"Filtered to {len(df)} records.")

    labels, courses = pd.factorize(df['course_name'])
    X = np.vstack(df[embedding_col].values).astype(np.float32)

    click.echo("Loading JobAutoencoder model...")
    model = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)
    model.eval()
    click.echo("Finished loading JobAutoencoder")

    for red in ("pca", "tsne", "umap"):
        output_path = f"data/processed/job_latent_space_by_course_facet_{red}_{embedding_col}.pdf"
        click.echo(f"Generating latent space facet plot for courses '{red}'...")
        JobAutoencoder.plot_latent_space_with_categories_facet(
            model,
            X,
            reduction=red,
            labels=labels,
            categories=list(courses),
            seed=JOBS_TRAIN_SEED,
            output_path=output_path
        )
        click.echo(f"Saved plot to {output_path}")

@plot.command("cv-autoencoder-faceted-highlight-experiences")
def cv_autoencoder_by_course_facet_experiences():
    embedding_col = "embedding"  # or "raw_embedding"

    # 1) Get your joined DataFrame:
    df = fetch_experiences_per_student(embedding_col)

    # 2) Factorize tem_experiencia → integer labels + list of names
    labels, tem_experiencia = pd.factorize(df["tem_experiencia"])

    # 3) Stack embeddings (N, D)
    X = np.vstack(df[embedding_col].values).astype(np.float32)

    # 4) Load your trained AE
    model = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
    model.eval()

    # 5) One‐line per reduction:
    for red in ("pca", "tsne", "umap"):
        CVAutoencoder.plot_latent_space_with_categories_facet(
            model,
            X,
            reduction=red,
            labels=labels,
            categories=list(tem_experiencia),
            seed=CVS_TRAIN_SEED,
            output_path=f"data/processed/cv_latent_space_by_experience_facet_{red}_{embedding_col}.pdf"
        )

@plot.command("cv-autoencoder-faceted-highlight-courses")
def cv_autoencoder_by_course_facet_courses():
    embedding_col = "embedding"  # or "raw_embedding"

    # 1) Get your joined DataFrame:
    df = fetch_embeddings_cv_with_courses(embedding_col)

    # 2) Factorize course_name → integer labels + list of names
    labels, course_names = pd.factorize(df["course_name"])

    # 3) Stack embeddings (N, D)
    X = np.vstack(df[embedding_col].values).astype(np.float32)

    # 4) Load your trained AE
    model = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
    model.eval()

    # 5) One‐line per reduction:
    for red in ("pca", "tsne", "umap"):
        CVAutoencoder.plot_latent_space_with_categories_facet(
            model,
            X,
            reduction=red,
            labels=labels,
            categories=list(course_names),
            seed=CVS_TRAIN_SEED,
            output_path=f"data/processed/cv_latent_space_by_course_facet_{red}_{embedding_col}.pdf"
        )

@plot.command("cv-autoencoder-by-course")
def cv_autoencoder_by_course():

    embedding_col = "embedding"  # or "raw_embedding"

    # 1) Get your joined DataFrame:
    df = fetch_embeddings_cv_with_courses(embedding_col)  

    # 2) Factorize course_name → integer labels + list of names
    labels, course_names = pd.factorize(df["course_name"])

    # 3) Stack embeddings (N, D)
    X = np.vstack(df[embedding_col].values)

    # 4) Load your trained AE
    model = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)

    # 5) One‐line per reduction:
    for red in ("pca","tsne","umap"):
        CVAutoencoder.plot_latent_space_with_categories(
            model,
            X,
            reduction=red,
            labels=labels,
            categories=list(course_names),
            seed=CVS_TRAIN_SEED,
            output_path=f"data/processed/cv_latent_space_by_course_{red}_{embedding_col}.png"
        )

@plot.command("job-nomic-principal-facet")
def job_nomic_by_principal_facet():
    """Facet Nomic embeddings by atividade_principal"""
    embedding_col = JOBS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching job metadata and embeddings...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df)} records.")

    topN = 40
    top_principals = df['atividade_principal_code'].value_counts().nlargest(topN).index.tolist()
    df = df[df['atividade_principal_code'].isin(top_principals)]
    click.echo(f"Filtered to {len(df)} records.")

    labels, principals = pd.factorize(df['atividade_principal_code'])

    for red in ("pca", "tsne", "umap"):
        output_path = (
            f"data/processed/job_nomic_latent_space_by_principal_{red}_{embedding_col}.pdf"
        )
        click.echo(f"Generating {red.upper()} plot for principals...")
        plot_embeddings_latent_space(
            df,
            embed_col=embedding_col,
            reduction=red,
            labels=labels,
            categories=list(principals),
            output_path=output_path,
            show=False
        )
        click.echo(f"Saved plot to {output_path}")

@plot.command("job-nomic-course-facet")
def job_nomic_by_course_facet():
    """Facet Nomic embeddings by course_name"""
    embedding_col = JOBS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching job metadata and embeddings...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df)} records.")

    topN = 80
    top_courses = df['course_name'].value_counts().nlargest(topN).index.tolist()
    df = df[df['course_name'].isin(top_courses)]
    click.echo(f"Filtered to {len(df)} records.")

    labels, courses = pd.factorize(df['course_name'])

    for red in ("pca", "tsne", "umap"):
        output_path = (
            f"data/processed/job_nomic_latent_space_by_course_{red}_{embedding_col}.pdf"
        )
        click.echo(f"Generating {red.upper()} plot for courses...")
        plot_embeddings_latent_space(
            df,
            embed_col=embedding_col,
            reduction=red,
            labels=labels,
            categories=list(courses),
            output_path=output_path,
            show=False
        )
        click.echo(f"Saved plot to {output_path}")

@plot.command("cv-nomic-experience-facet")
def cv_nomic_by_experience_facet():
    """Facet Nomic embeddings by tem_experiencia"""
    embedding_col = CVS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching CV experience embeddings...")
    df = fetch_experiences_per_student(embedding_col)
    click.echo(f"Fetched {len(df)} records.")

    labels, experiences = pd.factorize(df['tem_experiencia'])

    for red in ("pca", "tsne", "umap"):
        output_path = (
            f"data/processed/cv_nomic_latent_space_by_experience_{red}_{embedding_col}.pdf"
        )
        click.echo(f"Generating {red.upper()} plot for experiences...")
        plot_embeddings_latent_space(
            df,
            embed_col=embedding_col,
            reduction=red,
            labels=labels,
            categories=list(experiences),
            output_path=output_path,
            show=False
        )
        click.echo(f"Saved plot to {output_path}")

@plot.command("cv-nomic-course-facet")
def cv_nomic_by_course_facet():
    """Facet Nomic embeddings by course_name"""
    embedding_col = CVS_EMBEDDING_COLUMN_NAME
    click.echo("Fetching CV course embeddings...")
    df = fetch_embeddings_cv_with_courses(embedding_col)
    click.echo(f"Fetched {len(df)} records.")

    labels, course_names = pd.factorize(df['course_name'])

    for red in ("pca", "tsne", "umap"):
        output_path = (
            f"data/processed/cv_nomic_latent_space_by_course_{red}_{embedding_col}.pdf"
        )
        click.echo(f"Generating {red.upper()} plot for CV courses...")
        plot_embeddings_latent_space(
            df,
            embed_col=embedding_col,
            reduction=red,
            labels=labels,
            categories=list(course_names),
            output_path=output_path,
            show=False
        )
        click.echo(f"Saved plot to {output_path}")

@deprecated("Esse código é antigo e não deve ser usado. Foi substituído por outros comandos de plot.")
@plot.command("comparative-cv-autoencoder")
@click.option("--seed", default=f"{CVS_TRAIN_SEED}", show_default=True)
def plot_comparative_cv_autoencoder(seed : str):
    parsed_seed = int(seed)
    click.echo(f"🔑 Using seed {seed}.")
    embedding_columns = [ "embedding", "raw_embedding" ]
    for col in embedding_columns:
        auto_encoder_loss_types = [ "hybrid", "cosine", "mse" ]
        for loss_type in auto_encoder_loss_types:
            latent_sizes = [ 16, 32, 64, 128, 256 ]
            embeds = fetch_embeddings_cv(embedding_column=col)
            for latent_size in latent_sizes:
                reset_database_embeddings_size_cv(latent_size=latent_size)
                CVAutoencoder.train_from_db(latent_dim=latent_size, train_seed=parsed_seed, loss_type=loss_type)
                CVAutoencoder.generate_all_latents(latent_dim=latent_size)
                model = CVAutoencoder(latent_dim=latent_size)
                recons = model.get_reconstructions(embeds)
                if loss_type == "mse":
                    # Mean squared error per sample
                    errors = ((recons - embeds) ** 2).mean(axis=1)
                elif loss_type == "cosine":
                    # Cosine distance per sample
                    sims = cosine_similarity(recons, embeds)
                    errors = 1 - np.diag(sims)
                else:  # hybrid
                    mse_err = ((recons - embeds) ** 2).mean(axis=1)
                    sims = cosine_similarity(recons, embeds)
                    cos_err = 1 - np.diag(sims)
                    errors = mse_err + cos_err
                reductions = [ "pca", "tsne", "umap" ]
                for reduction in reductions:
                    CVAutoencoder.plot_latent_space(model, embeds, reduction, labels=errors, latent_size=latent_size, embedding_column=col, seed=parsed_seed, loss_type=loss_type)
                    model.save_round_trip_report(embeds, f"data/processed/cv_autoencoder_roundtrip_{reduction}_{latent_size}_{col}_{parsed_seed}_{loss_type}.json")

# ─── Shared gauge configuration ──────────────────────────────────────────
GAUGE_THRESHOLDS = [0.33, 0.66]     # boundaries in [0,1]
GAUGE_COLORS     = ["red", "yellow", "green"]

# ─── helper from your plot module ─────────────────────────────────────────
def draw_linear_gauge(ax, score: float, title: str = ""):
    """
    Draw a horizontal 0–1 gauge:
      - gradient background from red→yellow→green
      - black border
      - a vertical arrow at `score`
    """
    # 1) gradient background
    cmap = LinearSegmentedColormap.from_list("gauge", GAUGE_COLORS)
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    ax.imshow(
        grad,
        aspect="auto",
        cmap=cmap,
        extent=(0, 1, 0, 1),
        origin="lower"
    )

    # 2) border
    ax.add_patch(Rectangle((0, 0), 1, 1,
                           fill=False,
                           edgecolor="black",
                           lw=2))

    # 3) arrow marker
    # annotate from just below the bar up into the bar
    ax.annotate(
        "", 
        xy=(score, 0.7),      # arrow tip
        xytext=(score, -0.1), # arrow tail
        arrowprops=dict(
            arrowstyle="-|>",  # simple arrow
            lw=2,
            color="black"
        ),
        annotation_clip=False
    )

    # 4) clean up axes
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1);    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\n{int(round(score*100))}%", pad=8)

@plot.command("cv-job-gauge")
@click.option("-c", "--contract-id", required=True, type=int,
              help="Job contract ID to compare against.")
@click.option("--cv-text", help="CV text directly as input.")
@click.option("--cv-file", type=click.Path(exists=True),
              help="Path to a .txt file with CV text.")
@click.option("--top-k", default=10, show_default=True,
              help="Number of tokens for gap analysis.")
def distance_line_gauge(contract_id, cv_text, cv_file, top_k):
    """
    Plot before/after match as simple horizontal line gauges.
    """
    # 1) run your pipeline
    result = get_distance_suggestions_pipeline(
        contract_id,
        cv_text=cv_text,
        cv_file=Path(cv_file) if cv_file else None,
        embedding_type="NOMIC",
        top_k=top_k,
    )

    # 2) extract the piecewise‐powered scores
    init_raw   = result["initial_norm_similarity"]
    init_scaled= apply_piecewise_power(init_raw)
    imp_raw    = result["improved_norm_similarity"]
    imp_scaled = apply_piecewise_power(imp_raw)

    # 3) draw two gauges side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5))
    draw_linear_gauge(ax1, init_scaled, title="Initial Match")
    draw_linear_gauge(ax2, imp_scaled,  title="Improved Match")
    fig.tight_layout()

    # 4) save to disk
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("data/processed") / f"line_gauge_{contract_id}_{ts}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close(fig)

    click.echo(f"✅ Line‐gauge saved to {out}")