from datetime import datetime
import pickle
import json
import click
import hdbscan
import numpy as np
from sklearn.preprocessing import normalize

from ..data.plot import generate_cluster_names_manual
from .root import cli
from ..utils import deprecated
from ..data.clusters import store_clusters_and_centroids, sweep_hdbscan
from ..constants import CLUSTERER_PATHS, CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_EMBEDDING_COLUMN_NAME, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME
from ..data.cvs      import fetch_embeddings_cv_with_courses_filtered_with_experience
from ..data.jobs     import fetch_embeddings_job_with_metadata
from ..data.db import POSTGRES_URL

@cli.group()
def cluster():
    """Clustering commands."""
    pass

def _persist_clusters(
    df,
    latent_col: str,
    key_cols: list[str],
    clusters_table: str,
    centroids_table: str,
    cluster_file_path: str,
    params: dict,
    description: str
) -> None:
    """
    Generic helper to fit HDBSCAN, persist assignments, centroids, and pickle the clusterer.

    :param df: DataFrame containing latent codes in column `latent_col`.
    :param latent_col: Name of the column in df with latent_code vectors.
    :param key_cols: List of columns to use as primary key for assignments table.
    :param clusters_table: Postgres table name for cluster assignments.
    :param centroids_table: Postgres table name for centroids.
    :param cluster_file_path: Filesystem path to pickle the HDBSCAN clusterer.
    :param params: HDBSCAN parameters dict.
    :param description: String to echo at start (e.g., "CV" or "job").
    """
    click.echo(f"Loading {description} latent codes: {len(df):,} records.")

    # prepare matrix
    # ensure double precision for HDBSCAN
    X = np.vstack(df[latent_col].values).astype(np.float64)
    # normalize still yields float64
    X_norm = normalize(X, norm="l2")

    # fit
    click.echo(f"Training {description} HDBSCAN clusterer...")
    clusterer = hdbscan.HDBSCAN(**params)
    clusterer.fit(X_norm)

    # verify prediction data
    if getattr(clusterer, "prediction_data_", None) is None:
        raise RuntimeError(
            "HDBSCAN did not produce prediction_data_. "
            "Make sure prediction_data=True and metric is compatible."
        )

    # assign labels
    df["hdb_best"] = clusterer.labels_

    # persist to Postgres
    store_clusters_and_centroids(
        df=df,
        cluster_col="hdb_best",
        key_cols=key_cols,
        clusters_table=clusters_table,
        centroids_table=centroids_table,
        latent_col=latent_col
    )
    click.echo(f"Persisted {description} clusters to database ({clusters_table}, {centroids_table}).")

    # pickle clusterer
    with open(cluster_file_path, "wb") as fp:
        pickle.dump(clusterer, fp)
    click.echo(f"Saved {description} HDBSCAN clusterer to {cluster_file_path}")

# ---------------------------------------------------------------------------
# Generic label inspection – pre‑trained clusterers (no sweep)
# ---------------------------------------------------------------------------

@cluster.command("labels")
@click.option(
    "--source",
    type=click.Choice(
        ["job-nomic", "job-autoencode", "cv-nomic", "cv-autoencode"],
        case_sensitive=False,
    ),
    required=True,
    help="Which clustering model to load and inspect.",
)
def cluster_labels(source: str) -> None:
    """Load an existing HDBSCAN clusterer and print TF‑IDF labels for its clusters."""

    click.echo(f"Loading pre‑trained clusterer for {source}…")
    model_path = CLUSTERER_PATHS[source.lower()]
    with open(model_path, "rb") as f:
        clusterer = pickle.load(f)
    labels = clusterer.labels_

    # ------------------------------------------------------------------ fetch embeddings & text
    if source.startswith("job"):
        df = fetch_embeddings_job_with_metadata()
        text_col = "raw_input"
        vec_col = (
            JOBS_EMBEDDING_COLUMN_NAME if source == "job-nomic" else "latent_code"
        )
    else:  # cv
        df = fetch_embeddings_cv_with_courses_filtered_with_experience()
        text_col = "text"
        vec_col = (
            CVS_EMBEDDING_COLUMN_NAME if source == "cv-nomic" else "latent_code"
        )

    if len(df) != len(labels):
        raise click.ClickException(
            "Mismatch: number of embeddings fetched does not match clusterer.labels_."
        )

    df["hdb_best"] = labels

    # ------------------------------------------------------------------ generate names
    cluster_names = generate_cluster_names_manual(
        df, cluster_col="hdb_best", text_col=text_col
    )

    click.echo("\nCluster summaries ({0}):".format(source))
    for cid, label in cluster_names.items():
        click.echo(f"  Cluster {cid:3d}: {label}")

    click.echo("Done.")


# ─── CLUSTER QA / SUMMARY ─────────────────────────────────────────────
@cluster.command("stats")
@click.option(
    "--source",
    type=click.Choice(
        ["job", "cv", "job-nomic", "cv-nomic"],
        case_sensitive=False
    ),
    required=True,
    help="Which cluster table to analyse."
)
@click.option("--min-size", default=1, show_default=True,
              help="Discard clusters smaller than this.  Use 1 to keep all.")
@click.option("--top", default=None,
              help="Show only the TOP largest clusters in the console. "
                   "CSV export always contains everything that passed the filter.")
@click.option("--sample", default=300,
              help="Max items per cluster when computing similarity (-1 = no cap).")
@click.option("--pbar/--no-pbar", default=False,
              help="Show tqdm progress bar while crunching clusters.")
@click.option("--stats-only", is_flag=True,
              help="Skip per‑cluster table and print only corpus‑level aggregates.")
@click.option("--export-csv", type=click.Path(), default=None,
              help="Optional path to write the full stats CSV.")
def cluster_stats(source, min_size, top, sample, pbar, stats_only, export_csv):
    """
    Quality metrics for ALL clusters (median, min, p25, p75 similarity).

    Examples
    --------
    # Full corpus → CSV, no console spam
    internship cluster stats --source job-nomic --export-csv data/cluster_stats.csv --stats-only

    # Show biggest 50 clusters and progress bar
    internship cluster stats --source cv-nomic --top 50 --pbar
    """
    import pandas as pd, numpy as np, psycopg2, textwrap, sys
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    src_map = {
        "job":       ("job_clusters",       "job_embeddings",       "latent_code", "raw_input"),
        "job-nomic": ("job_nomic_clusters", "job_embeddings",       "embedding",   "raw_input"),
        "cv":        ("cv_clusters",        "cv_embeddings",        "latent_code", "llm_parsed_raw_input"),
        "cv-nomic":  ("cv_nomic_clusters",  "cv_embeddings",        "embedding",   "llm_parsed_raw_input"),
    }
    clust_tbl, emb_tbl, vec_col, text_col = src_map[source.lower()]

    # --- build JOIN query --------------------------------------------------
    on_cols = ["fonte_aluno", "matricula"] + (["contract_id"] if "job" in source else [])
    on_cond = " AND ".join([f"c.{c}=e.{c}" for c in on_cols])
    sql = f"""
        SELECT c.cluster_id,
               e.{vec_col}  AS vec,
               e.{text_col} AS txt
        FROM   {clust_tbl} c
        JOIN   {emb_tbl}   e
        ON     {on_cond}
        WHERE  c.cluster_id != -1
    """

    conn = psycopg2.connect(POSTGRES_URL) if isinstance(POSTGRES_URL, str) \
           else psycopg2.connect(**POSTGRES_URL)
    df = pd.read_sql(sql, conn)
    conn.close()

    df["vec"] = df["vec"].apply(lambda v: np.array(json.loads(v), dtype=np.float32))

    # --- crunch per‑cluster ------------------------------------------------
    clusters = df.groupby("cluster_id")
    iterator = tqdm(clusters, desc="Clusters") if pbar else clusters

    rows = []
    for cid, grp in iterator:
        sz = len(grp)
        if sz < min_size:
            continue

        # sampling
        take = grp if sample == -1 or sz <= sample else grp.sample(sample, random_state=42)
        vecs = np.stack(take["vec"])
        if vecs.shape[0] == 1:
            # only one unique vector (or only one item after sampling)
            upper = np.array([1.0])
        else:
            sims  = cosine_similarity(vecs)
            upper = sims[np.triu_indices_from(sims, k=1)]

        rows.append({
            "cluster_id": cid,
            "size": sz,
            "median_sim": np.median(upper),
            "min_sim":    upper.min(),
            "p25":        np.percentile(upper, 25),
            "p75":        np.percentile(upper, 75),
            "snippet": textwrap.shorten(
                take["txt"].mode()[0] if not take["txt"].empty else "",
                width=60, placeholder="…")
        })

    if not rows:
        click.echo("⚠️  No clusters survived the filters.")
        sys.exit()

    out = pd.DataFrame(rows).sort_values("size", ascending=False)

    # --- export / print ----------------------------------------------------
    if export_csv:
        out.to_csv(export_csv, index=False)
        click.echo(f"📄 Stats written to {export_csv}")

    # global aggregates
    agg = out.agg({
        "size":       ["count", "min", "median", "mean", "max"],
        "median_sim": ["min", "median", "mean", "max"],
        "p25":        ["min", "median", "mean", "max"],
        "p75":        ["min", "median", "mean", "max"]
    })
    click.echo("\n── Corpus‑level summary ─────────────────────────────")
    click.echo(agg.to_string(float_format=lambda x: f"{x:,.3f}"))
    click.echo("────────────────────────────────────────────────────\n")

    if stats_only:
        return

    if top:
        show = out.head(int(top))
    else:
        show = out
    # nicer formatting
    show = show.assign(
        median_sim=lambda d: d["median_sim"].round(5),
        min_sim=lambda d: d["min_sim"].round(5),
        p25=lambda d: d["p25"].round(5),
        p75=lambda d: d["p75"].round(5)
    )
    click.echo(show.to_string(index=False))

# ─── Updated CLI commands ─────────────────────────────────────────────
@cluster.command("persist-cv-clusters")
def persist_cv_clusters():
    """
    Train and persist HDBSCAN on CV latents.
    """
    # fetch data
    click.echo("Loading CV embeddings with experience...")
    df = fetch_embeddings_cv_with_courses_filtered_with_experience()
    click.echo(f"Fetched {len(df):,} CV records.")

    # parameters
    cv_params = {
        "min_cluster_size": 5,
        "min_samples": 3,
        "metric": "euclidean",
        "algorithm": "generic",
        "cluster_selection_method": "eom",
        "prediction_data": True
    }

    _persist_clusters(
        df=df,
        latent_col="latent_code",
        key_cols=["fonte_aluno", "matricula"],
        clusters_table="cv_clusters",
        centroids_table="cv_cluster_centroids",
        cluster_file_path=CV_CLUSTER_FILE_PATH,
        params=cv_params,
        description="CV"
    )


@cluster.command("persist-job-clusters")
def persist_job_clusters():
    """
    Train and persist HDBSCAN on job latents.
    """
    # fetch data
    click.echo("Loading job embeddings and metadata...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df):,} job records.")

    # parameters
    job_params = {
        "min_cluster_size": 5,
        "min_samples": 3,
        "metric": "euclidean", 
        "algorithm": "generic",
        "cluster_selection_method": "eom",
        "prediction_data": True
    }

    _persist_clusters(
        df=df,
        latent_col="latent_code",
        key_cols=["fonte_aluno", "matricula", "contract_id"],
        clusters_table="job_clusters",
        centroids_table="job_cluster_centroids",
        cluster_file_path=JOB_CLUSTER_FILE_PATH,
        params=job_params,
        description="job"
    )

@cluster.command("persist-nomic-cv-clusters")
def persist_nomic_cv_clusters():
    """
    Train and persist HDBSCAN on CV nomic embeddings.
    """
    # fetch data
    click.echo("Loading CV embeddings with experience...")
    df = fetch_embeddings_cv_with_courses_filtered_with_experience()
    click.echo(f"Fetched {len(df):,} CV records.")

    # parameters
    cv_params = {
        "min_cluster_size": 5,
        "min_samples": 3,
        "metric": "euclidean",
        "algorithm": "generic",
        "cluster_selection_method": "eom",
        "prediction_data": True
    }

    _persist_clusters(
        df=df,
        latent_col="embedding",
        key_cols=["fonte_aluno", "matricula"],
        clusters_table="cv_nomic_clusters",
        centroids_table="cv_nomic_centroids",
        cluster_file_path=CV_NOMIC_CLUSTER_FILE_PATH,
        params=cv_params,
        description="CV_NOMIC"
    )


@cluster.command("persist-nomic-job-clusters")
def persist_nomic_job_clusters():
    """
    Train and persist HDBSCAN on job nomic embeddings.
    """
    # fetch data
    click.echo("Loading job embeddings and metadata...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df):,} job records.")

    # parameters
    job_params = {
        "min_cluster_size": 5,
        "min_samples": 3,
        "metric": "euclidean", 
        "algorithm": "generic",
        "cluster_selection_method": "eom",
        "prediction_data": True
    }

    _persist_clusters(
        df=df,
        latent_col="embedding",
        key_cols=["fonte_aluno", "matricula", "contract_id"],
        clusters_table="job_nomic_clusters",
        centroids_table="job_nomic_centroids",
        cluster_file_path=JOB_NOMIC_CLUSTER_FILE_PATH,
        params=job_params,
        description="job_nomic"
    )

@cluster.command("job-cluster-check")
def job_cluster_check():
    """
    Sweep HDBSCAN hyperparameters on job latent codes,
    pick best by Calinski-Harabasz, and name clusters.
    """
    text_col      = "raw_input"

    click.echo("Fetching job metadata and embeddings...")
    df = fetch_embeddings_job_with_metadata()
    click.echo(f"Fetched {len(df):,} records.")

    # Build and normalize latent matrix
    X      = np.vstack(df["latent_code"].values).astype(np.float32)
    X_norm = normalize(X, norm="l2")

    # Run hyperparameter sweep via helper
    best, log_path = sweep_hdbscan(
        X_norm,
        log_prefix="job"
    )

    # Assign best labels
    df["hdb_best"] = best["labels"]

    # Generate human-readable cluster names
    cluster_names = generate_cluster_names_manual(
        df,
        cluster_col="hdb_best",
        text_col=text_col
    )
    df["cluster_readable"] = df["hdb_best"].map(cluster_names)

    # Print cluster summaries
    for cid, label in cluster_names.items():
        click.echo(f"Cluster {cid:3d}: {label}")

    click.echo(f"Saved clustering log to {log_path}")

@cluster.command("cv-cluster-check")
def cv_cluster_check():
    """
    Sweep HDBSCAN hyperparameters on CV latent codes with real experience,
    pick best by Calinski-Harabasz, and name clusters.
    """
    embedding_col = CVS_EMBEDDING_COLUMN_NAME
    text_col      = "text"

    click.echo("Fetching CVs with parsed experience and embeddings...")
    df = fetch_embeddings_cv_with_courses_filtered_with_experience(embedding_column=embedding_col)
    click.echo(f"Fetched {len(df):,} records.")

    # Build and normalize latent matrix
    X      = np.vstack(df["latent_code"].values).astype(np.float32)
    X_norm = normalize(X, norm="l2")

    # Run hyperparameter sweep via shared helper
    best, log_path = sweep_hdbscan(
        X_norm,
        log_prefix="cv"
    )

    # Assign best labels
    df["hdb_best"] = best["labels"]

    # Generate human-readable cluster names
    cluster_names = generate_cluster_names_manual(
        df,
        cluster_col="hdb_best",
        text_col=text_col
    )
    df["cluster_readable"] = df["hdb_best"].map(cluster_names)

    # Print cluster summaries
    for cid, label in cluster_names.items():
        click.echo(f"Cluster {cid:3d}: {label}")

    click.echo(f"Saved clustering log to {log_path}")

@deprecated("Esse comando é antigo e não deve ser usado.")
@cluster.command("cv")
def cv_cluster_gen_labels():
    embedding_col = "embedding"
    text_col      = "text"

    # 1) Load your filtered CVs
    df = fetch_embeddings_cv_with_courses_filtered_with_experience(embedding_col)

    # 2) Build and normalize the latent codes
    X      = np.vstack(df["latent_code"].values)
    X_norm = normalize(X, norm="l2")

    # 3) HDBSCAN micro-clusters
    mcs, ms = 2, 1
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="leaf"
    )
    labels = clusterer.fit_predict(X_norm)
    df["hdb_micro"] = labels

    # 4) Re-assign all “noise” (-1) as their own solo clusters
    max_label = df["hdb_micro"].max()
    next_label = max_label + 1
    # for each noise index, give a new unique cluster ID
    for idx in df.index[df["hdb_micro"] == -1]:
        df.at[idx, "hdb_micro"] = next_label
        next_label += 1

    # 5) Count and sort cluster sizes
    size_counts = df["hdb_micro"].value_counts().sort_values(ascending=False)
    click.echo("Top 10 clusters by size:")
    for cid, size in size_counts.head(10).items():
        click.echo(f"  Cluster {cid:3d}: {size} CVs")

    # 6) Generate human-readable names for ALL clusters
    cluster_names = generate_cluster_names_manual(
        df, cluster_col="hdb_micro", text_col=text_col, top_n=10
    )
    df["cluster_readable"] = df["hdb_micro"].map(cluster_names)

    click.echo("\nTop cluster labels:")
    for cid in size_counts.head(10).index:
        click.echo(f"  Cluster {cid:3d}: {cluster_names.get(cid, '—')}")

    # 7) Save size distribution
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dist_path = f"data/processed/cv_cluster_gen_sizes_{mcs}_{ms}_{timestamp}.txt"
    with open(dist_path, "w") as f:
        f.write("ClusterID\tSize\n")
        for cid, size in size_counts.items():
            f.write(f"{cid}\t{size}\n")
    click.echo(f"\nFull cluster size distribution saved to {dist_path}")

    # 8) Save per‐CV labels including the solo clusters
    labels_path = f"data/processed/cv_cluster_gen_labels_{mcs}_{ms}_{timestamp}.txt"
    df[["fonte_aluno", "matricula", "hdb_micro", "cluster_readable"]].to_csv(
        labels_path,
        sep=";",
        index=True,
        index_label="cv_index",
        header=["fonte_aluno", "matricula", "cluster_id", "cluster_label"]
    )
    click.echo(f"Cluster labels saved to {labels_path}")