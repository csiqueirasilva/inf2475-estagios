from typing import Dict, List, Union
import hdbscan
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from datetime import datetime
import click

from .db import POSTGRES_URL

def sweep_hdbscan(
    X_norm: np.ndarray,
    log_prefix: str = "job",
    metrics: list = None,
    selection_methods: list = None,
    min_cluster_sizes: list = None,
    min_samples_list: list = None,
    log_dir: str = "data/processed"
):
    """
    Sweep HDBSCAN hyperparameters over provided grids, log results,
    and return the best clustering labels by Calinski-Harabasz score.

    Args:
      X_norm: normalized feature matrix (N, D)
      log_prefix: prefix for log filename
      metrics: list of (metric, algorithm)
      selection_methods: list of cluster_selection_method
      min_cluster_sizes: list of ints for min_cluster_size
      min_samples_list: list of ints for min_samples
      log_dir: directory to save the log file

    Returns:
      tuple(best_config, log_path)
      where best_config is a dict with keys: 'labels', 'metric', 'algo', 'sel_method', 'mcs', 'ms', 'ch'
    """
    # Ensure data is double precision for HDBSCAN
    if X_norm.dtype != np.float64:
        X_norm = X_norm.astype(np.float64)

    # Default grids
    if metrics is None:
        metrics = [
            ("euclidean", "best"),
            ("manhattan", "best"),
            ("chebyshev", "best"),
            ("cosine", "generic"),
            ("correlation", "generic"),
        ]
    if selection_methods is None:
        selection_methods = ["leaf", "eom"]
    if min_cluster_sizes is None:
        min_cluster_sizes = [5, 10, 20]
    if min_samples_list is None:
        min_samples_list = [3, 5, 10]

    # Prepare log
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"{log_dir}/{log_prefix}_clustering_methods_{date_str}.txt"
    best = {
        "ch": -np.inf,
        "metric": None,
        "algo": None,
        "sel_method": None,
        "mcs": None,
        "ms": None,
        "labels": None
    }

    with open(log_path, "w") as log_file:
        header = f"Clustering sweep log ({log_prefix}) - {date_str}\n"
        click.echo(header.strip())
        log_file.write(header)
        log_file.flush()

        for metric_name, algo in metrics:
            for sel in selection_methods:
                for mcs in min_cluster_sizes:
                    for ms in min_samples_list:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric=metric_name,
                            algorithm=algo,
                            cluster_selection_method=sel
                        )
                        labels = clusterer.fit_predict(X_norm)
                        n_clusters = len(set(labels) - {-1})
                        if n_clusters < 2:
                            continue

                        sil = silhouette_score(X_norm, labels)
                        ch = calinski_harabasz_score(X_norm, labels)
                        db = davies_bouldin_score(X_norm, labels)

                        line = (
                            f"{metric_name:11s} | {sel:4s} | "
                            f"mcs={mcs:2d}, ms={ms:2d} -> "
                            f"{n_clusters:3d} clusters, "
                            f"sil={sil:.3f}, ch={ch:.3f}, db={db:.3f}\n"
                        )
                        click.echo(line.strip())
                        log_file.write(line)
                        log_file.flush()

                        if ch > best["ch"]:
                            best.update({
                                "ch": ch,
                                "metric": metric_name,
                                "algo": algo,
                                "sel_method": sel,
                                "mcs": mcs,
                                "ms": ms,
                                "labels": labels
                            })

        summary = (
            f"Best setting by CH score ({log_prefix}):\n"
            f"  metric={best['metric']} (algo={best['algo']}), "
            f"sel_method={best['sel_method']}, "
            f"min_cluster_size={best['mcs']}, "
            f"min_samples={best['ms']} -> "
            f"CH score={best['ch']:.3f}\n"
        )
        click.echo(summary.strip())
        log_file.write(summary)
        log_file.flush()

    return best, log_path

def store_clusters_and_centroids(
    df: pd.DataFrame,
    cluster_col: str,
    key_cols: List[str],
    clusters_table: str,
    centroids_table: str,
    latent_col: str = "latent_code",
    pg_conn_params: Union[str, dict] = POSTGRES_URL
):
    """
    1) Clear existing rows in clusters_table and centroids_table.
    2) Upsert assignments into clusters_table (one row per item).
    3) Compute centroids per cluster_id and upsert into centroids_table.
    """
    conn = psycopg2.connect(pg_conn_params) if isinstance(pg_conn_params, str) \
           else psycopg2.connect(**pg_conn_params)
    cur = conn.cursor()

    # ── 0) Clear out old data to avoid stale assignments/centroids ─────────
    cur.execute(f"DELETE FROM {clusters_table};")
    cur.execute(f"DELETE FROM {centroids_table};")
    conn.commit()

    # ── 1) Upsert assignments ─────────────────────────────────────────────
    num_keys = len(key_cols)
    placeholders = ", ".join(["%s"] * (num_keys + 1))
    assign_sql = f"""
    INSERT INTO {clusters_table} ({', '.join(key_cols)}, cluster_id)
      VALUES ({placeholders})
    ON CONFLICT ({', '.join(key_cols)}) DO UPDATE
      SET cluster_id = EXCLUDED.cluster_id;
    """
    for _, row in df.iterrows():
        values = [row[c] for c in key_cols] + [int(row[cluster_col])]
        cur.execute(assign_sql, values)
    conn.commit()

    # ── 2) Compute & upsert centroids ─────────────────────────────────────
    groups: Dict[int, List[np.ndarray]] = {}
    for _, row in df.iterrows():
        cid = int(row[cluster_col])
        vec = np.array(row[latent_col], dtype=np.float32)
        groups.setdefault(cid, []).append(vec)

    cent_sql = f"""
    INSERT INTO {centroids_table} (cluster_id, centroid)
      VALUES (%s, %s)
    ON CONFLICT (cluster_id) DO UPDATE
      SET centroid = EXCLUDED.centroid;
    """
    for cid, vecs in groups.items():
        centroid = np.mean(vecs, axis=0).tolist()
        cur.execute(cent_sql, (cid, centroid))

    conn.commit()
    cur.close()
    conn.close()