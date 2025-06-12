import uuid
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import click

from internship_matching.data.db import POSTGRES_URL

def match_jobs_pipeline(
    query_vec: np.ndarray,
    *,
    # if you want CV‐cluster logic, pass these:
    cv_cluster_file: str | None = None,
    cv_centroids_table: str = "cv_cluster_centroids",
    # always required for jobs:
    job_cluster_file: str,
    job_centroids_table: str = "job_cluster_centroids",
    job_assignments_table: str,
    jobs_fetcher,                  # e.g. fetch_embeddings_job_with_metadata
    embedding_col: str,            # "latent_code" or "embedding"
    primary_top_k: int = 5,
    skip_fit: bool = False,
) -> dict:
    """
    Runs the full match‐and‐rerank pipeline.
    Returns a dict with:
      - student_cv_cluster, student_job_cluster,
      - used_simple_assignment, clusters_by_similarity, matched_jobs
    """
    # ── (0) Optional CV‐cluster override ────────────────────────────────
    student_cv_cluster = None
    if cv_cluster_file and skip_fit is False:
        with open(cv_cluster_file, "rb") as f:
            cv_clust = pickle.load(f)
        labels, _ = hdbscan.approximate_predict(cv_clust, query_vec.reshape(1, -1))
        student_cv_cluster = int(labels[0])

        # fetch CV centroid & override query_vec if found
        conn = psycopg2.connect(POSTGRES_URL)
        cv_cent = conn.cursor()
        cv_cent.execute(
            f"SELECT centroid FROM {cv_centroids_table} WHERE cluster_id=%s",
            (student_cv_cluster,)
        )
        row = cv_cent.fetchone()
        conn.close()
        if row and student_cv_cluster != -1:
            cent = np.array(json.loads(row[0]) if isinstance(row[0], str) else row[0], dtype=np.float64)
            query_vec = normalize(cent.reshape(1, -1), norm="l2")[0]

    # ── (1) Score job‐cluster centroids ─────────────────────────────────
    conn = psycopg2.connect(POSTGRES_URL)
    job_curs = conn.cursor()
    job_curs.execute(f"SELECT cluster_id, centroid FROM {job_centroids_table}")
    centroids = {cid: np.array(json.loads(c) if isinstance(c, str) else c, dtype=np.float64)
                 for cid, c in job_curs.fetchall()}
    conn.close()

    cluster_sims = {}
    for cid, cent in centroids.items():
        c_n = normalize(cent.reshape(1, -1), norm="l2")[0]
        cluster_sims[cid] = float(cosine_similarity(query_vec.reshape(1,-1), c_n.reshape(1,-1))[0,0])
    sorted_clusters = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
    top_clusters    = sorted_clusters[:10]

    # ── (2) Assign student_job_cluster ──────────────────────────────────
    with open(job_cluster_file, "rb") as f:
        job_clust = pickle.load(f)
    labels, _ = hdbscan.approximate_predict(job_clust, query_vec.reshape(1,-1))
    orig = int(labels[0])
    if orig == -1:
        used_simple         = True
        student_job_cluster = top_clusters[0]
    else:
        used_simple         = False
        student_job_cluster = orig

    # ── (3) Fetch all jobs + their cluster assignments ─────────────────
    jobs_df = jobs_fetcher()
    conn    = psycopg2.connect(POSTGRES_URL)
    assignments_sql=f"SELECT fonte_aluno, matricula, contract_id, cluster_id FROM {job_assignments_table}"
    assignments = pd.read_sql(assignments_sql, conn)
    conn.close()
    df = jobs_df.merge(assignments,
                      on=["fonte_aluno","matricula","contract_id"],
                      how="inner")

    # ── (4) Rerank primary cluster + top K, then one‐per‐others ───────
    matched = []

    # primary cluster top K
    primary = df[df.cluster_id==student_job_cluster]
    if not primary.empty:
        mat = np.vstack(primary[embedding_col].tolist()).astype(np.float32)
        mat_n = normalize(mat, norm="l2")
        sims = cosine_similarity(query_vec.reshape(1,-1), mat_n).flatten()
        primary = primary.copy()
        primary["sim"] = sims
        for r in primary.sort_values("sim",ascending=False).head(primary_top_k).itertuples():
            matched.append({
                "cluster_id": student_job_cluster,
                "contract_id": int(r.contract_id),
                "similarity": float(r.sim),
                "raw_input": r.raw_input
            })

    # one best from each of the other clusters
    for cid in top_clusters:
        if cid == student_job_cluster:
            continue
        grp = df[df.cluster_id==cid]
        if grp.empty:
            continue
        mat2 = np.vstack(grp[embedding_col].tolist()).astype(np.float32)
        mat2_n = normalize(mat2, norm="l2")
        sims2 = cosine_similarity(query_vec.reshape(1,-1), mat2_n).flatten()
        grp = grp.copy()
        grp["sim"] = sims2
        best = grp.sort_values("sim",ascending=False).iloc[0]
        matched.append({
            "cluster_id": cid,
            "contract_id": int(best.contract_id),
            "similarity": float(best.sim),
            "raw_input": best.raw_input
        })

    # final sort
    matched.sort(key=lambda x: x["similarity"], reverse=True)

    return {
      "student_cv_cluster":    student_cv_cluster,
      "student_job_cluster":   student_job_cluster,
      "used_simple_assignment":used_simple,
      # "clusters_by_similarity":[{"cluster_id":cid,"similarity":cluster_sims[cid]} for cid in top_clusters],
      "matched_jobs":          matched
    }