import uuid
import json
import pickle
import psycopg2
import pandas as pd
import numpy as np
import hdbscan
from functools import lru_cache
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from ..data.db import POSTGRES_URL

# ─── cache loaders ──────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _load_hdbscan_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=None)
def _load_cv_centroid(cluster_id: int, table_name : str) -> np.ndarray | None:
    """
    Fetches the centroid vector for exactly one CV‐cluster.
    Returns None if not found.
    """
    sql = f"""
      SELECT centroid
        FROM {table_name}
       WHERE cluster_id = %s
    """
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        cur = conn.cursor()
        cur.execute(sql, (cluster_id,))          # note the comma
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None

    raw = row[0]  # the 'centroid' column
    # raw might be stored as JSON text, or as a Postgres array/list
    vec = json.loads(raw) if isinstance(raw, str) else raw
    return np.array(vec, dtype=np.float32)

@lru_cache(maxsize=None)
def _load_job_centroids(table: str) -> dict[int, np.ndarray]:
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    cur.execute(f"SELECT cluster_id, centroid FROM {table}")
    rows = cur.fetchall()
    conn.close()
    return {
        cid: np.array(json.loads(c) if isinstance(c, str) else c, dtype=np.float32)
        for cid, c in rows
    }

@lru_cache(maxsize=None)
def _load_job_assignments(table: str) -> pd.DataFrame:
    conn = psycopg2.connect(POSTGRES_URL)
    df = pd.read_sql(f"SELECT fonte_aluno, matricula, contract_id, cluster_id FROM {table}", conn)
    conn.close()
    return df

# ─── drop-in replacement ────────────────────────────────────────────────────
def match_jobs_pipeline(
    query_vec: np.ndarray,
    *,
    cv_cluster_file: str | None = None,
    cv_centroids_table: str = "cv_cluster_centroids",
    job_cluster_file: str,
    job_centroids_table: str = "job_cluster_centroids",
    job_assignments_table: str,
    jobs_fetcher,
    embedding_col: str,
    primary_top_k: int = 5,
    skip_fit: bool = False,
) -> dict:
    print("=== match_jobs_pipeline start ===")

    # 0) CV‐cluster override
    print("0) CV-cluster override")
    student_cv_cluster = None
    if cv_cluster_file and not skip_fit:
        print(f"  • loading CV cluster model from {cv_cluster_file}")
        cv_clust = _load_hdbscan_model(cv_cluster_file)
        labels, _ = hdbscan.approximate_predict(cv_clust, query_vec.reshape(1, -1))
        student_cv_cluster = int(labels[0])
        print(f"  • predicted student_cv_cluster = {student_cv_cluster}")
        if student_cv_cluster != -1:
            print(f"  • loading centroid for CV cluster {student_cv_cluster} from table {cv_centroids_table}")
            cent = _load_cv_centroid(student_cv_cluster, cv_centroids_table)
            if cent is not None and student_cv_cluster != -1:
                query_vec = normalize(cent.reshape(1, -1), norm="l2")[0]
                print("  • query_vec overridden with cluster centroid")

    # 1) Score job‐cluster centroids
    print("1) Score job-cluster centroids")
    centroids = _load_job_centroids(job_centroids_table)
    print(f"  • loaded {len(centroids)} centroids from {job_centroids_table}")
    cluster_sims = {
        cid: float(cosine_similarity(
            query_vec.reshape(1, -1),
            normalize(cent.reshape(1, -1), norm="l2")
        )[0, 0])
        for cid, cent in centroids.items()
    }
    sorted_clusters = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
    top_clusters = sorted_clusters[:10]
    print(f"  • top 10 clusters by similarity: {top_clusters}")

    # 2) Assign student_job_cluster
    print("2) Assign student_job_cluster")
    print(f"  • loading job cluster model from {job_cluster_file}")
    job_clust = _load_hdbscan_model(job_cluster_file)
    labels, _ = hdbscan.approximate_predict(job_clust, query_vec.reshape(1, -1))
    orig = int(labels[0])
    if orig == -1:
        used_simple = True
        student_job_cluster = top_clusters[0]
    else:
        used_simple = False
        student_job_cluster = orig
    print(f"  • orig cluster = {orig}, used_simple = {used_simple}, student_job_cluster = {student_job_cluster}")

    # 3) Fetch + merge jobs + assignments
    print("3) Fetch + merge jobs + assignments")
    jobs_df   = jobs_fetcher()
    print(f"  • fetched {len(jobs_df)} jobs")
    assign_df = _load_job_assignments(job_assignments_table)
    print(f"  • fetched {len(assign_df)} assignments from {job_assignments_table}")
    df = jobs_df.merge(assign_df,
                      on=["fonte_aluno", "matricula", "contract_id"],
                      how="inner")
    print(f"  • merged dataframe has {len(df)} rows")

    # 4) Rerank primary + one‐per‐other
    print("4) Rerank primary cluster + one per other")
    matched = []
    # primary cluster
    primary = df[df.cluster_id == student_job_cluster]
    print(f"  • primary cluster ({student_job_cluster}) size = {len(primary)}")
    if not primary.empty:
        mat = np.vstack(primary[embedding_col].tolist()).astype(np.float32)
        sims = cosine_similarity(
            query_vec.reshape(1, -1),
            normalize(mat, norm="l2")
        ).flatten()
        primary = primary.copy(); primary["sim"] = sims
        print(f"  • top {primary_top_k} in primary cluster: {list(primary.contract_id)}")
        for r in primary.nlargest(primary_top_k, "sim").itertuples():
            matched.append({
                "cluster_id":  student_job_cluster,
                "contract_id": int(r.contract_id),
                "similarity":  float(r.sim),
                "raw_input":   r.raw_input,
            })

    # one best from each other top cluster
    for cid in top_clusters:
        if cid == student_job_cluster: continue
        grp = df[df.cluster_id == cid]
        if grp.empty: 
            print(f"  • cluster {cid} empty, skipping")
            continue
        mat2 = np.vstack(grp[embedding_col].tolist()).astype(np.float32)
        sims2 = cosine_similarity(
            query_vec.reshape(1, -1),
            normalize(mat2, norm="l2")
        ).flatten()
        grp = grp.copy(); grp["sim"] = sims2
        best = grp.nlargest(1, "sim").iloc[0]
        print(f"  • best in cluster {cid} = contract {best.contract_id} (sim={best.sim:.4f})")
        matched.append({
            "cluster_id":  cid,
            "contract_id": int(best.contract_id),
            "similarity":  float(best.sim),
            "raw_input":   best.raw_input,
        })

    # final sort
    matched.sort(key=lambda x: x["similarity"], reverse=True)
    print(f"Matched {len(matched)} jobs in total, returning pipeline result\n=== done ===")

    return {
        "student_cv_cluster":     student_cv_cluster,
        "student_job_cluster":    student_job_cluster,
        "used_simple_assignment": used_simple,
        "matched_jobs":           matched,
    }
