from datetime import datetime
import logging
import math
from pathlib import Path
import pickle
import re
from typing import Counter
import uuid
import json
import sqlite3
import click
import hdbscan
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from sklearn.feature_extraction.text import CountVectorizer
import umap

from .data.match import match_jobs_pipeline

from .data.plot import plot_embeddings_latent_space

from .utils import deprecated

from .data.clusters import store_clusters_and_centroids, sweep_hdbscan

from .data.job_autoencoder import JobAutoencoder

from .constants import CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_AUTOENCODER_FILE_PATH, CVS_EMBEDDING_COLUMN_NAME, CVS_TRAIN_SEED, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_AUTOENCODER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME, JOBS_TRAIN_SEED

from .data.autolabeler import PT_STOPWORDS, CVAutoLabeler

from .data.autoencoder import CVAutoencoder

from .data.cvs      import cv_fill_raw_embeddings, fetch_embeddings_cv, fetch_embeddings_cv_with_courses, fetch_embeddings_cv_with_courses_filtered_with_experience, fetch_experiences_per_student, reset_database_embeddings_size_cv, sanitize_input_cvs, store_embeddings_cv, store_embeddings_singles_cv
from .data.jobs     import TORCH_DEVICE, fetch_embeddings_job_with_metadata, fetch_job_cluster_centroids, sanitize_input_jobs, store_embeddings_singles_job
from .training.train_cv_job    import train_cv_job_matching
from .training.train_cv_feat   import train_cv_feature_scoring
from .training.train_job_feat  import train_job_feature_scoring
from .models.explainers       import explain_cv_job, explain_cv_feat, explain_job_feat
from .models.inference       import infer_cv_job, infer_cv_feat, infer_job_feat
from .data.db import POSTGRES_URL, init_db, start_database_import
from .data.embed import get_embed_func

_DEFAULT_DATABASE_FILE = "data/processed/data.db"

# ─── ROOT CLI ────────────────────────────────────────────────────────────────
@click.group()
@click.option('--verbose', is_flag=True, help="Enable verbose output")
@click.option('--db-path', 'db_path', default=_DEFAULT_DATABASE_FILE, help="Path to the database file")
@click.option('--debug', is_flag=True, help="Wait for debugger to attach on port 5678")
@click.pass_context
def cli(ctx, verbose, db_path, debug):
    """Internship matching toolkit."""
    ctx.obj = {'verbose': verbose, 'db_path': db_path}
    if debug:
        import debugpy
        debugpy.connect(("localhost", 5678))
        click.echo("🐛 Connected to VS Code debugger on port 5678…")
        debugpy.wait_for_client()

# ─── SANITIZE GROUP ───────────────────────────────────────────────────────────
@cli.group()
def sanitize():
    """Data sanitization commands."""
    pass

@sanitize.command()
def embeddings():
    cv_fill_raw_embeddings()

@sanitize.command()
@click.pass_context
def database(ctx):
    """Sanitize and rebuild the database."""
    db_path = ctx.obj.get('db_path', _DEFAULT_DATABASE_FILE)
    start_database_import(db_path)
    click.echo(f"✅ Database rebuilt at {db_path}")

@sanitize.command()
def cvs():
    records = sanitize_input_cvs()
    embed_func = get_embed_func()
    store_embeddings_singles_cv(records, embed_func)
    click.echo(f"✅ Generated embeddings for CVs")

@sanitize.command()
def jobs():
    records = sanitize_input_jobs()
    embed_func = get_embed_func()
    store_embeddings_singles_job(records, embed_func)
    click.echo(f"✅ Generated embeddings for CVs")

# ─── TEST GROUP ───────────────────────────────────────────────────────────
@cli.group()
def test():
    """Data test commands."""
    pass

@test.command("cv-autoencoder")
@click.argument("matricula")
@click.option("--fonte", default="VRADM", show_default=True)
def test_autoencoder(matricula, fonte):
    if matricula == 'roundtrip':
        model = CVAutoencoder()
        embeds = fetch_embeddings_cv()
        enc_dec_mae, dec_enc_mae = model.round_trip_errors(embeds)
        click.echo(f"encode→decode MAE: {enc_dec_mae.min()}, {enc_dec_mae.mean()}, {enc_dec_mae.max()}")
        click.echo(f"decode→encode MAE: {dec_enc_mae.min()}, {dec_enc_mae.mean()}, {dec_enc_mae.max()}")
    else:
        try:
            result = CVAutoencoder.test_cv(matricula, fonte)
        except ValueError as e:
            click.echo(f"❌ {e}", err=True)
            raise click.Abort()
        click.echo(f"🔑 Latent code ({len(result['latent'])} dims):")
        click.echo("  " + ", ".join("{:.6f}".format(v) for v in result['latent']))
        click.echo(f"📊 Reconstruction MSE: {result['mse']:.6e}")

@test.command("cv-autolabeler")
@click.argument("matricula", type=str)
@click.option(
    "--fonte",
    default="VRADM",
    show_default=True,
    help="Fonte_aluno key to look up in cv_embeddings"
)
@click.option(
    "--type",
    "method",
    type=click.Choice(["KEYBERT", "CTFIDF"], case_sensitive=False),
    default="KEYBERT",
    show_default=True,
    help="Label naming strategy"
)
def test_autolabeler(matricula: str, fonte: str, method: str):
    """
    Fetch the latent_code for one CV and print its auto-label.
    """
    # Load the labeler with the appropriate naming strategy
    if method.upper() == "KEYBERT":
        labeler = CVAutoLabeler.load_keybert()
    else:
        labeler = CVAutoLabeler.load_ctfidf()

    # Get and print the auto-label
    label = labeler.get_auto_label(fonte, matricula)
    click.echo(label)

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
        "min_cluster_size": 20,
        "min_samples": 10,
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
        "min_cluster_size": 20,
        "min_samples": 10,
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
        "min_cluster_size": 20,
        "min_samples": 10,
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

# ─── PLOT GROUP ──────────────────────────────────────────────────────────────
@cli.group()
def plot():
    """Model plot commands."""
    pass

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


_word_re = re.compile(r"\b\w+\b", flags=re.UNICODE)

def generate_cluster_names_manual(
    df,
    cluster_col="hdbscan",
    text_col="text",
    top_n=3
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
            for w in _word_re.findall(doc)
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
                for w in _word_re.findall(doc)
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

# ─── TRAIN GROUP ──────────────────────────────────────────────────────────────
@cli.group()
def train():
    """Model training commands."""
    pass

@train.command("job-autoencoder")
def train_cv_autoencoder():
    JobAutoencoder.train_from_db()
    JobAutoencoder.generate_all_latents()

@train.command("cv-autoencoder")
def train_cv_autoencoder():
    CVAutoencoder.train_from_db()
    CVAutoencoder.generate_all_latents()

@deprecated("Foi um experimento, não é usado no trabalho. Acabei usando diversas tecnologias para testar o conceito.")
@train.command("cv-autolabeler")
def train_cv_autolabeler():
    labeler = CVAutoLabeler()
    # labeler.fit_kmeans()
    # labeler.name_clusters_keybert()
    # labeler.save_keybert()
    # labeler.name_clusters_tfidf()
    # labeler.save_tfidf()
    # labeler.name_clusters_ctfidf()
    # labeler.save_ctfidf()
    labeler.name_clusters_bertopic()
    labeler.save_bertopic()

@deprecated("Acabou sendo substituído pelo comando de treinar os clusters individualmente.")
@train.command("cv-job")
def train_cv_job():
    train_cv_job_matching()

@deprecated("Não foi possível extrair features dos CVs.")
@train.command("cv-feat")
def train_cv_feat():
    train_cv_feature_scoring()

@deprecated("Não foi possível extrair features dos Jobs.")
@train.command("job-feat")
def train_job_feat():
    train_job_feature_scoring()

# ─── RUN GROUP ────────────────────────────────────────────────────────────────
@cli.group()
def run():
    """Inference (run) commands."""
    pass

def log_to_db(uuid_str: str, group: str, command: str, inp: dict, outp: dict, logger: logging.Logger):
    """Append this run's input/output to a local SQLite audit log."""
    logger.debug(f"Logging run {uuid_str} to database runs.db")
    conn = sqlite3.connect("runs.db")
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, 
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
        grp TEXT, 
        cmd TEXT, 
        input TEXT, 
        output TEXT
      )""")
    c.execute(
        "INSERT INTO runs (id, grp, cmd, input, output) VALUES (?, ?, ?, ?, ?)",
        (uuid_str, group, command, json.dumps(inp), json.dumps(outp))
    )
    conn.commit()
    conn.close()

# ─── JOB‐AUTOENCODER ───────────────────────────────────────────────────────
@run.command("job-autoencoder")
@click.option(
    "--text", "-t",
    type=str,
    required=True,
    help="Raw job text to embed and match"
)
def run_job_autoencoder(text: str):
    """
    Embed job text via the JobAutoencoder and return the top 10 most
    similar job postings (by cosine similarity) without any clustering.
    """
    uid = str(uuid.uuid4())
    click.echo("Embedding job text and matching against jobs…")

    # 1) Raw embedding
    embed_func = get_embed_func()
    raw_embed = np.array(embed_func(text), dtype=np.float32)

    # 2) Encode into latent space
    model = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(raw_embed, device=device).unsqueeze(0)
    with torch.no_grad():
        _, z = model(x)
    latent = z.cpu().numpy().squeeze(0).astype(np.float64)

    # 3) Normalize query
    query_vec = normalize(latent.reshape(1, -1), norm="l2")[0]

    # 4) Fetch all job latents
    df = fetch_embeddings_job_with_metadata()

    # 5) Build & normalize matrix
    job_mat = np.vstack(df["latent_code"].tolist()).astype(np.float64)
    job_mat_norm = normalize(job_mat, norm="l2")

    # 6) Cosine similarities
    sims = cosine_similarity(query_vec.reshape(1, -1), job_mat_norm).flatten()

    # 7) Top 10
    top_idxs = sims.argsort()[::-1][:10]
    results = [
        {
            "contract_id": int(df.iloc[i].contract_id),
            "similarity": float(sims[i]),
            "raw_input": df.iloc[i].raw_input
        }
        for i in top_idxs
    ]

    click.echo(json.dumps({"top_10_jobs": results}, indent=2))


# ─── CV‐AUTOENCODER ────────────────────────────────────────────────────────
@run.command("cv-autoencoder")
@click.option(
    "--text", "-t",
    type=str,
    required=True,
    help="Raw CV text to embed and match"
)
def run_cv_autoencoder(text: str):
    """
    Embed CV text via the CVAutoencoder and return the top 10 most
    similar CVs (by cosine similarity) without any clustering.
    """
    uid = str(uuid.uuid4())
    click.echo("Embedding CV text and matching against CVs…")

    # 1) Raw embedding
    embed_func = get_embed_func()
    raw_embed = np.array(embed_func(text), dtype=np.float32)

    # 2) Encode into latent space
    model = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(raw_embed, device=device).unsqueeze(0)
    with torch.no_grad():
        _, z = model(x)
    latent = z.cpu().numpy().squeeze(0).astype(np.float64)

    # 3) Normalize query
    query_vec = normalize(latent.reshape(1, -1), norm="l2")[0]

    # 4) Fetch all CV latents
    df = fetch_embeddings_cv_with_courses_filtered_with_experience()

    # 5) Build & normalize matrix
    cv_mat = np.vstack(df["latent_code"].tolist()).astype(np.float64)
    cv_mat_norm = normalize(cv_mat, norm="l2")

    # 6) Cosine similarities
    sims = cosine_similarity(query_vec.reshape(1, -1), cv_mat_norm).flatten()

    # 7) Top 10
    top_idxs = sims.argsort()[::-1][:10]
    results = [
        {
            "matricula": df.iloc[i].matricula,
            "similarity": float(sims[i]),
            "cv": df.iloc[i].text,
        }
        for i in top_idxs
    ]

    click.echo(json.dumps({"top_10_cvs": results}, indent=2))

@run.command("cv-nomic-job")
@click.option('--cv-file', 'cv_file', type=click.File('r'),
              help="Path to a plaintext CV file (bypass matricula lookup)")
@click.option('--cv-text', 'cv_text', type=str,
              help="Direct CV text input (bypass matricula lookup)")
@click.option("--matricula", 'matricula', type=str,
              help="Student matricula to fetch stored CV (if no text provided)")
@click.option("--fonte", default="VRADM", show_default=True,
              help="Fonte_aluno key for lookup if matricula is used")
@click.option("--skip-fit", is_flag=True,
              help="Skip HDBSCAN prediction; assign by nearest centroid only")
def run_cv_nomic_job(matricula: str, fonte: str, cv_file, cv_text, skip_fit: bool):
    """
    Match a CV against existing job clusters, using either direct input
    (--cv-file or --cv-text) or a stored matricula lookup, **without** any
    autoencoder step — we pull the raw embed from the DB.
    """
    uid = str(uuid.uuid4())
    click.echo("Matching CV to job clusters…")

    # ── 1) Prepare query embedding ────────────────────────────────────────
    if cv_file:
        raw_cv = cv_file.read()
    elif cv_text:
        raw_cv = cv_text
    elif matricula:
        # fetch CV embedding from DB
        from .data.cvs import fetch_single_embedding_cv
        emb = fetch_single_embedding_cv(fonte, matricula)
        if emb is None:
            click.echo(f"❌ No embedding found for ({fonte}, {matricula})", err=True)
            raise click.Abort()
        query_vec = normalize(
            np.array(emb, dtype=np.float32).reshape(1, -1),
            norm="l2"
        )[0]
    else:
        click.echo("❌ Please provide --cv-file, --cv-text or --matricula", err=True)
        raise click.Abort()

    # if we have raw text, embed it now
    if 'raw_cv' in locals():
        embed_func = get_embed_func()
        emb = embed_func(raw_cv)
        query_vec = normalize(
            np.array(emb, dtype=np.float32).reshape(1, -1),
            norm="l2"
        )[0]

    out = match_jobs_pipeline(
        query_vec,
        cv_cluster_file=CV_NOMIC_CLUSTER_FILE_PATH,
        job_cluster_file=JOB_NOMIC_CLUSTER_FILE_PATH,
        job_centroids_table="job_nomic_centroids",
        job_assignments_table="job_nomic_clusters",
        jobs_fetcher=fetch_embeddings_job_with_metadata,
        embedding_col="embedding",
        skip_fit=skip_fit
    )
    click.echo(json.dumps(out, indent=2))

@run.command("cv-job")
@click.option('--cv-file', 'cv_file', type=click.File('r'),
              help="Path to a plaintext CV file (bypass matricula lookup)")
@click.option('--cv-text', 'cv_text', type=str,
              help="Direct CV text input (bypass matricula lookup)")
@click.option("--matricula", 'matricula', type=str,
              help="Student matricula to fetch stored CV (if no text provided)")
@click.option("--fonte", default="VRADM", show_default=True,
              help="Fonte_aluno key for lookup if matricula is used")
@click.option("--skip-fit", is_flag=True,
              help="Skip HDBSCAN prediction; assign by nearest centroid only")
def run_cv_job(matricula: str, fonte: str, cv_file, cv_text, skip_fit: bool):
    """
    Match a CV against existing job clusters, using either direct input
    (--cv-file or --cv-text) or a stored matricula lookup.
    """
    uid = str(uuid.uuid4())
    click.echo("Matching CV to job clusters...")

    # ── 1) Prepare raw CV embedding ─────────────────────────────────────
    embed_func = get_embed_func()
    if cv_file:
        raw_cv = cv_file.read()
    elif cv_text:
        raw_cv = cv_text
    elif matricula:
        from .data.cvs import fetch_single_embedding_cv
        vec = fetch_single_embedding_cv(fonte, matricula)
        if vec is None:
            click.echo(f"❌ No embedding found for ({fonte}, {matricula})", err=True)
            raise click.Abort()
        raw_embedding = np.array(vec, dtype=np.float32)
    else:
        click.echo("❌ Please provide --cv-file, --cv-text, or --matricula", err=True)
        raise click.Abort()

    if 'raw_cv' in locals():
        raw_embedding = np.array(embed_func(raw_cv), dtype=np.float32)

    # ── 2) Encode to latent space ───────────────────────────────────────
    ae = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
    ae.eval()
    x = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0)\
             .to(ae.encoder[0].weight.device)
    with torch.no_grad():
        _, z = ae(x)
    student_latent = z.cpu().numpy().squeeze(0).astype(np.float64)
    student_norm   = normalize(student_latent.reshape(1, -1), norm="l2")[0]

    # ── 3) CV clustering ────────────────────────────────────────────────
    click.echo("  → Predicting CV cluster from text input…")
    with open(CV_CLUSTER_FILE_PATH, 'rb') as f:
        cv_clust = pickle.load(f)
    cv_lbls, _ = hdbscan.approximate_predict(cv_clust, student_norm.reshape(1, -1))
    student_cv_cluster = int(cv_lbls[0])
    click.echo(f"  → CV cluster = {student_cv_cluster}")

    # ── 4) Fetch CV centroid ───────────────────────────────────────────
    conn = psycopg2.connect(POSTGRES_URL) if isinstance(POSTGRES_URL, str) \
           else psycopg2.connect(**POSTGRES_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT centroid FROM cv_cluster_centroids WHERE cluster_id=%s",
        (student_cv_cluster,)
    )
    row = cur.fetchone()
    cur.close(); conn.close()

    if row and student_cv_cluster != -1:
        cv_cent = np.array(
            json.loads(row[0]) if isinstance(row[0], str) else row[0],
            dtype=np.float64
        )
        query_vec = normalize(cv_cent.reshape(1, -1), norm="l2")[0]
        click.echo("  → Using CV centroid for query vector")
    else:
        query_vec = student_norm
        click.echo("  → No CV centroid; using individual CV vector")

    out = match_jobs_pipeline(
        query_vec,
        cv_cluster_file=CV_CLUSTER_FILE_PATH,
        cv_centroids_table="cv_cluster_centroids",
        job_cluster_file=JOB_CLUSTER_FILE_PATH,
        job_centroids_table="job_cluster_centroids",
        job_assignments_table="job_clusters",
        jobs_fetcher=fetch_embeddings_job_with_metadata,
        embedding_col="latent_code",
        skip_fit=skip_fit
    )
    click.echo(json.dumps(out, indent=2))

@deprecated("Não foi desenvolvido para o trabalho da disciplina")
@run.command("cv-feat")
@click.argument('input_file', type=click.Path(exists=True))
def run_cv_feat(input_file):
    """Run CV feature scoring inference."""
    uid = str(uuid.uuid4())
    inp = json.load(open(input_file))
    outp = infer_cv_feat(inp)
    click.echo(outp)
    #log_to_db(uid, "run", "cv-feat", inp, outp)

@deprecated("Não foi desenvolvido para o trabalho da disciplina")
@run.command("job-feat")
@click.argument('input_file', type=click.Path(exists=True))
def run_job_feat(input_file):
    """Run job posting feature scoring inference."""
    uid = str(uuid.uuid4())
    inp = json.load(open(input_file))
    outp = infer_job_feat(inp)
    click.echo(outp)
    #log_to_db(uid, "run", "job-feat", inp, outp)

# ─── EXPLAIN GROUP ────────────────────────────────────────────────────────────
@cli.group()
@deprecated("Não foi desenvolvido para o trabalho da disciplina")
def explain():
    """Explainability commands (using logged runs)."""
    pass

@deprecated("Não foi desenvolvido para o trabalho da disciplina")
@explain.command("why-cv-job")
@click.argument('run_id')
def explain_cv_job_cmd(run_id):
    """Explain a specific CV-job run via feature attributions."""
    explain_cv_job(run_id)

@deprecated("Não foi desenvolvido para o trabalho da disciplina")
@explain.command("why-cv-feat")
@click.argument('run_id')
def explain_cv_feat_cmd(run_id):
    """Explain a specific CV feature scoring run."""
    explain_cv_feat(run_id)

@deprecated("Não foi desenvolvido para o trabalho da disciplina")
@explain.command("why-job-feat")
@click.argument('run_id')
def explain_job_feat_cmd(run_id):
    """Explain a specific job feature scoring run."""
    explain_job_feat(run_id)

if __name__ == "__main__":
    cli()