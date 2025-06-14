import logging
import pickle
import uuid
import json
import sqlite3
import click
import hdbscan
import numpy as np
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from sklearn.metrics.pairwise import cosine_similarity
from .root import cli
from ..data.match import match_jobs_pipeline
from ..utils import deprecated
from ..data.job_autoencoder import JobAutoencoder
from ..constants import CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_AUTOENCODER_FILE_PATH, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_AUTOENCODER_FILE_PATH
from ..data.autoencoder import CVAutoencoder
from ..data.cvs      import fetch_embeddings_cv_with_courses_filtered_with_experience
from ..data.jobs     import fetch_embeddings_job_with_metadata
from ..models.inference       import infer_cv_feat, infer_job_feat
from ..data.db import POSTGRES_URL
from ..data.embed import get_embed_func

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
        from ..data.cvs import fetch_single_embedding_cv
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
        from ..data.cvs import fetch_single_embedding_cv
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
