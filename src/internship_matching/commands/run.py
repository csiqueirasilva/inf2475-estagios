from datetime import datetime
import logging
from pathlib import Path
import pickle
import uuid
import json
import sqlite3
import click
import hdbscan
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..data.sharedautoencoder import CVJobSharedAutoencoder
from .root import cli
from ..data.match import get_distance_suggestions_pipeline, match_jobs_pipeline, run_distance_batch
from ..utils import deprecated
from ..data.job_autoencoder import JobAutoencoder
from ..constants import COLUMN_SHARED_LATENT_CODE, CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_AUTOENCODER_FILE_PATH, DEFAULT_PIPELINE_TOP_K_LABELS, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_AUTOENCODER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME
from ..data.autoencoder import CVAutoencoder
from ..data.cvs      import fetch_embeddings_cv_with_courses_filtered_with_experience, fetch_single_embedding_cv
from ..data.jobs     import fetch_embedding_pairs, fetch_embeddings_job_with_metadata
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
@click.option("--top-k", default=DEFAULT_PIPELINE_TOP_K_LABELS, type=int,
              help="Top k results to return in primary cluster")
@click.option("--fonte", default="VRADM", show_default=True,
              help="Fonte_aluno key for lookup if matricula is used")
@click.option("--skip-fit", is_flag=True,
              help="Skip HDBSCAN prediction; assign by nearest centroid only")
def run_cv_nomic_job(matricula: str, fonte: str, cv_file, cv_text, skip_fit: bool, top_k: int):
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
        cv_centroids_table="cv_nomic_centroids",
        job_centroids_table="job_nomic_centroids",
        job_assignments_table="job_nomic_clusters",
        jobs_fetcher=fetch_embeddings_job_with_metadata,
        embedding_col="embedding",
        skip_fit=skip_fit,
        primary_top_k=top_k,
    )
    click.echo(json.dumps(out, indent=2))

@run.command("cv-job")
@click.option('--cv-file', 'cv_file', type=click.File('r'),
              help="Path to a plaintext CV file (bypass matricula lookup)")
@click.option('--cv-text', 'cv_text', type=str,
              help="Direct CV text input (bypass matricula lookup)")
@click.option("--matricula", 'matricula', type=str,
              help="Student matricula to fetch stored CV (if no text provided)")
@click.option("--top-k", default=DEFAULT_PIPELINE_TOP_K_LABELS, type=int,
              help="Top k results to return in primary cluster")
@click.option("--fonte", default="VRADM", show_default=True,
              help="Fonte_aluno key for lookup if matricula is used")
@click.option("--skip-fit", is_flag=True,
              help="Skip HDBSCAN prediction; assign by nearest centroid only")
def run_cv_job(matricula, fonte, cv_file, cv_text, skip_fit, top_k):
    """
    Match a CV against existing job clusters using the *shared* CV–Job
    autoencoder latent space (96-d).
    """
    click.echo("🔍 Matching CV (shared-AE 96-d) to job clusters…")

    # ── 1) Prepare raw CV embedding ─────────────────────────────────────────
    if cv_file:
        raw = cv_file.read()
        emb = get_embed_func()(raw)
    elif cv_text:
        emb = get_embed_func()(cv_text)
    elif matricula:
        emb = fetch_single_embedding_cv(fonte, matricula)
        if emb is None:
            click.echo(f"❌ No embedding found for ({fonte}, {matricula})", err=True)
            raise click.Abort()
    else:
        click.echo("❌ Please provide --cv-file, --cv-text, or --matricula", err=True)
        raise click.Abort()

    raw_embedding = np.array(emb, dtype=np.float32)
    click.echo("  → Raw 768-d embedding ready")

    # ── 2) Encode to **shared** latent space with CVJobSharedAutoencoder ────
    ae = CVJobSharedAutoencoder.load()  # assumes default path inside class
    ae.eval()
    with torch.no_grad():
        # .encode expects a 2-D array, returns (N,96)
        latent = ae.encode(raw_embedding.reshape(1, -1)).squeeze(0)
    query_latent = normalize(latent.reshape(1, -1), norm="l2")[0]
    click.echo("  → Encoded to shared-AE 96-d and normalized")

    # ── 3) Run match_jobs_pipeline in “auto” mode ────────────────────────────
    out = match_jobs_pipeline(
        query_latent,
        cv_cluster_file=CV_CLUSTER_FILE_PATH,
        job_cluster_file=JOB_CLUSTER_FILE_PATH,
        cv_centroids_table="cv_cluster_centroids",
        job_centroids_table="job_cluster_centroids",
        job_assignments_table="job_clusters",
        jobs_fetcher=fetch_embeddings_job_with_metadata,
        embedding_col=COLUMN_SHARED_LATENT_CODE,   # now using shared-AE codes
        primary_top_k=top_k,
        skip_fit=skip_fit,
    )

    click.echo(json.dumps(out, indent=2))

@run.command("cv-raw-job")
@click.option('--cv-file', 'cv_file', type=click.File('r'),
              help="Path to a plaintext CV file (bypass matricula lookup)")
@click.option('--cv-text', 'cv_text', type=str,
              help="Direct CV text input (bypass matricula lookup)")
@click.option("--matricula", 'matricula', type=str,
              help="Student matricula to fetch stored CV (if no text provided)")
@click.option("--fonte", default="VRADM", show_default=True,
              help="Fonte_aluno key for lookup if matricula is used")
@click.option("--top-k", default=DEFAULT_PIPELINE_TOP_K_LABELS, show_default=True,
              help="How many top matches to return")
def run_cv_raw_job(cv_file, cv_text, matricula, fonte, top_k):
    """
    Match a CV against all jobs by raw 768-d cosine similarity
    (no clusters, no autoencoder), but output in the same JSON format
    as cv-job/cv-nomic-job.
    """
    click.echo("Matching CV → Jobs by raw 768-d cosine…")

    # 1) build query embedding
    if cv_file:
        txt = cv_file.read()
        emb = get_embed_func()(txt)
    elif cv_text:
        emb = get_embed_func()(cv_text)
    elif matricula:
        emb = fetch_single_embedding_cv(fonte, matricula)
        if emb is None:
            click.echo(f"❌ No embedding found for ({fonte}, {matricula})", err=True)
            raise click.Abort()
    else:
        click.echo("❌ Please provide --cv-file, --cv-text or --matricula", err=True)
        raise click.Abort()

    qv = normalize(
        np.array(emb, dtype=np.float32).reshape(1, -1),
        norm="l2"
    )[0]
    click.echo("  → Query vector ready and normalized")

    # 2) load jobs + their raw_input
    jobs_df = fetch_embeddings_job_with_metadata()
    job_ids = jobs_df["contract_id"].tolist()
    raws    = jobs_df["raw_input"].tolist()
    mat     = np.vstack(jobs_df[JOBS_EMBEDDING_COLUMN_NAME].values).astype(np.float32)
    mat_n   = normalize(mat, norm="l2")
    click.echo(f"  → Loaded {len(job_ids)} job embeddings")

    # 3) compute and rank
    sims = (mat_n @ qv).tolist()
    ranked = sorted(
        zip(job_ids, sims, raws),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    click.echo(f"  → Picked top {top_k} matches")

    # 4) build JSON result
    out = {
        "student_cv_cluster":    None,
        "student_job_cluster":   None,
        "used_simple_assignment": False,
        "matched_jobs": [
            {
                "cluster_id":  None,
                "contract_id": int(cid),
                "similarity":  float(score),
                "raw_input":   raw_input,
            }
            for cid, score, raw_input in ranked
        ]
    }

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

@run.command("distance-suggestions")
@click.option("-c", "--contract-id", required=True, type=int,
              help="Job contract ID to compare against.")
@click.option("--cv-text", help="CV text directly as input.")
@click.option("--cv-file", type=click.Path(exists=True),
              help="Path to a .txt file with CV text.")
@click.option("--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True,
              help="Embedding space to use.")
@click.option("--top-k", default=10, show_default=True,
              help="Number of tokens for gap analysis.")
def distance_suggestions_cmd(contract_id, cv_text, cv_file, embedding_type, top_k):
    result = get_distance_suggestions_pipeline(
        contract_id,
        cv_text=cv_text,
        cv_file=Path(cv_file) if cv_file else None,
        embedding_type=embedding_type,
        top_k=top_k
    )

    click.echo(f"Original CV:\n{result['cv_text']}\n")
    click.echo(f"Job text:\n{result['job_text']}\n")
    click.echo(f"Initial raw similarity    : {result['initial_similarity']:.4f}")
    click.echo(f"Initial norm similarity   : {result['initial_norm_similarity']:.4f}")
    click.echo(f"Initial scaled similarity : {result['initial_scaled_similarity']:.4f}")
    click.echo(f"Initial gauge             : {result['initial_gauge_pct']}% ({result['initial_gauge_color']})\n")

    click.echo(f"Top CV tokens: {', '.join(result['cv_top_tokens'])}")
    click.echo(f"Top Job tokens: {', '.join(result['job_top_tokens'])}\n")

    click.echo(f"LLM Suggestions:\n{result['suggestions']}\n")

    click.echo(f"Improved CV:\n{result['improved_cv_text']}\n")
    click.echo(f"Improved raw similarity    : {result['improved_similarity']:.4f}")
    click.echo(f"Improved norm similarity   : {result['improved_norm_similarity']:.4f}")
    click.echo(f"Improved scaled similarity : {result['improved_scaled_similarity']:.4f}")
    click.echo(f"Improved gauge             : {result['improved_gauge_pct']}% ({result['improved_gauge_color']})")

@run.command("distance-suggestions-batch")
@click.option("--limit", "-n", default=500, show_default=True, type=int,
              help="Number of job–CV pairs to fetch and process.")
@click.option("--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True,
              help="Embedding space to use.")
@click.option("--top-k", default=10, show_default=True, type=int,
              help="Number of tokens for gap analysis.")
@click.option(
    "--output-csv", "-o",
    default="data/processed/distance_suggestions_batch_DTTIME.csv",
    show_default=True, type=click.Path(),
    help="Path to write the batch results CSV (use DTTIME in name to auto-stamp)."
)
def distance_suggestions_batch_cmd(limit, embedding_type, top_k, output_csv):
    """
    Fetch up to LIMIT job–CV cases, run the pipeline on each, write CSV (with IDs),
    print summary stats (±5-pt gauge margin, best/worst beyond margin), list cases
    that worsened, and emit a LaTeX summary table.
    """
    click.echo(f"Fetching up to {limit} job–CV pairs…")
    df_pairs = fetch_embedding_pairs(limit=limit)

    click.echo(f"Running pipeline over {len(df_pairs)} pairs…")
    df_results = run_distance_batch(
        df_pairs,
        embedding_type=embedding_type.upper(),
        top_k=top_k
    )

    # generate timestamped filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_csv.replace("DTTIME", ts))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # reorder columns: move gauge pct next to contract_id
    front = [
        "fonte_aluno", "matricula", "contract_id",
        "initial_gauge_pct", "improved_gauge_pct"
    ]
    rest = [c for c in df_results.columns if c not in front]
    df_to_save = df_results[front + rest]

    # Save CSV
    df_to_save.to_csv(out_path, index=False)
    click.echo(f"Wrote batch results to {out_path.resolve()}\n")

    # compute gauge diffs and updated summary stats
    gauge_diffs   = df_results["improved_gauge_pct"] - df_results["initial_gauge_pct"]
    total         = len(gauge_diffs)
    improved      = int((gauge_diffs > 5).sum())       # only those > +5 pts
    worse         = int((gauge_diffs < -5).sum())      # only those < -5 pts
    within_margin = int((gauge_diffs.abs() <= 5).sum())

    best_idx   = gauge_diffs.idxmax()
    worst_idx  = gauge_diffs.idxmin()
    best_diff  = gauge_diffs.loc[best_idx]
    worst_diff = gauge_diffs.loc[worst_idx]

    # Pretty-print summary
    click.echo("=== Batch Summary ===")
    click.echo(f"Total cases                  : {total}")
    click.echo(f"Improved beyond +5 pts       : {improved}")
    click.echo(f"Worse beyond -5 pts          : {worse}")
    click.echo(f"Within ±5-point margin       : {within_margin}")
    click.echo(f"Best improvement (pts)       : +{best_diff:.1f}")
    click.echo(f"Worst deterioration (pts)    : {worst_diff:.1f}\n")

    # List worsened cases for inspection (contract_id, initial and improved gauge)
    worsened = df_results[gauge_diffs < -5]
    if not worsened.empty:
        click.echo("Cases worsened beyond -5 pts:")
        for _, row in worsened.iterrows():
            click.echo(
                f" - contract {row['contract_id']}: "
                f"initial {row['initial_gauge_pct']}% → "
                f"improved {row['improved_gauge_pct']}%"
            )
        click.echo()

    # Build & emit LaTeX summary table
    summary_df = pd.DataFrame({
        "Metric": [
            "Total cases",
            "Improved beyond +5 pts",
            "Worse beyond -5 pts",
            "Within ±5-pt margin",
            "Best improvement (pts)",
            "Worst deterioration (pts)"
        ],
        "Value": [
            total,
            improved,
            worse,
            within_margin,
            f"{best_diff:.1f}",
            f"{worst_diff:.1f}"
        ]
    })

    click.echo("LaTeX summary table (paste into your paper):\n")
    latex_summary = summary_df.to_latex(
        index=False,
        column_format="lr",
        caption="Summary of Batch Distance-Suggestion Results",
        label="tab:distance_suggestions_batch_summary"
    )
    click.echo(latex_summary)