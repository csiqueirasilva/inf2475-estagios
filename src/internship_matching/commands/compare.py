# commands/compare.py   (only the command body shown)

import json, math, numpy as np, pandas as pd
from collections import Counter
from pathlib import Path
from typing import Literal

import click
from sklearn.metrics.pairwise import cosine_similarity

from ..data.autoencoder import CVAutoencoder
from ..data.job_autoencoder import JobAutoencoder

from .root import cli
from ..constants import (
    CVS_EMBEDDING_COLUMN_NAME,
    DEFAULT_TOP_K_LABELS,
    JOBS_EMBEDDING_COLUMN_NAME,
    COLUMN_LATENT_CODE,                        # ← "latent_code"
    CVS_AUTOENCODER_FILE_PATH,
    JOBS_AUTOENCODER_FILE_PATH,
)
from ..data.cvs  import fetch_embeddings_cv_with_courses_filtered_with_experience
from ..data.jobs import fetch_embeddings_job_with_metadata
from ..data.embed import NomicTokenExplainer, LatentTokenExplainer

@cli.group()
def compare() -> None:
    """Cross-entity comparisons (CV ↔ Job)."""
    ...

# ──────────────────────────────────────────────────────────────────────────
@compare.command("labels")
@click.option("-f", "--fonte-aluno", default="VRADM", show_default=True,
              help="Fonte_aluno do CV (default VRADM).")
@click.option("-m", "--matricula",    required=True,
              help="Matrícula (7 dígitos) do CV.")
@click.option("-c", "--contract-id",  required=True, type=int,
              help="Primary key do Job (contract_id).")
@click.option("-t", "--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True,
              help="Espaço de embedding a usar.")
@click.option("--top-n",  default=DEFAULT_TOP_K_LABELS, show_default=True,
              help="Qtde de tokens a exibir por documento.")
@click.option("--output", type=click.Path(path_type=Path),
              help="Se fornecido, grava JSON com as listas de tokens.")
def compare_labels(
    fonte_aluno: str,
    matricula:   str,
    contract_id: int,
    embedding_type: Literal["NOMIC", "AUTOENCODE"],
    top_n: int,
    output: Path | None,
):
    # ─── 1) Fetch CV & Job rows ────────────────────────────────────────────
    cvs  = fetch_embeddings_cv_with_courses_filtered_with_experience()
    jobs = fetch_embeddings_job_with_metadata()

    try:
        cv_row  = cvs.set_index(["fonte_aluno", "matricula"]).loc[(fonte_aluno, matricula)]
    except KeyError:
        raise click.ClickException("CV não encontrado.")

    try:
        job_row = jobs.set_index("contract_id").loc[contract_id]
    except KeyError:
        raise click.ClickException("Job contract_id não encontrado.")

    # ─── 2) Pick embedding columns & vectors ───────────────────────────────
    if embedding_type.upper() == "NOMIC":
        cv_vec  = np.asarray(cv_row[CVS_EMBEDDING_COLUMN_NAME],  dtype=np.float32)
        job_vec = np.asarray(job_row[JOBS_EMBEDDING_COLUMN_NAME], dtype=np.float32)
    else:                                   # AUTOENCODE
        cv_vec  = np.asarray(cv_row[COLUMN_LATENT_CODE],  dtype=np.float32)
        job_vec = np.asarray(job_row[COLUMN_LATENT_CODE], dtype=np.float32)

    # ─── 3) Cosine similarity ------------------------------------------------
    sim  = float(cosine_similarity(cv_vec.reshape(1,-1), job_vec.reshape(1,-1))[0,0])
    dist = 1.0 - sim
    click.secho(f"\nCosine similarity : {sim:.4f}",   fg="magenta")
    click.secho(f"Distance (1-sim)    : {dist:.4f}", fg="magenta")

    # ─── 4) Token explanation via caches ------------------------------------
    if embedding_type.upper() == "NOMIC":
        expl = NomicTokenExplainer()                        # uses cached 768-d matrix
        # tiny per-call vocab ⇒ instant (no new Ollama hits)
        expl.build_vocab_from_texts([cv_row["text"], job_row["raw_input"]])
        cv_terms  = expl.nearest_tokens(cv_vec,  k=top_n)
        job_terms = expl.nearest_tokens(job_vec, k=top_n)

    else:  # AUTOENCODE  → use latent caches
        # load encoders once; caches already on disk
        ae_cv  = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
        ae_job = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)

        expl_cv  = LatentTokenExplainer(ae_cv.encoder,  suffix="cv")
        expl_job = LatentTokenExplainer(ae_job.encoder, suffix="job")

        # build_vocab_from_texts is not needed – caches exist
        cv_terms  = expl_cv.nearest_tokens(cv_vec,  k=top_n)
        job_terms = expl_job.nearest_tokens(job_vec, k=top_n)

    overlap = sorted(set(cv_terms) & set(job_terms))

    # ─── 5) Pretty-print -----------------------------------------------------
    click.secho("\nCV  top tokens :",  fg="cyan");  click.echo(" / ".join(cv_terms))
    click.secho("\nJob top tokens :", fg="green"); click.echo(" / ".join(job_terms))
    click.secho(f"\nOverlap ({len(overlap)}):", fg="yellow")
    click.echo(" / ".join(overlap) if overlap else "—")

    # ─── 6) Optional JSON dump ---------------------------------------------
    if output:
        payload = {
            "cv_top": cv_terms, "job_top": job_terms, "overlap": overlap,
            "fonte_aluno": fonte_aluno, "matricula": matricula,
            "contract_id": contract_id, "embedding_type": embedding_type,
            "similarity": sim, "distance": dist,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        click.echo(f"\n✅ JSON salvo em {output}")
