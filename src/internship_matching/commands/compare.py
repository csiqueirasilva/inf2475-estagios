from datetime import datetime
import math
from pathlib import Path
import pickle
import json
import re
from typing import Counter, List
import click
import hdbscan
import numpy as np
from sklearn.preprocessing import normalize
from ..data.autolabeler import PT_STOPWORDS
from .root import cli
from ..utils import deprecated
from ..data.clusters import store_clusters_and_centroids, sweep_hdbscan
from ..constants import CLUSTERER_PATHS, CV_CLUSTER_FILE_PATH, CV_NOMIC_CLUSTER_FILE_PATH, CVS_EMBEDDING_COLUMN_NAME, JOB_CLUSTER_FILE_PATH, JOB_NOMIC_CLUSTER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME
from ..data.cvs      import fetch_embeddings_cv_with_courses_filtered_with_experience
from ..data.jobs     import fetch_embeddings_job_with_metadata
from ..data.db import POSTGRES_URL
from ..data.plot import word_re

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _doc_tokens(text: str) -> List[str]:
    """Tokenise → lower → stop-word / digit filter."""
    return [
        w.lower()
        for w in word_re.findall(text)
        if w.lower() not in PT_STOPWORDS and not re.search(r"\d", w)
    ]

def _top_terms_for_doc(doc_text: str,
                       df_counts: Counter,
                       total_docs: int,
                       top_n: int = 10) -> List[str]:
    tokens = _doc_tokens(doc_text)
    if not tokens:
        return []
    tf = Counter(tokens)
    scores = {
        term: (cnt / len(tokens)) * math.log(total_docs / (1 + df_counts.get(term, 0)))
        for term, cnt in tf.items()
    }
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


# --------------------------------------------------------------------------- #
# Click group & command
# --------------------------------------------------------------------------- #
@cli.group()
def compare() -> None:
    """Cross-entity comparisons (CV ↔ Job)."""
    pass

@compare.command("labels")
@click.option(
    "--fonte-aluno",
    "-f",
    default="VRADM",
    show_default=True,
    help="Fonte_aluno do CV (default VRADM).",
)
@click.option(
    "--matricula",
    "-m",
    required=True,
    help="Matrícula (7 dígitos) do CV.",
)
@click.option(
    "--contract-id",
    "-c",
    required=True,
    help="Primary key do Job (contract_id).",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    show_default=True,
    help="Qtde de termos TF-IDF a retornar.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Opcional: caminho para JSON com as três listas de termos.",
)
def compare_labels(
    fonte_aluno: str,
    matricula: str,
    contract_id: str,
    top_n: int,
    output: Path | None,
):
    """
    Compara os **top-N termos TF-IDF** de um CV e de um Job, exibindo também a intersecção.
    Texto usado:

      • CVs  → `llm_parsed_raw_input`
      • Jobs → `raw_input`
    """
    # ------------------------------------------------------------- fetch rows
    cvs  = fetch_embeddings_cv_with_courses_filtered_with_experience(
        embedding_column=CVS_EMBEDDING_COLUMN_NAME
    )
    jobs = fetch_embeddings_job_with_metadata()

    try:
        cv_row = (
            cvs.set_index(["fonte_aluno", "matricula"])
            .loc[(fonte_aluno, matricula)]
        )
    except KeyError:
        raise click.ClickException(
            f"CV não encontrado (fonte_aluno={fonte_aluno!r}, "
            f"matricula={matricula!r})."
        )

    try:
        job_row = jobs.set_index("contract_id").loc[contract_id]
    except KeyError:
        raise click.ClickException(f"Job contract_id '{contract_id}' não encontrado.")

    # -------------------------------------------------------- corpus DF counts
    df_counts: Counter = Counter()
    all_texts = (
        cvs["llm_parsed_raw_input"].astype(str)
        .append(jobs["raw_input"].astype(str), ignore_index=True)
    )
    total_docs = len(all_texts)
    for doc in all_texts:
        df_counts.update(set(_doc_tokens(doc)))

    # ---------------------------------------------------- per-doc TF-IDF terms
    cv_terms  = _top_terms_for_doc(
        cv_row["llm_parsed_raw_input"], df_counts, total_docs, top_n
    )
    job_terms = _top_terms_for_doc(
        job_row["raw_input"], df_counts, total_docs, top_n
    )
    overlap   = sorted(set(cv_terms) & set(job_terms))

    # ----------------------------------------------------------- pretty print
    click.secho("\nCV  top terms :", fg="cyan")
    click.echo(" / ".join(cv_terms) if cv_terms else "—")

    click.secho("\nJob top terms :", fg="green")
    click.echo(" / ".join(job_terms) if job_terms else "—")

    click.secho(f"\nOverlap ({len(overlap)}):", fg="yellow")
    click.echo(" / ".join(overlap) if overlap else "—")

    # ----------------------------------------------------------- optional dump
    if output:
        payload = {
            "cv_top": cv_terms,
            "job_top": job_terms,
            "overlap": overlap,
            "fonte_aluno": fonte_aluno,
            "matricula": matricula,
            "contract_id": contract_id,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        click.echo(f"\n✅ JSON salvo em {output}")
