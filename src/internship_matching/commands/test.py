from pathlib import Path
import sys
import textwrap
from typing import Literal
import click
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..constants import COLUMN_LATENT_CODE, CVS_AUTOENCODER_FILE_PATH, CVS_EMBEDDING_COLUMN_NAME, DEFAULT_TOP_K_LABELS, JOBS_AUTOENCODER_FILE_PATH, JOBS_EMBEDDING_COLUMN_NAME
from ..data.embed import LatentTokenExplainer, NomicTokenExplainer, get_embed_func
from ..data.job_autoencoder import JobAutoencoder
from ..data.jobs import TORCH_DEVICE, fetch_embeddings_job_with_metadata
from .root import cli
from ..utils import deprecated
from ..data.autolabeler import CVAutoLabeler
from ..data.autoencoder import CVAutoencoder
from ..data.cvs      import fetch_embeddings_cv, fetch_embeddings_cv_with_courses_filtered_with_experience

# ── helper that maps type → vector + explainer ────────────────────────────
def _vec_and_explainer(row, which: Literal["cv", "job"], emb_type: str):
    if emb_type.upper() == "NOMIC":
        vec = np.asarray(
            row[CVS_EMBEDDING_COLUMN_NAME if which == "cv" else JOBS_EMBEDDING_COLUMN_NAME],
            dtype=np.float32,
        )
        expl = NomicTokenExplainer()   # cached 768-d matrix auto-loads
        expl.build_vocab_from_texts([row.get("text", ""), row.get("raw_input", "")])
    else:  # AUTOENCODE
        vec = np.asarray(row[COLUMN_LATENT_CODE], dtype=np.float32)
        suffix = "cv" if which == "cv" else "job"
        ae = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH) if which == "cv" \
             else JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)
        expl = LatentTokenExplainer(ae.encoder, suffix=suffix)  # lazy-loads latent cache
    return vec, expl

# ─── TEST GROUP ───────────────────────────────────────────────────────────
@cli.group()
def test():
    """Data test commands."""
    pass

# ---------------------------------------------------------------------------
# Gap-analysis: what’s missing in the CV to fit the Job?
# ---------------------------------------------------------------------------
@test.command("gap-analysis")
@click.option("-m", "--matricula",    required=True)
@click.option("-f", "--fonte-aluno", default="VRADM", show_default=True)
@click.option("-c", "--contract-id", required=True, type=int)
@click.option("-t", "--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True)
@click.option("--top-k", default=15, show_default=True,
              help="Máximo de tokens a enviar ao LLM.")
def gap_analysis(matricula, fonte_aluno, contract_id,
                 embedding_type, top_k):
    """
    Pede ao LLM sugestões de como o CV pode se aproximar da vaga escolhida.
    """
    # ─── fetch rows ───────────────────────────────────────────────────────
    cvs  = fetch_embeddings_cv_with_courses_filtered_with_experience()
    jobs = fetch_embeddings_job_with_metadata()
    try:
        cv_row  = cvs.set_index(["fonte_aluno", "matricula"]).loc[(fonte_aluno, matricula)]
        job_row = jobs.set_index("contract_id").loc[contract_id]
    except KeyError as e:
        raise click.ClickException(str(e))

    cv_text  = cv_row["text"]
    job_text = job_row["raw_input"]

    # ─── get model-faithful tokens in requested space ─────────────────────
    if embedding_type.upper() == "NOMIC":
        expl = NomicTokenExplainer()
        expl.build_vocab_from_texts([cv_text, job_text])
        cv_tokens  = expl.nearest_tokens(np.asarray(cv_row[CVS_EMBEDDING_COLUMN_NAME]),  k=50)
        job_tokens = expl.nearest_tokens(np.asarray(job_row[JOBS_EMBEDDING_COLUMN_NAME]), k=50)
    else:
        # latent space
        ae_cv  = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
        ae_job = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)
        expl_cv  = LatentTokenExplainer(ae_cv.encoder,  suffix="cv")
        expl_job = LatentTokenExplainer(ae_job.encoder, suffix="job")
        cv_tokens  = expl_cv.nearest_tokens(np.asarray(cv_row[COLUMN_LATENT_CODE]),  k=50)
        job_tokens = expl_job.nearest_tokens(np.asarray(job_row[COLUMN_LATENT_CODE]), k=50)

    missing = [tok for tok in job_tokens if tok not in cv_tokens]

    # ─── call the LLM helper ──────────────────────────────────────────────
    from ..data.embed import suggest_cv_improvements    # local import to avoid cycles
    suggestions = suggest_cv_improvements(
        cv_text=cv_text,
        job_text=job_text,
        missing_tokens=missing,
        top_k=top_k
    )

    # ─── display ──────────────────────────────────────────────────────────
    click.secho("📝 Sugestões do LLM:", fg="yellow")
    click.echo(suggestions)

# ===========================================================================
#  NEW: inspect a single CV embedding
# ===========================================================================
@test.command("cv-query")
@click.option("-m", "--matricula",    required=True)
@click.option("-f", "--fonte-aluno", default="VRADM", show_default=True)
@click.option("-t", "--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True)
@click.option("--top-k", default=DEFAULT_TOP_K_LABELS, show_default=True,
              help="Qtde de tokens a exibir.")
def query_cv(matricula, fonte_aluno, embedding_type, top_k):
    cvs = fetch_embeddings_cv_with_courses_filtered_with_experience()
    try:
        row = cvs.set_index(["fonte_aluno", "matricula"]).loc[(fonte_aluno, matricula)]
    except KeyError:
        raise click.ClickException("CV não encontrado.")

    vec, expl = _vec_and_explainer(row, "cv", embedding_type)
    tokens = expl.nearest_tokens(vec, k=top_k)

    click.secho(f"📄 CV ({fonte_aluno}, {matricula}) — raw text:", fg="cyan")
    click.echo(row["text"])
    click.echo()

    click.secho(f"🔑 Top-{top_k} tokens ({embedding_type.upper()}):", fg="yellow")
    click.echo(" / ".join(tokens))

    click.secho("\n🔬 Embedding preview:", fg="magenta")
    click.echo(", ".join(f"{v:.4f}" for v in vec[:8]) + (" …" if len(vec) > 8 else ""))


# ===========================================================================
#  NEW: inspect a single Job embedding
# ===========================================================================
@test.command("job-query")
@click.option("-c", "--contract-id",  required=True, type=int)
@click.option("-t", "--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True)
@click.option("--top-k", default=DEFAULT_TOP_K_LABELS, show_default=True)
def query_job(contract_id, embedding_type, top_k):
    jobs = fetch_embeddings_job_with_metadata()
    try:
        row = jobs.set_index("contract_id").loc[contract_id]
    except KeyError:
        raise click.ClickException("Job contract_id não encontrado.")

    vec, expl = _vec_and_explainer(row, "job", embedding_type)
    tokens = expl.nearest_tokens(vec, k=top_k)

    click.secho(f"📄 Job {contract_id} — raw_input:", fg="green")
    click.echo(row["raw_input"])
    click.echo()

    click.secho(f"🔑 Top-{top_k} tokens ({embedding_type.upper()}):", fg="yellow")
    click.echo(" / ".join(tokens))

    click.secho("\n🔬 Embedding preview:", fg="magenta")
    click.echo(", ".join(f"{v:.4f}" for v in vec[:8]) + (" …" if len(vec) > 8 else ""))

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
@deprecated("teste antigo, descontinuado")
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

# ---------------------------------------------------------------------------
# On-the-fly CV vs. Job similarity, no DB writes
# ---------------------------------------------------------------------------
@test.command("cv-simulate")
@click.option("-c", "--contract-id", required=True, type=int,
              help="Job contract_id usado na comparação.")
@click.option("-t", "--embedding-type",
              type=click.Choice(["NOMIC", "AUTOENCODE"], case_sensitive=False),
              default="NOMIC", show_default=True,
              help="Espaço onde a similaridade será medida.")
@click.option("--cv-text", help="Novo texto do CV (pode conter quebras de linha).")
@click.option("--cv-file", type=click.Path(exists=True, dir_okay=False),
              help="Arquivo .txt com o novo texto do CV.")
@click.option("--top-k", default=10, show_default=True,
              help="Qtde de tokens a exibir.")
def cv_simulate(contract_id, embedding_type, cv_text, cv_file, top_k):
    """
    Calcula **sem gravar no banco** a similaridade do CV revisado
    (passado por --cv-text/--cv-file ou stdin) com a vaga escolhida.
    """

    # 1)  Obtain new CV text -----------------------------------------------
    if not cv_text and not cv_file and sys.stdin.isatty():
        raise click.UsageError("Forneça --cv-text, --cv-file ou envie texto via stdin.")
    if not cv_text:
        cv_text = Path(cv_file).read_text(encoding="utf-8") if cv_file else sys.stdin.read()

    # 2)  Load the job row --------------------------------------------------
    jobs = fetch_embeddings_job_with_metadata()
    try:
        job_row = jobs.set_index("contract_id").loc[contract_id]
    except KeyError:
        raise click.ClickException("Job contract_id não encontrado.")

    # 3)  Get job vector & appropriate explainer ---------------------------
    job_vec_768 = np.asarray(job_row[JOBS_EMBEDDING_COLUMN_NAME], dtype=np.float32)

    if embedding_type.upper() == "NOMIC":
        # ----- embed the new CV (768-d) on the fly
        cv_vec_768 = np.asarray(get_embed_func()(cv_text)[0], dtype=np.float32)

        # similarity
        sim  = float(cosine_similarity(cv_vec_768.reshape(1, -1),
                                       job_vec_768.reshape(1, -1))[0, 0])

        # token explainers
        exp = NomicTokenExplainer()
        exp.build_vocab_from_texts([cv_text, job_row["raw_input"]])
        cv_tokens  = exp.nearest_tokens(cv_vec_768,  k=top_k)
        job_tokens = exp.nearest_tokens(job_vec_768, k=top_k)

    else:  # AUTOENCODE
        # ----- embed in 768-d then encode → 96-d latent
        cv_vec_768 = np.asarray(get_embed_func()(cv_text)[0], dtype=np.float32)
        ae_cv  = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
        ae_job = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)

        with torch.no_grad():
            cv_lat  = ae_cv.encoder(torch.tensor(cv_vec_768, device=TORCH_DEVICE)).squeeze().cpu().numpy()
            job_lat = ae_job.encoder(torch.tensor(job_vec_768, device=TORCH_DEVICE)).squeeze().cpu().numpy()

        sim = float(cosine_similarity(cv_lat.reshape(1, -1),
                                      job_lat.reshape(1, -1))[0, 0])

        exp_cv  = LatentTokenExplainer(ae_cv.encoder,  suffix="cv")
        exp_job = LatentTokenExplainer(ae_job.encoder, suffix="job")
        cv_tokens  = exp_cv.nearest_tokens(cv_lat,  k=top_k)
        job_tokens = exp_job.nearest_tokens(job_lat, k=top_k)

    # 4)  Display results ---------------------------------------------------
    dist = 1.0 - sim
    click.secho(f"\nCosine similarity : {sim:.4f}", fg="magenta")
    click.secho(f"Distance (1-sim)    : {dist:.4f}\n", fg="magenta")

    click.secho("CV  top tokens :",  fg="cyan");  click.echo(" / ".join(cv_tokens))
    click.secho("Job top tokens :", fg="green"); click.echo(" / ".join(job_tokens))

    overlap = sorted(set(cv_tokens) & set(job_tokens))
    click.secho(f"Overlap ({len(overlap)}):", fg="yellow")
    click.echo(" / ".join(overlap) if overlap else "—")

