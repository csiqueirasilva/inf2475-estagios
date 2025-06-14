import click

from ..data.autoencoder import CVAutoencoder
from ..data.job_autoencoder import JobAutoencoder
from .root import cli
from ..constants import CVS_AUTOENCODER_FILE_PATH, DEFAULT_DATABASE_FILE, JOBS_AUTOENCODER_FILE_PATH
from ..data.cvs      import cv_fill_raw_embeddings, sanitize_input_cvs, store_embeddings_singles_cv
from ..data.jobs     import sanitize_input_jobs, store_embeddings_singles_job
from ..data.db import start_database_import
from ..data.embed import CACHE_DIR, LatentTokenExplainer, get_embed_func, NomicTokenExplainer, tokens_in
from ..data.cvs  import fetch_embeddings_cv_with_courses_filtered_with_experience
from ..data.jobs import fetch_embeddings_job_with_metadata

# ─── SANITIZE GROUP ───────────────────────────────────────────────────────────
@cli.group()
def sanitize():
    """Data sanitization commands."""
    pass

@sanitize.command("latent-token-cache")
@click.option("--batch-size", default=128, show_default=True,
              help="Tokens per forward pass through each AE encoder.")
@click.option("--force", is_flag=True,
              help="Delete existing latent caches before rebuilding.")
def build_latent_token_cache(batch_size: int, force: bool):
    """
    Build *both* latent-space token caches:

      • data/cache/token_lat_cv.npy   (CV autoencoder)
      • data/cache/token_lat_job.npy  (Job autoencoder)

    Uses the shared vocabulary from **all** CV+Job texts.
    """
    click.echo("🔍 Fetching texts from Postgres …")
    cvs_df  = fetch_embeddings_cv_with_courses_filtered_with_experience()
    jobs_df = fetch_embeddings_job_with_metadata()
    vocab = {tok for t in cvs_df["text"]       for tok in tokens_in(t)}
    vocab |= {tok for t in jobs_df["raw_input"] for tok in tokens_in(t)}
    vocab = sorted(vocab)
    click.echo(f"✅ {len(vocab):,} unique tokens collected")

    # Prepare AE encoders
    ae_cv  = CVAutoencoder.load(CVS_AUTOENCODER_FILE_PATH)
    ae_job = JobAutoencoder.load(JOBS_AUTOENCODER_FILE_PATH)

    for name, ae in [("cv", ae_cv), ("job", ae_job)]:
        expl = LatentTokenExplainer(ae.encoder,
                                    batch_size=batch_size,
                                    suffix=name)
        if force:
            expl.clear_cache()
            click.echo(f"♻️  Cleared previous cache for '{name}'.")
        expl.build_cache(vocab)
        click.echo(f"🧠  '{name}' cache ready → "
                   f"{CACHE_DIR / ('token_lat_'+name+'.npy')}")

    click.echo("✅ Latent-space token caches up-to-date.")

@sanitize.command("nomic-token-cache")
@click.option("--batch-size", default=64, show_default=True, help="Tokens per Ollama call.")
@click.option("--force", is_flag=True, help="Rebuild cache even if it looks up-to-date.")
def build_token_cache(batch_size: int, force: bool):
    """
    Embed **all distinct tokens** from every CV and Job using the same
    Nomic model that produced the 768-d vectors, and persist them in
    data/cache/{token_mat.npy, token_arr.txt}.  Subsequent CLI commands
    (e.g. *compare labels*) will reuse this cache for instant look-up.
    """
    click.echo("🔍 Fetching texts from Postgres ...")
    cvs_df  = fetch_embeddings_cv_with_courses_filtered_with_experience()
    jobs_df = fetch_embeddings_job_with_metadata()
    all_texts = cvs_df["text"].tolist() + jobs_df["raw_input"].tolist()
    click.echo(f"✅ Collected {len(all_texts):,} documents (CVs + Jobs)")

    explainer = NomicTokenExplainer(batch_size=batch_size)
    if force:
        explainer.clear_cache()
    explainer.build_vocab_from_texts(all_texts)
    vocab_size = len(explainer._vocab)        # after build()

    click.echo(
        f"🧠 Token cache ready: {vocab_size:,} unique tokens embedded "
        f"and stored in 'data/cache/'."
    )

@sanitize.command()
def embeddings():
    cv_fill_raw_embeddings()

@sanitize.command()
@click.pass_context
def database(ctx):
    """Sanitize and rebuild the database."""
    db_path = ctx.obj.get('db_path', DEFAULT_DATABASE_FILE)
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
    click.echo(f"✅ Generated embeddings for Jobs")