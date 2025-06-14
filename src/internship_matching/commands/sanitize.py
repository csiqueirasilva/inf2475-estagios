import click
from .root import cli
from ..constants import DEFAULT_DATABASE_FILE
from ..data.cvs      import cv_fill_raw_embeddings, sanitize_input_cvs, store_embeddings_singles_cv
from ..data.jobs     import sanitize_input_jobs, store_embeddings_singles_job
from ..data.db import start_database_import
from ..data.embed import get_embed_func

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
    click.echo(f"✅ Generated embeddings for CVs")