import logging
from pathlib import Path
import uuid
import json
import sqlite3
import click

from .data.autolabeler import CVAutoLabeler

from .data.autoencoder import CVAutoencoder

from .data.cvs      import sanitize_input_cvs, store_embeddings_cv
from .data.jobs     import sanitize_input_jobs
from .training.train_cv_job    import train_cv_job_matching
from .training.train_cv_feat   import train_cv_feature_scoring
from .training.train_job_feat  import train_job_feature_scoring
from .models.explainers       import explain_cv_job, explain_cv_feat, explain_job_feat
from .models.inference       import infer_cv_job, infer_cv_feat, infer_job_feat
from .data.db import init_db, start_database_import
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
    store_embeddings_cv(records, embed_func)
    click.echo(f"✅ Generated embeddings for CVs")

@sanitize.command()
def jobs():
    sanitize_input_jobs()

# ─── TEST GROUP ───────────────────────────────────────────────────────────
@cli.group()
def test():
    """Data test commands."""
    pass

@test.command("cv-autoencoder")
@click.argument("matricula")
@click.option("--fonte", default="VRADM", show_default=True)
def test_autoencoder(matricula, fonte):
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

# ─── TRAIN GROUP ──────────────────────────────────────────────────────────────
@cli.group()
def train():
    """Model training commands."""
    pass

@train.command("cv-autoencoder")
def train_cv_autoencoder():
    CVAutoencoder.train_from_db()
    CVAutoencoder.generate_all_latents()

@train.command("cv-autolabeler")
def train_cv_autolabeler():
    labeler = CVAutoLabeler()
    labeler.fit_kmeans()
    labeler.name_clusters_keybert()
    labeler.save_keybert()
    labeler.name_clusters_tfidf()
    labeler.save_tfidf()
    labeler.name_clusters_ctfidf()
    labeler.save_ctfidf()

@train.command("cv-job")
def train_cv_job():
    train_cv_job_matching()

@train.command("cv-feat")
def train_cv_feat():
    train_cv_feature_scoring()

@train.command("job-feat")
def train_job_feat():
    train_job_feature_scoring()

# ─── RUN GROUP ────────────────────────────────────────────────────────────────
@cli.group()
def run():
    """Inference (run) commands."""
    pass

def log_to_db(uuid_str: str, group: str, command: str, inp: dict, outp: dict, logger: logging.Logger):
    """Append this run’s input/output to a local SQLite audit log."""
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

@run.command("cv-job")
@click.argument('input_file', type=click.Path(exists=True))
def run_cv_job(input_file):
    """Run CV-job matching inference."""
    uid = str(uuid.uuid4())
    with open(input_file) as f:
        inp = json.load(f)
    # assume infer_cv_job returns a dict
    outp = infer_cv_job(inp)  
    click.echo(outp)
    #log_to_db(uid, "run", "cv-job", inp, outp)

@run.command("cv-feat")
@click.argument('input_file', type=click.Path(exists=True))
def run_cv_feat(input_file):
    """Run CV feature scoring inference."""
    uid = str(uuid.uuid4())
    inp = json.load(open(input_file))
    outp = infer_cv_feat(inp)
    click.echo(outp)
    #log_to_db(uid, "run", "cv-feat", inp, outp)

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
def explain():
    """Explainability commands (using logged runs)."""
    pass

@explain.command("cv-job")
@click.argument('run_id')
def explain_cv_job_cmd(run_id):
    """Explain a specific CV-job run via feature attributions."""
    explain_cv_job(run_id)

@explain.command("cv-feat")
@click.argument('run_id')
def explain_cv_feat_cmd(run_id):
    """Explain a specific CV feature scoring run."""
    explain_cv_feat(run_id)

@explain.command("job-feat")
@click.argument('run_id')
def explain_job_feat_cmd(run_id):
    """Explain a specific job feature scoring run."""
    explain_job_feat(run_id)

if __name__ == "__main__":
    cli()