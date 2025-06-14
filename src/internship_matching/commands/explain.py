import click
from .root import cli
from ..utils import deprecated
from ..models.explainers import explain_cv_job, explain_cv_feat, explain_job_feat

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