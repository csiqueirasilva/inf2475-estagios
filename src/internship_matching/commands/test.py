import click
from .root import cli
from ..utils import deprecated
from ..data.autolabeler import CVAutoLabeler
from ..data.autoencoder import CVAutoencoder
from ..data.cvs      import fetch_embeddings_cv

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
