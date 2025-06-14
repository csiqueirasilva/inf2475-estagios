import click
from ..constants import DEFAULT_DATABASE_FILE

# ─── ROOT CLI ────────────────────────────────────────────────────────────────
@click.group()
@click.option('--verbose', is_flag=True, help="Enable verbose output")
@click.option('--db-path', 'db_path', default=DEFAULT_DATABASE_FILE, help="Path to the database file")
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
