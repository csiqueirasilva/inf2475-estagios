from .commands.root import cli
from .commands import cluster, plot, compare, sanitize, train, test, run, explain

if __name__ == "__main__":
    cli()