from .commands.root import cli
from .commands import cluster, plot, compare, sanitize, train, test, run, explain, plot_simulate

if __name__ == "__main__":
    cli()