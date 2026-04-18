"""``mlxz`` CLI entry point.

Registers all subcommands under a single :class:`typer.Typer` app.
The ``app`` callable is referenced by ``pyproject.toml`` as the console
script entry point.
"""

from __future__ import annotations

import typer

from mlxz.cli.doctor import doctor
from mlxz.cli.serve import serve

app = typer.Typer(
    name="mlxz",
    help="High-throughput local inference server for Apple Silicon.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

app.command()(doctor)
app.command()(serve)


@app.command()
def bench() -> None:
    """Run benchmarks (not yet implemented)."""
    typer.echo("bench: not yet implemented (Phase 0)")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
