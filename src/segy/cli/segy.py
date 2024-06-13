"""Main entrypoint to the segy cli."""

import typer

from segy.cli import dump as dump_command

app = typer.Typer(
    name="segy", no_args_is_help=True, pretty_exceptions_show_locals=False
)

app.add_typer(dump_command.app)
