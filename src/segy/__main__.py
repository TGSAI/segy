"""Command line interface for segy."""


from pathlib import Path
from typing import Annotated
from typing import Optional
from typing import TypeAlias

import typer
from rich import print

from segy import SegyFile
from segy.schema.segy import SegyInfo

BYTE_TO_MB = 1024**2


app = typer.Typer(no_args_is_help=True)

UriArgumentType: TypeAlias = Annotated[
    str, typer.Argument(help="URI of the segy file to load.")
]
SaveJsonType: TypeAlias = Annotated[
    Optional[Path],
    typer.Option(help="Path to output information as JSON."),
]
SaveTextType: TypeAlias = Annotated[
    Optional[Path], typer.Option(help="Path to output information as plain text.")
]


@app.command()
def info(uri: UriArgumentType, save: SaveJsonType = None) -> None:
    """Get basic information about a SEG-Y file."""
    segy = SegyFile(uri)
    spec = segy.spec

    info = SegyInfo(
        uri=uri,
        segy_standard=spec.segy_standard.value,
        num_traces=segy.num_traces,
        samples_per_trace=segy.binary_header["samples_per_trace"],
        sample_interval=segy.binary_header["sample_interval"],
        file_size=segy.file_size / BYTE_TO_MB,
    )

    info_json = info.model_dump_json(indent=2)
    print(info_json)

    if save is not None:
        if save.suffix == "":
            save = save.with_suffix(".json")

        with save.open(mode="w") as f:
            f.write(info_json)


@app.command()
def text(uri: UriArgumentType, save: SaveTextType = None) -> None:
    """Print or save the text header of a SEG-Y file."""
    segy = SegyFile(uri)
    text = segy.text_header

    print(text)

    if save is not None:
        if save.suffix == "":
            save = save.with_suffix(".txt")

        with save.open(mode="w") as f:
            f.write(text)
