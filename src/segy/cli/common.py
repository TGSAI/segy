"""Common components for the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from typing import Optional

import typer

UriArgument = Annotated[
    str, typer.Argument(help="Valid URI for loading the SEG-Y file.")
]

ListOfIntegersOption = Annotated[list[int], typer.Option(help="List of integers.")]

ListOfFieldNamesOption = Annotated[
    list[str], typer.Option(default_factory=list, help="List of field names.")
]

JsonFileOutOption = Annotated[
    Optional[Path], typer.Option(help="Path for JSON output.")
]

TextFileOutOption = Annotated[
    Optional[Path], typer.Option(help="Path for text output.")
]


def modify_path(
    path: Path, suffix: str, default_extension: str, delimiter: str = "_"
) -> Path:
    """Modify a path with a suffix appended and ensure default extension is honored."""
    new_stem = f"{path.stem}{delimiter}{suffix}"

    if path.suffix:  # If there's an existing extension
        extension = path.suffix
        extension = default_extension if extension != default_extension else extension
        new_name = f"{new_stem}{extension}"
    else:  # If there's no extension
        new_name = f"{new_stem}{default_extension}"

    return path.with_name(new_name)
