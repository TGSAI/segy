"""Command line interface for segy."""
import logging
import sys

import typer
from rich import print
from rich.logging import RichHandler

from segy.cli.common import JsonFileOutOption
from segy.cli.common import ListOfFieldNamesOption
from segy.cli.common import ListOfIntegersOption
from segy.cli.common import TextFileOutOption
from segy.cli.common import UriArgument
from segy.cli.common import modify_path
from segy.exceptions import InvalidFieldError
from segy.exceptions import NonSpecFieldError

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.ERROR,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("segy")

app = typer.Typer(
    name="dump",
    no_args_is_help=True,
    help="Dump SEG-Y file components to screen or file.",
)


@app.command(rich_help_panel="General Commands")
def info(uri: UriArgument, output: JsonFileOutOption = None) -> None:
    """Get basic information about a SEG-Y file."""
    from segy import SegyFile
    from segy.schema.segy import SegyInfo

    segy = SegyFile(uri)
    spec = segy.spec

    info = SegyInfo(
        uri=uri,
        segy_standard=spec.segy_standard,
        num_traces=segy.num_traces,
        samples_per_trace=segy.samples_per_trace,
        sample_interval=segy.sample_interval,
        file_size=segy.file_size,
    )

    info_json = info.model_dump_json(indent=2)

    if output is None:
        print(info_json)

    else:
        output = modify_path(output, suffix="info", default_extension=".json")
        with output.open(mode="w") as f:
            f.write(info_json)
        print(f"Saved SEG-Y info to file {output}")


@app.command(rich_help_panel="Header Commands")
def text_header(uri: UriArgument, output: TextFileOutOption = None) -> None:
    """Print or save the text header of a SEG-Y file."""
    from segy import SegyFile

    segy = SegyFile(uri)
    text = segy.text_header

    if output is None:
        print(text)

    else:
        output = modify_path(output, suffix="text", default_extension=".txt")
        with output.open(mode="w") as f:
            f.write(text)
        print(f"Saved SEG-Y text header to file {output}")


@app.command(rich_help_panel="Header Commands")
def binary_header(uri: UriArgument, output: JsonFileOutOption = None) -> None:
    """Print or save the binary header of a SEG-Y file."""
    from segy import SegyFile

    segy = SegyFile(uri)
    bin_header = segy.binary_header
    bin_header_json = bin_header.to_json()

    if output is None:
        print(bin_header_json)

    else:
        output = modify_path(output, suffix="binary", default_extension=".json")
        with output.open(mode="w") as f:
            f.write(bin_header_json)
        print(f"Saved SEG-Y binary header to file {output}")


@app.command(rich_help_panel="Trace Commands")
def trace(
    uri: UriArgument, index: ListOfIntegersOption, output: JsonFileOutOption = None
) -> None:
    """Get one or more traces with headers."""
    from segy import SegyFile

    segy = SegyFile(uri)

    print(segy.trace[index])

    if output is not None:
        msg = "Trace dump to file is not implemented yet."
        raise NotImplementedError(msg)


@app.command(rich_help_panel="Trace Commands")
def trace_header(
    uri: UriArgument,
    index: ListOfIntegersOption,
    field: ListOfFieldNamesOption,
    output: JsonFileOutOption = None,
) -> None:
    """Get one or more trace's headers (without samples)."""
    from pandas import Index

    from segy import SegyFile

    segy = SegyFile(uri)
    headers = segy.header[index]

    if len(field) > 0:
        try:
            headers = headers[field]
        except (InvalidFieldError, NonSpecFieldError) as e:
            logger.error(e)
            sys.exit(1)

    headers_df = headers.to_dataframe()

    row_index = Index(index, name="trace_index")
    headers_df.set_index(row_index, inplace=True)

    print(headers_df)

    if output is not None:
        msg = "Trace header dump to file is not implemented yet."
        raise NotImplementedError(msg)


@app.command(rich_help_panel="Trace Commands")
def trace_data(
    uri: UriArgument, index: ListOfIntegersOption, output: JsonFileOutOption = None
) -> None:
    """Get one or more trace's samples (without headers)."""
    from segy import SegyFile

    segy = SegyFile(uri)

    print(segy.sample[index])

    if output is not None:
        msg = "Trace data dump to file is not implemented yet."
        raise NotImplementedError(msg)
