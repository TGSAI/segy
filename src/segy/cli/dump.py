"""Command line interface for segy."""

import json

import pandas as pd
import typer
from rich import print

from segy import SegyFile
from segy.cli.common import JsonFileOutOption
from segy.cli.common import ListOfFieldNamesOption
from segy.cli.common import ListOfIntegersOption
from segy.cli.common import TextFileOutOption
from segy.cli.common import UriArgument
from segy.cli.common import modify_path
from segy.schema.segy import SegyInfo

app = typer.Typer(
    name="dump",
    no_args_is_help=True,
    help="Dump SEG-Y file components to screen or file.",
)


@app.command(rich_help_panel="General Commands")
def info(uri: UriArgument, output: JsonFileOutOption = None) -> None:
    """Get basic information about a SEG-Y file."""
    segy = SegyFile(uri)
    spec = segy.spec

    info = SegyInfo(
        uri=uri,
        segy_standard=spec.segy_standard.value,
        num_traces=segy.num_traces,
        samples_per_trace=segy.binary_header["samples_per_trace"].iloc[0],
        sample_interval=segy.binary_header["sample_interval"].iloc[0],
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
    segy = SegyFile(uri)
    bin_header = segy.binary_header
    bin_header_json = bin_header.to_json(orient="records")

    # Extract the first element since we know it's a single row
    parsed_json = json.loads(bin_header_json)
    bin_header_json = json.dumps(parsed_json[0], indent=2)

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
    segy = SegyFile(uri)
    headers = segy.header[index]

    row_index = pd.Index(index, name="trace_index")
    headers.set_index(row_index, inplace=True)

    if len(field) > 0:
        headers = headers[field]

    print(headers)

    if output is not None:
        msg = "Trace header dump to file is not implemented yet."
        raise NotImplementedError(msg)


@app.command(rich_help_panel="Trace Commands")
def trace_data(
    uri: UriArgument, index: ListOfIntegersOption, output: JsonFileOutOption = None
) -> None:
    """Get one or more trace's samples (without headers)."""
    segy = SegyFile(uri)

    print(segy.data[index])

    if output is not None:
        msg = "Trace data dump to file is not implemented yet."
        raise NotImplementedError(msg)
