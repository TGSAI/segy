# Command-Line Usage

## Introduction

`segy` comes with a useful CLI tool to interrogate SEG-Y files either on disk
or any remote store.

In the [cli reference] section, you can see all the options.

## Command Line Usage

SEG-Y provides a convenient command-line-interface (CLI) to do
various tasks.

For each command / subcommand you can provide `--help` argument to
get information about usage.

At the highest level, the `segy` command line offers various options
to choose from. Below you can see the usage for the main entry point.

```{eval-rst}
.. typer:: segy.cli.segy:app
    :prog: segy
    :width: 90
    :theme: dark
    :preferred: svg
```

### Dumping Data

When we use `segy dump` subcommand, we have some options to choose from.
As usual, the `uri` (local or remote paths) will allow us to use the same
toolkit for local and cloud / web files.

```{eval-rst}
.. typer:: segy.cli.segy:app:dump
    :width: 90
    :theme: dark
    :preferred: svg
```

For instance, we can output a basic summary of the file using the `info`
command.

```console
$ segy dump info path/to/seismic.segy

{
  "uri": "path/to/seismic.segy",
  "segyStandard": 0.0,
  "numTraces": 17367161,
  "samplesPerTrace": 1501,
  "sampleInterval": 4000,
  "fileSize": 103416.97395706177
}
```

This is how we can get three header fields for a few traces.

```console
$ segy dump trace-header "path/to/seismic.segy" \
    --index 0 --index 5 --index 101 --index 12001 \
    --field trace_seq_line --field trace_no_field_rec
             trace_seq_line     src_x      src_y

trace_index
0                         1  41613223  844759437
5                         6  41608435  844763454
101                     102  41516509  844840591
12001                  1896  39801062  846284951
```
