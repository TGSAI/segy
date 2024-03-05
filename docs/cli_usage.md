# Command-Line Usage

## Introduction

`segy` comes with a useful CLI tool to interrogate SEG-Y files either on disk
or any remote store.

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

For the CLI demos, we will use a public SEG-Y file located in Amazon Web
Services' (AWS) Simple Storage Service (S3), also known as a cloud object
store.

This dataset, the Stratton 3D is made available for worldwide education and training
by the [Bureau of Economic Geology at the University of Texas at Austin][beg].
Available information and data acquisition details are accessible via the
[SEG Wiki][seg wiki].

[seg wiki]: https://wiki.seg.org/wiki/Parihaka-3D
[beg]: https://www.beg.utexas.edu

We will take a look at the 3D unprocessed shot gathers (swath 1).

#### Configuration Options

When accessing public datasets from S3, we need to set
`SegyFileSettings().storage_options = {"anon": True}`{l=python} for anonymous
access. [SegyFileSettings](#SegyFileSettings) exposes all configuration options
as environment variables. We just need to set `storage_options` with the `JSON`
string `{"anon": true}`{l=python}. On Linux you can do this by the command below.
Environment variables can be configured in many ways, please refer to the options
for your specific Operating System (OS).

```shell
export SEGY__STORAGE_OPTIONS='{"anon": true}'
```

#### Basic Info

Now that we can access public S3 buckets anonymously, we can output a basic
summary of the file using the `info` command.

```console
$ segy dump info \
    s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy
{
  "uri": "s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy",
  "segyStandard": 0.0,
  "numTraces": 136530,
  "samplesPerTrace": 3000,
  "sampleInterval": 2000,
  "fileSize": 1671130800
}
```

#### File Text Header

Let's take a look at the text header.

```console
$ segy dump text-header \
    s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy
C 1 CLIENT: BUREAU OF ECONOMIC GEOLOGY  COMPANY: HALLIBURTON GS CREW: #1768
C 2 SURVEY: WARDNER LEASE 3-D (STRATTON FIELD)       AREA: NUECES CO, TEXAS
C 3 RECORDING DATE: 1992
C 4 2MS SAMPLE INTERVAL   3000 SAMPLES/TRACE   4 BYTES/SAMPLE
C 5
C 6 DATA STORED AS SEG-Y FORMAT #1 (IBM FLOATING POINT)
C 7 KEY STANDARD TRACE HEADERS USED:
C 8 FFID =  9-12
C 9 SOURCE X = 73-76  SOURCE Y = 77-80 SOURCE Z = 45-48
C10    REC X = 81-84     REC Y = 85-88    REC Z = 41-44
C11 COORD SCALER = 71-72   ELEV. SCALER = 69-70
C12
C13 NOTE: X = NORTHING, Y = EASTING (RIGHT-HAND Z-DOWN COORDINATES)
C14
C15 NON-STANDARD TRACE HEADERS:
C16 CHANNEL  = 25-28
C17    CHANNELS 1-720 ARE LIVE DATA, 996-999 ARE AUXILIARY TRACES
C18 RECEIVER LINE = 181-184  RECEIVER NUMBER = 185-188
C19 SOURCE LINE = 189-192  SOURCE NUMBER = 193-196
C20
C21 PROCESSED BY EGL: EXPLORATION GEOPHYSICS LABORATORY (PAUL E MURRAY)
C22 BUREAU OF ECONOMIC GEOLOGY, JACKSON SCHOOL OF GEOSCIENCES, UT - AUSTIN
C23 **********PROCESSING**********
C24 1) FIELD SEG-Y FILES REFORMAT TO EGLTOOLS SDF FORMAT
C25 2) CHANNEL RENUMBERING AND GEOMETRY LOADED TO HEADERS
C26 3) PADDED TRACES, BAD AND TEST RECORDS REMOVED
C27 4) REFORMAT TO SEG-Y
C28
C29
C30 SWATH 1 OF 4 CONTAINS FFIDS 1-262
C31
C32 COORDINATES ARE IN FEET
C33 ELLIPSOID: CLARKE 1866
C34 DATUM = NAD27
C35 TEXAS STATE PLANE SOUTH ZONE, LAMBERT PROJECTION
C36 FALSE NORTHING =  485012.85
C37 FALSE EASTING =  2000000.00
C38
C39 written from EGLTools for Matlab on 14-Dec-2009
C40 END EBCDIC
```

#### File Binary Header

```console
$ segy dump binary-header \
    s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy
{
  "job_id": 0,
  "line_no": 0,
  "reel_no": 1,
  "data_traces_ensemble": 724,
  "aux_traces_ensemble": 0,
  "sample_interval": 2000,
  "sample_interval_orig": 2000,
  "samples_per_trace": 3000,
  "samples_per_trace_orig": 3000,
  "data_sample_format": 1,
  "ensemble_fold": 724,
  "trace_sorting": 1,
  "vertical_sum": 0,
  "sweep_freq_start": -30480,
  "sweep_freq_end": -2692,
  "sweep_length": 0,
  "sweep_type": 0,
  "sweep_trace_no": 0,
  "sweep_taper_start": 0,
  "sweep_taper_end": 0,
  "taper_type": 0,
  "correlated_traces": 0,
  "binary_gain": 0,
  "amp_recovery_method": 0,
  "measurement_system": 0,
  "impulse_signal_polarity": 0,
  "vibratory_polarity": 0
}
```

#### Trace Header

This is how we can get three header fields for a few traces.

```console
$ s3://open.source.geoscience/open_data/stratton/segy/navmerged/swath_1_geometry.sgy \
    --index 100 --index 101 --index 500 --index 501 \
    --field src_x --field src_y \
    --field rec_x --field rec_y \
    --field scalar_apply_coords
                src_x      src_y     rec_x      rec_y  scalar_apply_coords
trace_index
100          70628086  219412572  70616707  218875760                 -100
101          70628086  219412572  70616695  218864765                 -100
500          70650057  219412488  70880968  219271571                 -100
501          70650057  219412488  70880940  219260587                 -100
```
