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

For the CLI demos, we will use a public SEG-Y file over HTTPS.

This file is one shot record from the Soda Lake 2010 3D/3C survey on the
[Geothermal Data Repository][gdr] ([CC-BY-4.0][cc-by]).

[gdr]: https://gdr.openei.org/submissions/1655
[cc-by]: https://creativecommons.org/licenses/by/4.0/

We will take a look at field record `F7733R1.SGY` (shot 7733).
No credentials and no `storage_options` are required. `{"anon": true}` is an
S3 option and breaks this HTTPS URL.

#### Basic Info

We can output a basic summary of the file using the `info` command.

```console
$ segy dump info \
    https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY
{
  "uri": "https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY",
  "segyStandard": 0.0,
  "numTraces": 958,
  "samplesPerTrace": 2000,
  "sampleInterval": 2000,
  "fileSize": 7897520
}
```

#### File Text Header

Let's take a look at the text header.

```console
$ segy dump text-header \
    https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY
C 1 CLIENT                        COMPANY                       CREW NO
C 2 LINE            AREA                        MAP ID
C 3 REEL NO           DAY-START OF REEL     YEAR      OBSERVER
C 4 INSTRUMENT: MFG            MODEL            SERIAL NO
C 5 DATA TRACES/RECORD        AUXILIARY TRACES/RECORD         CDP FOLD
C 6 SAMPLE INTERVAL         SAMPLES/TRACE       BITS/IN      BYTES/SAMPLE
C 7 RECORDING FORMAT        FORMAT THIS REEL        MEASUREMENT SYSTEM
C 8 SAMPLE CODE: FLOATING PT     FIXED PT     FIXED PT-GAIN     CORRELATED
C 9 GAIN  TYPE: FIXED     BINARY     FLOATING POINT     OTHER
C10 FILTERS: ALIAS     HZ  NOTCH     HZ  BAND     -     HZ  SLOPE    -    DB/OCT
C11 SOURCE: TYPE            NUMBER/POINT        POINT INTERVAL
C12     PATTERN:                           LENGTH        WIDTH
C13 SWEEP: START     HZ  END     HZ  LENGTH      MS  CHANNEL NO     TYPE
C14 TAPER: START LENGTH       MS  END LENGTH       MS  TYPE
C15 SPREAD: OFFSET        MAX DISTANCE        GROUP INTERVAL
C16 GEOPHONES: PER GROUP     SPACING     FREQUENCY     MFG          MODEL
C17     PATTERN:                           LENGTH        WIDTH
C18 TRACES SORTED BY: RECORD     CDP     OTHER
C19 AMPLITUDE RECOVERY: NONE      SPHERICAL DIV      AGC     OTHER
C20 MAP PROJECTION                      ZONE ID       COORDINATE UNITS
C21 PROCESSING:
C22 PROCESSING:
C23
C24
C25
C26
C27
C28
C29
C30
C31
C32
C33
C34
C35
C36
C37
C38
C39
C40 END EBCDIC
```

#### File Binary Header

```console
$ segy dump binary-header \
    https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY
{
  "job_id": 18909,
  "line_num": 1,
  "reel_num": 1,
  "data_traces_per_ensemble": 957,
  "aux_traces_per_ensemble": 1,
  "sample_interval": 2000,
  "orig_sample_interval": 2000,
  "samples_per_trace": 2000,
  "orig_samples_per_trace": 2000,
  "data_sample_format": 1,
  "ensemble_fold": 0,
  "trace_sorting_code": 1,
  "vertical_sum_code": 1,
  "sweep_freq_start": 0,
  "sweep_freq_end": 0,
  "sweep_length": 0,
  "sweep_type_code": 4,
  "sweep_trace_num": 0,
  "sweep_taper_start": 0,
  "sweep_taper_end": 0,
  "taper_type_code": 3,
  "correlated_data_code": 1,
  "binary_gain_code": 2,
  "amp_recovery_code": 1,
  "measurement_system_code": 2,
  "impulse_polarity_code": 0,
  "vibratory_polarity_code": 0,
  "segy_revision_major": 0,
  "segy_revision_minor": 0
}
```

#### Trace Header

This is how we can get three header fields for a few traces.

```console
$ segy dump trace-header https://gdr-data-lake.s3.us-west-2.amazonaws.com/soda_lake/raw_seismic/2010/v1.0.0/F7733R1.SGY \
    --index 100 --index 101 --index 500 --index 501 \
    --field source_coord_x --field source_coord_y \
    --field group_coord_x --field group_coord_y \
    --field coordinate_scalar
             source_coord_x  source_coord_y  group_coord_x  group_coord_y  coordinate_scalar
trace_index
100                 7735193        45340080        7747690       45337925                 -1
101                 7735193        45340080        7748244       45337549                 -1
500                 7735193        45340080        7745319       45347606                 -1
501                 7735193        45340080        7745874       45347232                 -1
```

(env-configuration-options)=

## Configuration Options

The examples above use public HTTPS. Do not set `SEGY_STORAGE_OPTIONS`
for that URL.

When accessing public datasets from S3, we need to set
`SegySettings().storage_options = {"anon": True}`{l=python} for anonymous
access. [SegySettings](#SegySettings) exposes all configuration options
as environment variables. We just need to set `storage_options` with the `JSON`
string `{"anon": true}`{l=python}. On Linux you can do this by the command below.
Environment variables can be configured in many ways, please refer to the options
for your specific Operating System (OS).

```shell
export SEGY_STORAGE_OPTIONS='{"anon": true}'
```

```{seealso}
[Settings Management](#settings)
```
