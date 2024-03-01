[![PyPI](https://img.shields.io/pypi/v/segy.svg)][install_pip]
[![Conda](https://img.shields.io/conda/vn/conda-forge/segy)][install_conda]
[![Python Version](https://img.shields.io/pypi/pyversions/multidimio)][python version]
[![Status](https://img.shields.io/pypi/status/segy.svg)][status]
[![License](https://img.shields.io/pypi/l/segy)][apache 2.0 license]

[![Tests](https://github.com/TGSAI/segy/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/TGSAI/segy/branch/main/graph/badge.svg)][codecov]
[![Read the documentation at https://segy.readthedocs.io/](https://img.shields.io/readthedocs/segy/latest.svg?label=Read%20the%20Docs)][read the docs]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)][ruff]

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/segy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20downloads)][pypi_]
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/segy?label=Conda%20downloads&style=flat)][conda-forge_]

[pypi_]: https://pypi.org/project/segy/
[conda-forge_]: https://anaconda.org/conda-forge/segy
[status]: https://pypi.org/project/segy/
[python version]: https://pypi.org/project/segy
[read the docs]: https://segy.readthedocs.io/
[tests]: https://github.com/TGSAI/segy/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/TGSAI/segy
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff]: https://github.com/astral-sh/ruff
[install_pip]: https://segy.readthedocs.io/en/latest/installation.html#using-pip-and-virtualenv
[install_conda]: https://segy.readthedocs.io/en/latest/installation.html#using-conda

# SEG-Y

This is an efficient and comprehensive SEG-Y parsing library.

See the [documentation][read the docs] for more information.

This is not an official TGS product.

## Installation

Clone the repo and install it using pip:

```shell
pip install .
```

## Basic Usage

It's simple to operate the library:

```python
from segy import SegyFile

sgy = SegyFile("gs://bucket/prefix")

full_trace = sgy.trace[1000]
just_data_header = sgy.header[1000]
just_trace_data = sgy.data[1000]
```

## Features

The library utilizes `numpy` and `fsspec`, includes the reading from various local
and remote resources at a high speed. It also allows the users to build their own
SEG-Y specifications.

### Compatibility

The library provides full `numpy` compatibility with `ndarray`s of scalar or
structured types.

### Reading Capabilities

It supports reading from local and cloud files (object store). It can read:

- Sequential traces (fastest)
- Disjoint sequential regions (fast)
- Random traces (slow)

### High Performance

The performance is high and to be proven with upcoming benchmarks. The initial
subjective benchmarks is very acceptable.

### Flexibility

The library provides a fully flexible, schematized SEG-Y structure, including
data models and JSON schema parsing and validation.

### Predefined SEG-Y Standards

It supports predefined SEG-Y "standards" for various versions. However,
some versions are still in progress:

- [x] Rev 0 (1975)
- [x] Rev 1 (2002)
- [ ] Rev 2 (2017)
- [ ] Rev 2.1 (2023)

### Custom SEG-Y Standards

You can build your own SEG-Y "standard" with composition of specs for:

- Text header (file + extended)
- Binary header
- Traces (header + extended header + samples)

Feel free to share the thoughts and raise the issues to improve the library.

<!-- github-only -->

[apache 2.0 license]: https://github.com/TGSAI/segy/blob/main/LICENSE
