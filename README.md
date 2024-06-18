[![PyPI](https://img.shields.io/pypi/v/segy.svg)][install_pip]
[![Conda](https://img.shields.io/conda/vn/conda-forge/segy)][install_conda]
[![Python Version](https://img.shields.io/pypi/pyversions/multidimio)][python version]
[![Status](https://img.shields.io/pypi/status/segy.svg)][status]
[![License](https://img.shields.io/pypi/l/segy)][apache 2.0 license]

[![Tests](https://github.com/TGSAI/segy/actions/workflows/tests.yaml/badge.svg?branch=main)][tests]
[![Codecov](https://codecov.io/gh/TGSAI/segy/branch/main/graph/badge.svg)][codecov]
[![Read the documentation at https://segy.readthedocs.io/](https://img.shields.io/readthedocs/segy/stable.svg?label=Read%20the%20Docs)][read the docs]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)][ruff]

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/segy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=PyPI%20downloads)][pypi_]
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/segy?label=Conda%20downloads&style=flat)][conda-forge_]

[pypi_]: https://pypi.org/project/segy/
[conda-forge_]: https://anaconda.org/conda-forge/segy
[status]: https://pypi.org/project/segy/
[python version]: https://pypi.org/project/segy
[read the docs]: https://segy.readthedocs.io/
[tests]: https://github.com/TGSAI/segy/actions/workflows/tests.yaml
[codecov]: https://app.codecov.io/gh/TGSAI/segy
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff]: https://github.com/astral-sh/ruff
[install_pip]: https://segy.readthedocs.io/en/stable/installation.html#using-pip-and-virtualenv
[install_conda]: https://segy.readthedocs.io/en/stable/installation.html#using-conda

# SEG-Y

> 🚧👷🏻 This project is under active development, expect breaking changes
> the to API 👷🏻🚧
> _\- March, 2024_

This is an efficient and comprehensive SEG-Y parsing library.

See the [documentation][read the docs] for more information.

This is not an official TGS product.

## Features

The library utilizes `numpy` and `fsspec`, includes the reading from various local
and remote resources at a high speed. It also allows the users to build their own
SEG-Y specifications.

## Installing `segy`

Clone the repo and install it using pip:

Simplest way to install `segy` is via [pip] from [PyPI]:

```shell
$ pip install segy
```

or install `segy` via [conda] from [conda-forge]:

```shell
$ conda install -c conda-forge segy
```

> Extras must be installed separately on `Conda` environments.

For details, please see the [installation instructions]
in the documentation.

## Using `segy`

Please see the [Command-line Usage] for details.

For Python API please see the [API Reference] for details.

### Reading Capabilities

It supports reading from local and cloud files (object store). It can read:

- Sequential traces (fastest)
- Disjoint sequential regions (fast)
- Random traces (slow)

The library will also try to infer the endianness and the revision of the SEG-Y
file automatically. If it can't, users can override the endianness, revision, and
more parameters using the settings.

### High Performance

The performance is high and to be proven with upcoming benchmarks. The initial
subjective benchmarks is very acceptable.

### Flexibility

The library provides a fully flexible, schematized SEG-Y structure, including
data models and JSON schema parsing and validation.

### Predefined SEG-Y Standards

It supports predefined SEG-Y "standards" for various versions. However, some versions
are still in progress and not all validation logic is implemented yet:

- ✅ Rev 0 (1975)
- ✅ Rev 1 (2002)
- ✅ Rev 2 (2017)
- 🔲 Rev 2.1 (2023)

### Custom SEG-Y Standards

You can build your own SEG-Y "standard" with composition of specs for:

- Text header (file + extended)
- Binary header
- Traces (header + extended header + samples)

## Contributing to `segy`

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## Licensing

Distributed under the terms of the [Apache 2.0 license].
`segy` is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was established at [TGS](https://www.tgs.com/). Current
maintainer is [Altay Sansal](https://github.com/tasansal) with the support
of many more great colleagues.

The CI/CD tooling is loosely based on [Hypermodern Python Cookiecutter]
with more modern tooling applied elsewhere.

[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[pypi]: https://pypi.org/
[conda-forge]: https://conda-forge.org/
[file an issue]: https://github.com/TGSAI/segy/issues
[pip]: https://pip.pypa.io/
[conda]: https://docs.conda.io/

<!-- github-only -->

[apache 2.0 license]: https://github.com/TGSAI/segy/blob/main/LICENSE
[contributor guide]: https://github.com/TGSAI/segy/blob/main/CONTRIBUTING.md
[command-line usage]: https://segy.readthedocs.io/en/stable/cli_usage.html
[api reference]: https://segy.readthedocs.io/en/stable/api_reference.html
[installation instructions]: https://segy.readthedocs.io/en/stable/installation.html
