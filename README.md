[[_TOC_]]

# SEG-Y
This is an efficient and comprehensive SEG-Y parsing library.

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

See the
[demo](https://dev.azure.com/TGSCloud/Datascience/_git/segy?path=/examples/demo.ipynb&version=GBmain&_a=preview).

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
- [X] Rev 0 (1975)
- [X] Rev 1 (2002)
- [ ] Rev 2 (2017)
- [ ] Rev 2.1 (2023)

### Custom SEG-Y Standards
You can build your own SEG-Y "standard" with composition of specs for:
- Text header (file + extended)
- Binary header
- Traces (header + extended header + samples)



Feel free to share the thoughts and raise the issues to improve the library.