"""Tests for functions in indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import fsspec
import numpy as np
import pytest

from segy.ibm import ieee2ibm
from segy.indexing import bounds_check
from segy.indexing import merge_cat_file
from segy.indexing import trace_ibm2ieee_inplace
from segy.schema import Endianness
from segy.schema import TraceDescriptor

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

err_msg_format = "indices {0} are out of bounds."


@pytest.mark.parametrize(
    ("indices", "max_", "expected_err"),
    [
        ([1, 2, 3], 4, ""),
        (list(range(15)), 15, ""),
        ([-1, 2, 4], 8, err_msg_format.format([-1])),
        ([-19, -1000, 100, 25000], 25001, err_msg_format.format([-19, -1000])),
    ],
)
def test_bounds_check(indices: list[int], max_: int, expected_err: str) -> None:
    """Tests the bounds checking for indexing headers+traces."""
    if expected_err != "":
        with pytest.raises(IndexError):
            bounds_check(indices, max_, "")
    else:
        bounds_check(indices, max_, "")


@pytest.mark.parametrize(
    ("values", "formats", "offsets"),
    [
        ((-120, 255, 1234.54321), ["<i4", "B", "<f8"], [4, 50, 72]),
        ((523455, 0xFFFFFFFF), [">u4", ">u4"], [12, 40]),
        ((-123, 0xFA10), ["<i4", ">u4"], [10, 14]),
    ],
)
def test_merge_cat_file(
    values: list[int | float], formats: str, offsets: list[int], tmp_path: Path
) -> None:
    """Test merging byte ranges from file(s)."""
    # Generate a bytestream with gaps and write to file.
    names = [f"val{i}" for i in range(len(values))]
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets})
    struct = np.zeros(1, dtype=dtype)
    struct[0] = values

    tmp_file = tmp_path / "temp_cat_file.bin"
    struct.tofile(tmp_file)

    # Read values from byte start/stop pairs
    fs = fsspec.filesystem("local")
    starts = offsets
    stops = []
    for start, format_ in zip(starts, formats):
        stops.append(start + np.dtype(format_).itemsize)

    cat_res = merge_cat_file(fs, tmp_file, starts, stops)

    packed_dtype = np.dtype({"names": names, "formats": formats})
    cast_res = np.frombuffer(cat_res, dtype=packed_dtype)

    np.testing.assert_array_equal(cast_res, struct)


@pytest.mark.parametrize(
    ("header_params", "data_params", "float_vals"),
    [
        (
            {
                "dt_string": "i2,i4",
                "names": ["a", "b"],
                "endianness": Endianness.LITTLE,
            },
            {"format": "uint32", "endianness": Endianness.LITTLE, "samples": 3},
            [0.0, 0.1, 3.141593],
        ),
        (
            {"dt_string": "i2,i4", "names": ["a", "b"], "endianness": Endianness.BIG},
            {"format": "uint32", "endianness": Endianness.BIG, "samples": 3},
            [1.01, -2.01, 33.11],
        ),
    ],
)
def test_trace_ibm2ieee_inplace(
    header_params: dict[str, Any],
    data_params: dict[str, Any],
    float_vals: list[float],
    make_trace_descriptor: Callable[..., TraceDescriptor],
) -> None:
    """Test changing dtype of IBM32 values inplace."""
    trace_descr = make_trace_descriptor(header_params, data_params)
    samp_trace = np.zeros(1, dtype=trace_descr.dtype)
    ieee_floats = np.array(float_vals, dtype="<f4")
    ibm_floats = ieee2ibm(ieee_floats)
    if trace_descr.data_descriptor.endianness == Endianness.BIG:
        # emulating how trace indexer swaps byte order
        # of big endian data types
        samp_trace = samp_trace.byteswap(inplace=True).newbyteorder()

    samp_trace["data"].squeeze()[:] = ibm_floats[:]
    inplace_swap_vals = trace_ibm2ieee_inplace(samp_trace)
    actual_floats = inplace_swap_vals["data"].squeeze()
    np.testing.assert_array_almost_equal(ieee_floats, actual_floats)
