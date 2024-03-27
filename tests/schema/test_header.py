"""Tests for the Text Headers, Binary Headers, and Trace Headers for different SEGY revisions."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

from segy.schema.header import TextHeaderDescriptor

if TYPE_CHECKING:
    from numpy.typing import DTypeLike
    from numpy.typing import NDArray

    from segy.schema.data_type import StructuredDataTypeDescriptor


@pytest.mark.parametrize(
    "text_header_params",
    [
        (
            {
                "rows": 40,
                "cols": 80,
                "encoding": "ebcdic",
                "format": "uint8",
                "offset": 0,
            }
        ),
        ({"rows": 40, "cols": 80, "encoding": "ascii", "format": "uint8", "offset": 0}),
    ],
)
def test_full_text_headers(
    text_header_samples: str, text_header_params: dict[str, Any]
) -> None:
    """Test for reading text headers encoded as ASCII or EBCDIC and wrapping into formed lines."""
    new_text_head_desc = TextHeaderDescriptor(**text_header_params)
    raw_string = new_text_head_desc._encode(text_header_samples)
    decoded_str = new_text_head_desc._decode(raw_string)
    split_lines = new_text_head_desc._wrap(decoded_str).split("\n")
    assert decoded_str == text_header_samples
    assert (new_text_head_desc.rows, new_text_head_desc.cols) == (
        len(split_lines),
        len(split_lines[0]),
    )
    # assertions for exception cases
    with pytest.raises(ValueError, match="Text length must be equal to rows x cols."):
        new_text_head_desc._encode(text_header_samples[:-10])
    with pytest.raises(
        ValueError, match="rows x cols must be equal wrapped text length."
    ):
        new_text_head_desc._wrap(text_header_samples[:-10])
    assert "\n" not in new_text_head_desc._unwrap(text_header_samples)


def test_binary_header_descriptors(
    binary_header_descriptors: StructuredDataTypeDescriptor,
) -> None:
    """Test for reading binary headers.

    Tested on a buffer of random values and compares descriptor dtype results
    to a standard numpy struct to parse the same values.
    """
    dt_info = get_dt_info(binary_header_descriptors.dtype)
    vbuffer = void_buffer(binary_header_descriptors.item_size or 0)
    assert (
        binary_header_descriptors.item_size == binary_header_descriptors.dtype.itemsize
    )
    assert (
        vbuffer.view(binary_header_descriptors.dtype)[0].tolist()
        == vbuffer.view(np.dtype(dt_info["combo_str"]))[0].tolist()
    )


def test_trace_header_descriptors(
    trace_header_descriptors: StructuredDataTypeDescriptor,
) -> None:
    """Test for reading trace headers.

    Tested on a buffer of random values and compares descriptor
    dtype results to a standard numpy struct to parse the same values.
    """
    dt_info = get_dt_info(trace_header_descriptors.dtype)
    vbuffer = void_buffer(trace_header_descriptors.item_size or 0)
    assert trace_header_descriptors.item_size == trace_header_descriptors.dtype.itemsize
    assert (
        vbuffer.view(trace_header_descriptors.dtype)[0].tolist()
        == vbuffer.view(np.dtype(dt_info["combo_str"]))[0].tolist()
    )


def void_buffer(buff_size: int) -> NDArray[np.void]:
    """Creates a new buffer of requested number of bytes with void(number_bytes) datatype.

    Prefills with random bytes.
    """
    rng = np.random.default_rng()
    return np.frombuffer(rng.bytes(buff_size), dtype=np.void(buff_size))


def get_dt_info(dtype: DTypeLike) -> dict[str, Any]:
    """Helper function to get info about a numpy dtype."""
    atrnames = [
        "descr",
        "str",
        "fields",
        "itemsize",
        "byteorder",
        "shape",
        "names",
    ]
    dt_info = dict(zip(atrnames, operator.attrgetter(*atrnames)(dtype)))
    dt_info["offsets"] = [f[-1] for f in dt_info["fields"].values()]
    dt_info["combo_str"] = ",".join([f[1] for f in dt_info["descr"]])
    return dt_info
