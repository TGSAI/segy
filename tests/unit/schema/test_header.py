"""Tests for the Text Headers, Binary Headers, and Trace Headers for different SEGY revisions."""

import numpy as np
import pytest
from tests.helpers.helper_tools import TestHelpers

from segy.schema.header import TextHeaderDescriptor


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
def test_full_text_headers(text_header_samples, text_header_params) -> None:
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


def test_binary_header_descriptors(binary_header_descriptors) -> None:
    """Test for reading binary headers, tested on a buffer
    of random values and compares descriptor dtype results
    to a standard numpy struct to parse the same values.
    """
    dt_info = TestHelpers.get_dt_info(binary_header_descriptors.dtype)
    vbuffer = TestHelpers.void_buffer(binary_header_descriptors.item_size)
    assert (
        binary_header_descriptors.item_size == binary_header_descriptors.dtype.itemsize
    )
    assert (
        vbuffer.view(binary_header_descriptors.dtype)[0].tolist()
        == vbuffer.view(np.dtype(dt_info["combo_str"]))[0].tolist()
    )


def test_trace_header_descriptors(trace_header_descriptors) -> None:
    """Test for reading trace headers, tested on a buffer
    of random values and compares descriptor dtype results
    to a standard numpy struct to parse the same values.
    """
    dt_info = TestHelpers.get_dt_info(trace_header_descriptors.dtype)
    vbuffer = TestHelpers.void_buffer(trace_header_descriptors.item_size)
    assert trace_header_descriptors.item_size == trace_header_descriptors.dtype.itemsize
    assert (
        vbuffer.view(trace_header_descriptors.dtype)[0].tolist()
        == vbuffer.view(np.dtype(dt_info["combo_str"]))[0].tolist()
    )
