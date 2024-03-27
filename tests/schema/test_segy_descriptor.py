"""Tests for SegyDescriptor components."""

from typing import Any

import pytest

from segy.standards import SegyStandard
from segy.standards.registry import get_spec


@pytest.mark.parametrize(
    ("rev_number", "custom_segy_file_descriptors"),
    [(0.0, 1), (1.0, 1)],
    indirect=["custom_segy_file_descriptors"],
)
def test_custom_segy_descriptor(
    rev_number: float, custom_segy_file_descriptors: dict[str, Any]
) -> None:
    """Test for creating customized SegyDescriptor."""
    rev_spec = get_spec(SegyStandard(rev_number))
    custom_spec = rev_spec.customize(
        text_header_spec=custom_segy_file_descriptors["text_header_descriptor"],
        binary_header_fields=custom_segy_file_descriptors[
            "binary_header_descriptor"
        ].fields,
        extended_text_spec=None,
        trace_header_fields=custom_segy_file_descriptors[
            "trace_header_descriptor"
        ].fields,
        trace_data_spec=custom_segy_file_descriptors["trace_sample_descriptor"],
    )
    assert (
        custom_spec.text_file_header
        == custom_segy_file_descriptors["text_header_descriptor"]
    )
    assert (
        custom_spec.binary_file_header.fields
        == custom_segy_file_descriptors["binary_header_descriptor"].fields
    )
    assert (
        custom_spec.trace.header_descriptor.fields
        == custom_segy_file_descriptors["trace_header_descriptor"].fields
    )
    assert (
        custom_spec.trace.sample_descriptor
        == custom_segy_file_descriptors["trace_sample_descriptor"]
    )
