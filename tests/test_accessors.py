"""Tests for accessors for SegyArrays."""

from __future__ import annotations

import pytest

from segy.accessors import TraceAccessor
from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec
from segy.transforms import Transform
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline


def compare_transform(transform_a: Transform, transform_b: Transform) -> bool:
    """Helper function for equality comparison of transforms."""
    return all(
        a == b
        for a, b in [
            (type(transform_a), type(transform_b)),
            (transform_a.keys, transform_b.keys),
        ]
    )


def compare_transform_sequence(
    transform_a: list[Transform], transform_b: list[Transform]
) -> bool:
    """Helper function for equality comparison of two sequences of transforms."""
    if len(transform_a) != len(transform_b):
        return False
    return all(
        compare_transform(a, b) for a, b in zip(transform_a, transform_b, strict=True)
    )


@pytest.fixture(
    params=[(ScalarType.IBM32, ScalarType.IBM32), (ScalarType.INT32, ScalarType.IBM32)]
)
def mock_trace_spec(
    request: pytest.FixtureRequest,
) -> tuple[TraceSpec, dict[str, list[Transform]]]:
    """Create mock trace spec for testing and expected values for accessors."""
    header_field_type = request.param[0]
    sample_field_type = request.param[1]
    trace_header_spec = HeaderSpec(
        fields=[HeaderField(name="h1", format=header_field_type, byte=1)],
        item_size=4,
        endianness=Endianness.BIG,
        offset=0,
    )
    trace_data_spec = TraceDataSpec(format=sample_field_type, samples=3)
    trace_spec = TraceSpec(header=trace_header_spec, data=trace_data_spec)
    expected = {}
    base_transform = TransformFactory.create("byte_swap", Endianness.LITTLE)
    expected["header"] = (
        [base_transform]
        if header_field_type != ScalarType.IBM32
        else [base_transform, TransformFactory.create("ibm_float", "to_ieee", ["h1"])]
    )
    expected["data"] = (
        [base_transform]
        if sample_field_type != ScalarType.IBM32
        else [base_transform, TransformFactory.create("ibm_float", "to_ieee")]
    )

    hdr_pipeline = TransformPipeline()
    map(hdr_pipeline.add_transform, expected["header"])
    data_pipeline = TransformPipeline()
    map(data_pipeline.add_transform, expected["data"])
    expected["trace"] = [TransformFactory.create("trace", hdr_pipeline, data_pipeline)]

    return trace_spec, expected


def test_trace_accessor(
    mock_trace_spec: tuple[TraceSpec, dict[str, list[Transform]]],
) -> None:
    """Test for trace accessor decoder transforms."""
    trace_spec = mock_trace_spec[0]
    expected_transforms = mock_trace_spec[1]
    trace_accessor = TraceAccessor(trace_spec)
    assert compare_transform_sequence(
        trace_accessor.header_decode_pipeline.transforms, expected_transforms["header"]
    )
    assert compare_transform_sequence(
        trace_accessor.sample_decode_pipeline.transforms, expected_transforms["data"]
    )
    assert compare_transform_sequence(
        trace_accessor.trace_decode_pipeline.transforms, expected_transforms["trace"]
    )
