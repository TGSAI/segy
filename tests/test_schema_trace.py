"""Tests for trace descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import StructuredDataTypeDescriptor
from segy.schema import StructuredFieldDescriptor
from segy.schema import TraceDescriptor
from segy.schema import TraceSampleDescriptor


@pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
@pytest.mark.parametrize(
    ("sample_format", "samples_per_trace"),
    [
        (ScalarType.IBM32, 11),
        (ScalarType.FLOAT32, 21),
        (ScalarType.UINT16, 5),
    ],
)
class TestTraceDescriptors:
    """Tests for trace sample descriptor and its dtype."""

    def test_trace_descriptor(
        self,
        endianness: Endianness,
        sample_format: ScalarType,
        samples_per_trace: int,
    ) -> None:
        """Testing dtype attribute and contents of TraceDescriptor."""
        header_descr = StructuredDataTypeDescriptor(
            fields=[
                StructuredFieldDescriptor(name="h1", format=ScalarType.INT16, offset=0),
                StructuredFieldDescriptor(name="h2", format=ScalarType.INT8, offset=16),
            ]
        )
        sample_descr = TraceSampleDescriptor(
            format=sample_format, samples=samples_per_trace
        )
        trace_descr = TraceDescriptor(
            header_descriptor=header_descr,
            sample_descriptor=sample_descr,
            endianness=endianness,
        )

        expected_itemsize = header_descr.dtype.itemsize + sample_descr.dtype.itemsize
        expected_header_itemsize = header_descr.dtype.itemsize
        expected_sample_subtype = (np.dtype(sample_format.char), (samples_per_trace,))
        assert trace_descr.dtype.itemsize == expected_itemsize
        assert trace_descr.header_descriptor.dtype.names == ("h1", "h2")
        assert trace_descr.header_descriptor.dtype.itemsize == expected_header_itemsize
        assert trace_descr.sample_descriptor.dtype.subdtype == expected_sample_subtype
        if endianness == Endianness.LITTLE:
            assert trace_descr.dtype.isnative
        else:
            assert not trace_descr.dtype.isnative
