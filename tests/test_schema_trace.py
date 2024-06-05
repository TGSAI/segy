"""Tests for trace spec."""

from __future__ import annotations

import numpy as np
import pytest

from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import HeaderSpec
from segy.schema import ScalarType
from segy.schema import TraceDataSpec
from segy.schema import TraceSpec


@pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
@pytest.mark.parametrize(
    ("sample_format", "samples_per_trace"),
    [
        (ScalarType.IBM32, 11),
        (ScalarType.FLOAT32, 21),
        (ScalarType.UINT16, 5),
    ],
)
class TestTraceSpec:
    """Tests for trace spec and its dtype."""

    def test_trace_spec(
        self,
        endianness: Endianness,
        sample_format: ScalarType,
        samples_per_trace: int,
    ) -> None:
        """Testing dtype attribute and contents of TraceSpec."""
        header_spec = HeaderSpec(
            fields=[
                HeaderField(name="h1", format=ScalarType.INT16, byte=1),
                HeaderField(name="h2", format=ScalarType.INT8, byte=17),
            ]
        )
        data_spec = TraceDataSpec(format=sample_format, samples=samples_per_trace)
        trace_spec = TraceSpec(
            header_spec=header_spec,
            data_spec=data_spec,
            endianness=endianness,
        )

        expected_itemsize = header_spec.dtype.itemsize + data_spec.dtype.itemsize
        expected_header_itemsize = header_spec.dtype.itemsize
        expected_sample_subtype = (np.dtype(sample_format.char), (samples_per_trace,))
        assert trace_spec.dtype.itemsize == expected_itemsize
        assert trace_spec.header_spec.dtype.names == ("h1", "h2")
        assert trace_spec.header_spec.dtype.itemsize == expected_header_itemsize
        assert trace_spec.data_spec.dtype.subdtype == expected_sample_subtype
        if endianness == Endianness.LITTLE:
            assert trace_spec.dtype.isnative
        else:
            assert not trace_spec.dtype.isnative
