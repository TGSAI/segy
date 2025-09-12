"""Tests for segy spec components."""

import numpy as np
import pytest

from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.schema import SegySpec
from segy.schema import TextHeaderEncoding
from segy.schema import TextHeaderSpec
from segy.schema import TraceDataSpec
from segy.schema.text_header import ExtendedTextHeaderSpec
from segy.standards import SegyStandard
from segy.standards import get_segy_standard


@pytest.fixture(params=[SegyStandard.REV0, SegyStandard.REV1])
def segy_spec(request: pytest.FixtureRequest) -> SegySpec:
    """Fixture for creating known SegySpec instances for customizing."""
    standard = request.param
    return get_segy_standard(standard)


class TestSegySpecCustomize:
    """Tests for SegySpec customization."""

    def test_custom_textual_file_header(self, segy_spec: SegySpec) -> None:
        """Test customizing text header spec."""
        custom_text_spec = TextHeaderSpec(
            rows=1,
            cols=5,
            encoding=TextHeaderEncoding.EBCDIC,
        )

        custom_spec = segy_spec.customize(text_header_spec=custom_text_spec)

        assert custom_spec.segy_standard is None
        assert custom_spec.text_header == custom_text_spec

    def test_custom_binary_file_headers(self, segy_spec: SegySpec) -> None:
        """Test customizing binary file header spec."""
        custom_fields = [
            # Replace Sample Interval in microseconds with custom field f1
            HeaderField(name="f1", format=ScalarType.UINT8, byte=17),
            # Replace Sweep Frequency at Start with custom field f2
            HeaderField(name="f2", format=ScalarType.INT16, byte=33),
            # Replace SEG-Y Revision Number + Fixed length trace flag with custom field f3
            HeaderField(name="f3", format=ScalarType.UINT32, byte=301),
            # Above header field will add for Rev0 and replace 2 fields in Rev1
        ]

        custom_spec = segy_spec.customize(binary_header_fields=custom_fields)

        expected_itemsize = segy_spec.binary_header.dtype.itemsize
        assert custom_spec.segy_standard is None
        assert custom_spec.binary_header.dtype.itemsize == expected_itemsize

        all_start_bytes = [field.byte for field in custom_spec.binary_header.fields]

        for field in custom_fields:
            assert field in custom_spec.binary_header.fields
            # Check that the custom field's byte position appears exactly once in the list
            assert all_start_bytes.count(field.byte) == 1

    def test_custom_trace_headers(self, segy_spec: SegySpec) -> None:
        """Test customizing trace header spec."""
        custom_fields = [
            HeaderField(name="f1", format=ScalarType.UINT8, byte=9),
            HeaderField(name="f2", format=ScalarType.UINT32, byte=151),
        ]

        custom_spec = segy_spec.customize(trace_header_fields=custom_fields)

        expected_itemsize = segy_spec.trace.header.dtype.itemsize
        assert custom_spec.segy_standard is None
        assert len(custom_spec.trace.header.fields) == len(
            segy_spec.trace.header.fields
        )
        assert custom_spec.trace.header.dtype.itemsize == expected_itemsize

        all_start_bytes = [field.byte for field in custom_spec.trace.header.fields]

        for field in custom_fields:
            assert field in custom_spec.trace.header.fields
            assert all_start_bytes.count(field.byte) == 1

    def test_custom_extended_text_header(self, segy_spec: SegySpec) -> None:
        """Test customizing extended text header spec."""
        custom_text_spec = TextHeaderSpec(
            rows=3,
            cols=7,
            encoding=TextHeaderEncoding.ASCII,
        )

        custom_ext_text_spec = ExtendedTextHeaderSpec(spec=custom_text_spec, count=1)
        custom_spec = segy_spec.customize(ext_text_spec=custom_ext_text_spec)

        assert custom_spec.segy_standard is None
        assert custom_spec.ext_text_header == custom_ext_text_spec

    def test_custom_trace_samples(self, segy_spec: SegySpec) -> None:
        """Test customizing trace data spec."""
        custom_samples = TraceDataSpec(format=ScalarType.UINT16, samples=3)

        custom_spec = segy_spec.customize(trace_data_spec=custom_samples)

        expected_subdtype = (np.dtype("uint16"), (3,))
        assert custom_spec.segy_standard is None
        assert custom_spec.trace.dtype.itemsize == 246  # noqa: PLR2004
        assert custom_spec.trace.data.dtype.itemsize == 6  # noqa: PLR2004
        assert custom_spec.trace.data.dtype.subdtype == expected_subdtype
