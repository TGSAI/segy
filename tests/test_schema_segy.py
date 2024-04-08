"""Tests for SegyDescriptor components."""

import numpy as np
import pytest

from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import StructuredFieldDescriptor
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceSampleDescriptor
from segy.standards import SegyStandard
from segy.standards import get_segy_standard


@pytest.fixture(params=[SegyStandard.REV0, SegyStandard.REV1])
def segy_descriptor(request: pytest.FixtureRequest) -> SegyDescriptor:
    """Fixture for creating known SegyDescriptor instances for customizing."""
    standard = request.param
    return get_segy_standard(standard)


class TestSegyDescriptorCustomize:
    """Tests for SegyDescriptor customization."""

    def test_custom_textual_file_header(self, segy_descriptor: SegyDescriptor) -> None:
        """Test customizing text descriptor."""
        custom_text_descr = TextHeaderDescriptor(
            rows=1,
            cols=5,
            encoding=TextHeaderEncoding.EBCDIC,
            format=ScalarType.UINT8,
        )

        custom_spec = segy_descriptor.customize(text_header_spec=custom_text_descr)

        assert custom_spec.segy_standard is None
        assert custom_spec.text_file_header == custom_text_descr

    def test_custom_binary_file_headers(self, segy_descriptor: SegyDescriptor) -> None:
        """Test customizing binary file header descriptor."""
        custom_fields = [
            StructuredFieldDescriptor(name="f1", format=ScalarType.UINT8, offset=16),
            StructuredFieldDescriptor(name="f2", format=ScalarType.INT16, offset=32),
            StructuredFieldDescriptor(name="f3", format=ScalarType.UINT32, offset=300),
        ]

        custom_spec = segy_descriptor.customize(binary_header_fields=custom_fields)

        expected_itemsize = segy_descriptor.binary_file_header.dtype.itemsize
        assert custom_spec.segy_standard is None
        assert len(custom_spec.binary_file_header.fields) == len(custom_fields)
        assert custom_spec.binary_file_header.dtype.names == ("f1", "f2", "f3")
        assert custom_spec.binary_file_header.dtype.itemsize == expected_itemsize

    def test_custom_trace_headers(self, segy_descriptor: SegyDescriptor) -> None:
        """Test customizing trace header descriptor."""
        custom_fields = [
            StructuredFieldDescriptor(name="f1", format=ScalarType.UINT8, offset=8),
            StructuredFieldDescriptor(name="f2", format=ScalarType.UINT32, offset=150),
        ]

        custom_spec = segy_descriptor.customize(trace_header_fields=custom_fields)

        expected_itemsize = segy_descriptor.trace.header_descriptor.dtype.itemsize
        assert custom_spec.segy_standard is None
        assert len(custom_spec.trace.header_descriptor.fields) == len(custom_fields)
        assert custom_spec.trace.header_descriptor.dtype.names == ("f1", "f2")
        assert custom_spec.trace.header_descriptor.dtype.itemsize == expected_itemsize

    def test_custom_extended_text_header(self, segy_descriptor: SegyDescriptor) -> None:
        """Test customizing extended text header descriptor."""
        custom_text_descr = TextHeaderDescriptor(
            rows=3,
            cols=7,
            encoding=TextHeaderEncoding.ASCII,
            format=ScalarType.UINT8,
        )

        custom_spec = segy_descriptor.customize(extended_text_spec=custom_text_descr)

        assert custom_spec.segy_standard is None
        assert custom_spec.extended_text_header == custom_text_descr

    def test_custom_trace_samples(self, segy_descriptor: SegyDescriptor) -> None:
        """Test customizing trace sample descriptor."""
        custom_samples = TraceSampleDescriptor(format=ScalarType.UINT16, samples=3)

        custom_spec = segy_descriptor.customize(trace_data_spec=custom_samples)

        expected_subdtype = (np.dtype("uint16"), (3,))
        assert custom_spec.segy_standard is None
        assert custom_spec.trace.dtype.itemsize == 246  # noqa: PLR2004
        assert custom_spec.trace.sample_descriptor.dtype.itemsize == 6  # noqa: PLR2004
        assert custom_spec.trace.sample_descriptor.dtype.subdtype == expected_subdtype
