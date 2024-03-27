"""Tests for the SegyFactory used in creation."""

from __future__ import annotations

import numpy as np
import pytest
from fsspec import filesystem

from segy import SegyFile
from segy.factory import SegyFactory
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyDescriptor
from segy.schema import SegyStandard
from segy.schema import StructuredDataTypeDescriptor
from segy.schema import StructuredFieldDescriptor
from segy.schema import TextHeaderDescriptor
from segy.schema import TextHeaderEncoding
from segy.schema import TraceDataDescriptor
from segy.schema import TraceDescriptor


@pytest.fixture()
def mock_segy_spec() -> SegyDescriptor:
    """Create a fixture for the mock SegyDescriptor class."""
    return SegyDescriptor(
        endianness=Endianness.BIG,
        segy_standard=SegyStandard.REV0,
        text_file_header=TextHeaderDescriptor(
            rows=2,
            cols=20,
            offset=0,
            encoding=TextHeaderEncoding.EBCDIC,
            format=ScalarType.UINT8,  # noqa: A003
        ),
        binary_file_header=StructuredDataTypeDescriptor(
            fields=[
                StructuredFieldDescriptor(
                    name="seg_y_revision",
                    offset=0,
                    format=ScalarType.UINT16,
                ),
                StructuredFieldDescriptor(
                    name="sample_interval",
                    offset=4,
                    format=ScalarType.UINT16,
                ),
                StructuredFieldDescriptor(
                    name="sample_interval_orig",
                    offset=6,
                    format=ScalarType.UINT16,
                ),
                StructuredFieldDescriptor(
                    name="samples_per_trace",
                    offset=8,
                    format=ScalarType.UINT16,
                ),
                StructuredFieldDescriptor(
                    name="samples_per_trace_orig",
                    offset=10,
                    format=ScalarType.UINT16,
                ),
            ],
            item_size=16,
            offset=40,
        ),
        trace=TraceDescriptor(
            header_descriptor=StructuredDataTypeDescriptor(
                fields=[
                    StructuredFieldDescriptor(
                        name="samples_per_trace",
                        offset=4,
                        format=ScalarType.INT32,
                    ),
                    StructuredFieldDescriptor(
                        name="sample_interval",
                        offset=8,
                        format=ScalarType.INT32,
                    ),
                    StructuredFieldDescriptor(
                        name="custom_header",
                        offset=12,
                        format=ScalarType.INT16,
                    ),
                ],
                item_size=16,
            ),
            data_descriptor=TraceDataDescriptor(
                format=ScalarType.IBM32,  # noqa: A003
            ),
        ),
    )


text_header_expected = "C01 Test header     \nC02 This is line 2  "


@pytest.fixture()
def mock_segy_file(mock_segy_spec: SegyDescriptor) -> SegyFile:
    """Create a mock file in memory and open it with SegyFile and return fixture."""
    factory = SegyFactory(mock_segy_spec, sample_interval=2000, samples_per_trace=11)
    fs = filesystem("memory")
    file = fs._open("test.sgy", mode="wb")

    file.write(factory.create_textual_header(text_header_expected))
    file.write(factory.create_binary_header())

    num_traces = 15
    headers = factory.create_trace_header_template(num_traces)
    headers["custom_header"] = np.arange(num_traces)

    data = factory.create_trace_data_template(num_traces)
    data[:] = np.arange(num_traces)[..., None]

    file.write(factory.create_traces(headers, data))

    return SegyFile("memory://test.sgy", mock_segy_spec)


class TestSegyFactoryFile:
    """Test if file created with SegyFactory has correct values."""

    def test_text_header(self, mock_segy_file: SegyFile) -> None:
        """Check that the text header is correct."""
        assert mock_segy_file.text_header == text_header_expected

    def test_binary_header(self, mock_segy_file: SegyFile) -> None:
        """Check that the binary header is correct."""
        assert mock_segy_file.binary_header.item() == (0, 2000, 2000, 11, 11)

    def test_trace_header(self, mock_segy_file: SegyFile) -> None:
        """Check that the trace header is correct."""
        assert mock_segy_file.header[7].item() == (11, 2000, 7)

    @pytest.mark.parametrize("trace_idx", [0, 5, 11])
    def test_trace_data(self, mock_segy_file: SegyFile, trace_idx: int) -> None:
        """Check that the trace data is correct."""
        assert (mock_segy_file.data[trace_idx] == trace_idx).all()

    @pytest.mark.parametrize("trace_idx", [0, 5, 11])
    def test_trace(self, mock_segy_file: SegyFile, trace_idx: int) -> None:
        """Check that the trace header + data accessor is correct."""
        assert mock_segy_file.trace[trace_idx]["header"].item() == (11, 2000, trace_idx)
        assert (mock_segy_file.trace[trace_idx]["data"] == trace_idx).all()


class TestSegyFactoryExceptions:
    """Test the exceptions to SegyFactory creation methods."""

    def test_create_trace_incorrect_ndim(self, mock_segy_spec: SegyDescriptor) -> None:
        """Check if trace dimensions are wrong."""
        factory = SegyFactory(mock_segy_spec, sample_interval=2, samples_per_trace=5)
        header_1d = factory.create_trace_header_template(5, fill=False)
        array_4d = np.empty(shape=(5, 5, 5, 5))

        with pytest.raises(AttributeError, match="Data array must be 2-dimensional"):
            factory.create_traces(header_1d, array_4d)

    def test_create_data_nsamp_mismatch(self, mock_segy_spec: SegyDescriptor) -> None:
        """Check if trace number of samples are wrong."""
        factory = SegyFactory(mock_segy_spec, sample_interval=2, samples_per_trace=5)
        header_1d = factory.create_trace_header_template(size=5, fill=False)
        array_2d = np.empty(shape=(5, 11))

        with pytest.raises(ValueError, match="Trace length must be"):
            factory.create_traces(header_1d, array_2d)

    def test_create_header_data_mismatch(self, mock_segy_spec: SegyDescriptor) -> None:
        """Check if headers and traces are different sizes."""
        factory = SegyFactory(mock_segy_spec, sample_interval=2, samples_per_trace=11)
        header_1d = factory.create_trace_header_template(size=5, fill=False)
        array_2d = factory.create_trace_data_template(size=7, fill=False)

        with pytest.raises(ValueError, match="same number of rows as data array"):
            factory.create_traces(header_1d, array_2d)
