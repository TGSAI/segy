"""Test the usage of SegyFile class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.testing import assert_array_equal

from segy import SegyFactory
from segy import SegyFile
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import registry

if TYPE_CHECKING:
    from fsspec.implementations.memory import MemoryFileSystem


SAMPLE_RATE = 2000
SAMPLES_PER_TRACE = 21
NUM_TRACES = 15

EXPECTED_SAMPLE_LABELS = range(0, SAMPLES_PER_TRACE * SAMPLE_RATE, SAMPLE_RATE)


@pytest.fixture(params=[Endianness.BIG, Endianness.LITTLE])
def endianness(request: pytest.FixtureRequest) -> Endianness:
    """Fixture for testing endianness."""
    return request.param


@pytest.fixture(params=[SegyStandard.REV0, SegyStandard.REV1])
def segy_standard(request: pytest.FixtureRequest) -> SegyStandard:
    """Fixture for testing different SEG-Y standards."""
    return request.param


@pytest.fixture(params=[ScalarType.IBM32, ScalarType.FLOAT32])
def sample_format(request: pytest.FixtureRequest) -> ScalarType:
    """Fixture for testing different sample formats."""
    return request.param


@pytest.fixture()
def mock_segy_uri(
    mock_filesystem: MemoryFileSystem,
    segy_standard: SegyStandard,
    endianness: Endianness,
    sample_format: ScalarType,
) -> str:
    """Fixture for mocking a SEG-Y file at a in memory URI."""
    spec = registry.get_spec(segy_standard)
    spec.endianness = endianness
    spec.trace.sample_descriptor.format = sample_format

    factory = SegyFactory(
        spec=spec,
        sample_interval=SAMPLE_RATE,
        samples_per_trace=SAMPLES_PER_TRACE,
    )

    text_file_hdr_bytes = factory.create_textual_header()
    bin_file_hdr_bytes = factory.create_binary_header()

    headers = factory.create_trace_header_template(NUM_TRACES)
    samples = factory.create_trace_sample_template(NUM_TRACES)
    trace_bytes = factory.create_traces(headers, samples)

    uri = f"memory://segyfile_{segy_standard.name}_{endianness.value}"
    fp = mock_filesystem.open(uri, mode="wb")

    fp.write(text_file_hdr_bytes)
    fp.write(bin_file_hdr_bytes)
    fp.write(trace_bytes)

    return uri


@pytest.fixture()
def mock_segy_file(mock_segy_uri: str) -> SegyFile:
    """Fixture to get instances of SegyFile from mock binary files."""
    return SegyFile(mock_segy_uri)


class TestSegyFile:
    """Test the usage of SegyFile class."""

    def test_segy_rev0(
        self,
        mock_segy_file: SegyFile,
    ) -> None:
        """Tests various attributes and methods of a SegyFile with Rev 0 specs."""
        assert_array_equal(mock_segy_file.sample_labels, EXPECTED_SAMPLE_LABELS)
        #
        # assert "This is a sample text header" in mock_segy_rev0.text_header
        # assert mock_segy_rev0.num_traces == num_traces
        # assert mock_segy_rev0.samples_per_trace == num_samples
        # assert mock_segy_rev0.num_ext_text == 0
        #
        # assert mock_segy_rev0.spec.trace.sample_descriptor.samples == num_samples
        # assert len(mock_segy_rev0.sample[:]) == num_traces
        # assert (
        #     mock_segy_rev0.header[:]["trace_seq_line"] == list(range(1, num_traces + 1))
        # ).all()
        #
        # expected_value = 1.0
        # assert_array_equal(mock_segy_rev0.sample[:], expected_value)
        # assert_array_equal(mock_segy_rev0.trace[:].header, mock_segy_rev0.header[:])
        # assert_array_equal(mock_segy_rev0.trace[:].sample, mock_segy_rev0.sample[:])
