"""Test the usage of SegyFile class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from numpy.testing import assert_array_equal

from segy import SegyFactory
from segy import SegyFile
from segy.factory import DEFAULT_TEXT_HEADER
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import registry
from segy.standards.mapping import SEGY_FORMAT_MAP

if TYPE_CHECKING:
    from fsspec.implementations.memory import MemoryFileSystem


SAMPLE_INTERVAL = 2000
SAMPLES_PER_TRACE = 21
NUM_TRACES = 15

EXPECTED_SAMPLE_LABELS = range(0, SAMPLES_PER_TRACE * SAMPLE_INTERVAL, SAMPLE_INTERVAL)


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


@dataclass
class SegyFileTestConfig:
    """Configuration for testing SegyFile class."""

    uri: str
    segy_standard: SegyStandard
    endianness: Endianness
    sample_format: ScalarType


@pytest.fixture()
def test_config(
    mock_filesystem: MemoryFileSystem,
    segy_standard: SegyStandard,
    endianness: Endianness,
    sample_format: ScalarType,
) -> SegyFileTestConfig:
    """Fixture for mocking a SEG-Y file at a in memory URI."""
    spec = registry.get_spec(segy_standard)
    spec.endianness = endianness
    spec.trace.sample_descriptor.format = sample_format

    factory = SegyFactory(
        spec=spec,
        sample_interval=SAMPLE_INTERVAL,
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

    return SegyFileTestConfig(uri, segy_standard, endianness, sample_format)


class TestSegyFile:
    """Test the usage of SegyFile class."""

    def test_infer_spec(self, test_config: SegyFileTestConfig) -> None:
        """Tests various attributes and methods of a SegyFile with Rev 0 specs."""
        segy_file = SegyFile(test_config.uri)

        # Assert spec
        sample_descriptor = segy_file.spec.trace.sample_descriptor
        assert segy_file.spec.segy_standard == test_config.segy_standard
        assert segy_file.spec.endianness == test_config.endianness
        assert sample_descriptor.format == test_config.sample_format

        # Assert attributes
        assert segy_file.num_traces == NUM_TRACES
        assert segy_file.samples_per_trace == SAMPLES_PER_TRACE
        assert segy_file.num_ext_text == 0
        assert_array_equal(segy_file.sample_labels, EXPECTED_SAMPLE_LABELS)

    def test_text_file_header(self, test_config: SegyFileTestConfig) -> None:
        """Test text file header attribute."""
        segy_file = SegyFile(test_config.uri)

        assert segy_file.text_header == DEFAULT_TEXT_HEADER

    def test_binary_file_header(self, test_config: SegyFileTestConfig) -> None:
        """Test binary file header values."""
        segy_file = SegyFile(test_config.uri)
        binary_header = segy_file.binary_header

        expected_sample_format = SEGY_FORMAT_MAP[test_config.sample_format]
        assert binary_header["sample_interval"] == SAMPLE_INTERVAL
        assert binary_header["sample_interval_orig"] == SAMPLE_INTERVAL
        assert binary_header["samples_per_trace"] == SAMPLES_PER_TRACE
        assert binary_header["samples_per_trace_orig"] == SAMPLES_PER_TRACE
        assert binary_header["data_sample_format"] == expected_sample_format
