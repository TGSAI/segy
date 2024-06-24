"""Test the usage of SegyFile class."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from segy import SegyFactory
from segy import SegyFile
from segy.config import SegySettings
from segy.exceptions import EndiannessInferenceError
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.mapping import SEGY_FORMAT_MAP

if TYPE_CHECKING:
    from typing import Any

    from fsspec import AbstractFileSystem
    from fsspec.implementations.memory import MemoryFileSystem
    from numpy.typing import NDArray

SAMPLE_INTERVAL = 2000
SAMPLES_PER_TRACE = 21
NUM_TRACES = 15

EXPECTED_SAMPLE_LABELS = range(0, SAMPLES_PER_TRACE * SAMPLE_INTERVAL, SAMPLE_INTERVAL)


@dataclass
class SegyFileTestConfig:
    """Configuration container for testing SegyFile class."""

    uri: str
    segy_standard: SegyStandard
    endianness: Endianness
    sample_format: ScalarType
    expected_headers: NDArray[Any]
    expected_samples: NDArray[Any]


def generate_test_trace_data(
    factory: SegyFactory,
    num_traces: int,
) -> tuple[NDArray[np.void], NDArray[Any]]:
    """Generate random header and sample data for testing."""
    rng = np.random.default_rng()
    header_spec = factory.spec.trace.header
    data_spec = factory.spec.trace.data

    header_dtype = header_spec.dtype.newbyteorder("=")
    header_arr = np.empty(num_traces, dtype=header_dtype)
    for field in header_spec.fields:
        random_field_data = rng.uniform(-128, 127, size=num_traces)
        header_arr[field.name] = random_field_data.astype(field.format)

    # Cast to float32 if IBM.
    if data_spec.format == ScalarType.IBM32:
        sample_dtype = np.dtype("float32")
    else:
        sample_dtype = np.dtype(data_spec.format)
    sample_shape = (num_traces, SAMPLES_PER_TRACE)
    sample_arr = np.empty(shape=sample_shape, dtype=sample_dtype)
    random_sample_data = rng.normal(size=sample_shape)
    sample_arr[:] = random_sample_data.astype("float32")

    return header_arr, sample_arr


def generate_test_segy(
    filesystem: AbstractFileSystem,
    segy_standard: SegyStandard = SegyStandard.REV0,
    endianness: Endianness = Endianness.BIG,
    sample_format: ScalarType = ScalarType.IBM32,
) -> SegyFileTestConfig:
    """Function for mocking a SEG-Y file with in memory URI."""
    spec = get_segy_standard(segy_standard)
    spec.endianness = endianness
    spec.trace.data.format = sample_format

    factory = SegyFactory(
        spec=spec,
        sample_interval=SAMPLE_INTERVAL,
        samples_per_trace=SAMPLES_PER_TRACE,
    )

    text_file_hdr_bytes = factory.create_textual_header()
    bin_file_hdr_bytes = factory.create_binary_header()

    headers = factory.create_trace_header_template(NUM_TRACES)
    samples = factory.create_trace_sample_template(NUM_TRACES)

    header_data, sample_data = generate_test_trace_data(factory, NUM_TRACES)
    headers[:] = header_data
    samples[:] = sample_data

    trace_bytes = factory.create_traces(headers, samples)

    uri = f"memory://{segy_standard.name}_{endianness.value}_{sample_format.value}.segy"
    fp = filesystem.open(uri, mode="wb")

    fp.write(text_file_hdr_bytes)
    fp.write(bin_file_hdr_bytes)
    fp.write(trace_bytes)

    return SegyFileTestConfig(
        uri, segy_standard, endianness, sample_format, header_data, sample_data
    )


class TestSegyFile:
    """Test the usage of SegyFile class."""

    @pytest.mark.parametrize("standard", [SegyStandard.REV0, SegyStandard.REV1])
    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize("sample_format", [ScalarType.IBM32, ScalarType.FLOAT32])
    def test_infer_spec(
        self,
        mock_filesystem: MemoryFileSystem,
        endianness: Endianness,
        standard: SegyStandard,
        sample_format: ScalarType,
    ) -> None:
        """Tests various attributes and methods of a SegyFile with Rev 0 specs."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=standard,
            endianness=endianness,
            sample_format=sample_format,
        )

        segy_file = SegyFile(test_config.uri)

        # Assert spec
        trace_data_spec = segy_file.spec.trace.data
        assert segy_file.spec.segy_standard == test_config.segy_standard
        assert segy_file.spec.endianness == test_config.endianness
        assert trace_data_spec.format == test_config.sample_format

        # Assert attributes
        assert segy_file.num_traces == NUM_TRACES
        assert segy_file.samples_per_trace == SAMPLES_PER_TRACE
        assert segy_file.num_ext_text == 0
        assert_array_equal(segy_file.sample_labels, EXPECTED_SAMPLE_LABELS)

        # Check if JSON-able dict representation is valid
        assert segy_file.spec._repr_json_() == segy_file.spec.model_dump(mode="json")

    def test_text_file_header(
        self, mock_filesystem: MemoryFileSystem, default_text: str
    ) -> None:
        """Test text file header attribute."""
        test_config = generate_test_segy(mock_filesystem)

        segy_file = SegyFile(test_config.uri)

        # Compare first 5 lines because rest is dynamic.
        assert segy_file.text_header[:400] == default_text[:400]

    @pytest.mark.parametrize("standard", [SegyStandard.REV0, SegyStandard.REV1])
    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize("sample_format", [ScalarType.IBM32, ScalarType.INT32])
    def test_binary_file_header(
        self,
        mock_filesystem: MemoryFileSystem,
        endianness: Endianness,
        standard: SegyStandard,
        sample_format: ScalarType,
    ) -> None:
        """Test binary file header values."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=standard,
            endianness=endianness,
            sample_format=sample_format,
        )

        segy_file = SegyFile(test_config.uri)
        binary_header = segy_file.binary_header

        expected_sample_format = SEGY_FORMAT_MAP[test_config.sample_format]
        assert binary_header["sample_interval"] == SAMPLE_INTERVAL
        assert binary_header["orig_sample_interval"] == SAMPLE_INTERVAL
        assert binary_header["samples_per_trace"] == SAMPLES_PER_TRACE
        assert binary_header["orig_samples_per_trace"] == SAMPLES_PER_TRACE
        assert binary_header["data_sample_format"] == expected_sample_format

    @pytest.mark.parametrize("standard", [SegyStandard.REV0, SegyStandard.REV1])
    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    def test_trace_header_accessor(
        self,
        mock_filesystem: MemoryFileSystem,
        endianness: Endianness,
        standard: SegyStandard,
    ) -> None:
        """Test trace header accessor and values."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=standard,
            endianness=endianness,
        )

        segy_file = SegyFile(test_config.uri)

        assert_array_equal(segy_file.header[:], test_config.expected_headers)

    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize("sample_format", [ScalarType.IBM32, ScalarType.FLOAT32])
    def test_trace_sample_accessor(
        self,
        mock_filesystem: MemoryFileSystem,
        endianness: Endianness,
        sample_format: ScalarType,
    ) -> None:
        """Test trace sample accessor and values."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            endianness=endianness,
            sample_format=sample_format,
        )

        segy_file = SegyFile(test_config.uri)

        assert_array_almost_equal(segy_file.sample[:], test_config.expected_samples)

    @pytest.mark.parametrize("standard", [SegyStandard.REV0, SegyStandard.REV1])
    @pytest.mark.parametrize("endianness", [Endianness.BIG, Endianness.LITTLE])
    @pytest.mark.parametrize("sample_format", [ScalarType.IBM32, ScalarType.UINT8])
    def test_trace_accessor(
        self,
        mock_filesystem: MemoryFileSystem,
        endianness: Endianness,
        standard: SegyStandard,
        sample_format: ScalarType,
    ) -> None:
        """Test trace accessor and values."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=standard,
            endianness=endianness,
            sample_format=sample_format,
        )

        segy_file = SegyFile(test_config.uri)
        traces = segy_file.trace[:]

        assert_array_equal(traces.header, test_config.expected_headers)
        assert_array_almost_equal(traces.sample, test_config.expected_samples)


class TestSegyFileExceptions:
    """Test exceptions for SegyFile."""

    @pytest.mark.parametrize(
        ("standard_override", "sample_increment_override", "sample_format_override"),
        [
            (0.5, 2000, 1),  # bad revision, ok increment, ok format
            (1.0, -100, 1),  # ok revision, bad increment, ok format
            (1.0, 2000, 100),  # ok revision, ok increment, bad format
        ],
    )
    def test_spec_inference_failure(
        self,
        mock_filesystem: MemoryFileSystem,
        standard_override: float,
        sample_format_override: int,
        sample_increment_override: int,
    ) -> None:
        """Test bad values in binary header triggering spec inference error."""
        test_config = generate_test_segy(filesystem=mock_filesystem)

        fp = mock_filesystem.open(test_config.uri, mode="r+b")
        fp.seek(3216)
        fp.write(struct.pack(">h", sample_increment_override))
        fp.seek(3224)
        fp.write(struct.pack(">h", sample_format_override))
        fp.seek(3500)
        fp.write(struct.pack(">h", int(standard_override * 256)))
        fp.close()

        with pytest.raises(
            EndiannessInferenceError, match="Can't infer file endianness"
        ):
            SegyFile(test_config.uri)


class TestSegyFileSettingsOverride:
    """Test if settings overrides work fine for SegyFile."""

    def test_revision_override(self, mock_filesystem: MemoryFileSystem) -> None:
        """Make rev0 file and open it as rev1 from settings override."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem, segy_standard=SegyStandard.REV0
        )

        settings = SegySettings.model_validate({"binary": {"revision": 1.0}})
        segy_file = SegyFile(test_config.uri, settings=settings)

        assert segy_file.spec.segy_standard == SegyStandard.REV1

    def test_revision_endian_override(self, mock_filesystem: MemoryFileSystem) -> None:
        """Make big-rev0 file and open it as little-rev1 from settings override."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=SegyStandard.REV0,
            endianness=Endianness.BIG,
        )

        settings_dict = {"binary": {"revision": 1.0}, "endianness": "little"}
        settings = SegySettings.model_validate(settings_dict)
        segy_file = SegyFile(test_config.uri, settings=settings)

        assert segy_file.spec.segy_standard == SegyStandard.REV1
        assert segy_file.spec.endianness == Endianness.LITTLE
        # Rev1 should have below field, but the value will be zero
        assert "segy_revision" in segy_file.binary_header.dtype.names
        assert segy_file.binary_header["segy_revision"] == 0

    @pytest.mark.parametrize("num_ext_text", [1])
    def test_ext_text_header_override(
        self, mock_filesystem: MemoryFileSystem, num_ext_text: int
    ) -> None:
        """Test if settings override for extended header count work for SegyFile."""
        test_config = generate_test_segy(
            mock_filesystem,
            segy_standard=SegyStandard.REV1,
        )

        settings = SegySettings.model_validate(
            {"binary": {"ext_text_header": num_ext_text}}
        )
        segy_file = SegyFile(test_config.uri, settings=settings)

        assert segy_file.num_ext_text == num_ext_text
