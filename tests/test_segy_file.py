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
from segy.config import BinaryHeaderSettings
from segy.config import SegySettings
from segy.exceptions import EndiannessInferenceError
from segy.exceptions import SegyFileSpecMismatchError
from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards import get_segy_standard
from segy.standards.codes import DataSampleFormatCode

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
    num_extended_text_headers: int = 0,
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

    binary_update_dict = {}
    if num_extended_text_headers > 0 and segy_standard > SegyStandard.REV0:
        binary_update_dict["num_extended_text_headers"] = num_extended_text_headers

    text_file_hdr_bytes = factory.create_textual_header()
    bin_file_hdr_bytes = factory.create_binary_header(binary_update_dict)

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

    for _ in range(num_extended_text_headers):
        fp.write(factory.create_textual_header())

    fp.write(trace_bytes)

    return SegyFileTestConfig(
        uri, segy_standard, endianness, sample_format, header_data, sample_data
    )


class TestSegyFile:
    """Test the usage of SegyFile class."""

    @pytest.mark.parametrize(
        "standard",
        [SegyStandard.REV0, SegyStandard.REV1, SegyStandard.REV2, SegyStandard.REV21],
    )
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

        # Test the other case where we exactly specify the spec.
        spec_expected = get_segy_standard(standard)
        spec_expected.endianness = endianness
        spec_expected.trace.data.format = sample_format
        segy_file_expected = SegyFile(test_config.uri, spec=spec_expected)

        assert segy_file_expected.spec == segy_file.spec

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

        expected_sample_format = DataSampleFormatCode[test_config.sample_format.name]
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

        # Test random access
        index = [0, 2, 4]
        traces = segy_file.trace[index]
        assert_array_equal(traces.header, test_config.expected_headers[index])
        assert_array_almost_equal(traces.sample, test_config.expected_samples[index])

        # Test reverse order random access
        index = [5, 3, 0]
        traces = segy_file.trace[index]
        assert_array_equal(traces.header, test_config.expected_headers[index])
        assert_array_almost_equal(traces.sample, test_config.expected_samples[index])

        # Test random access with duplicates
        index = [5, 3, 0, 5, 5, 3]
        traces = segy_file.trace[index]
        assert_array_equal(traces.header, test_config.expected_headers[index])
        assert_array_almost_equal(traces.sample, test_config.expected_samples[index])


class TestSegyFileExceptions:
    """Test exceptions for SegyFile."""

    def test_endian_code_err_handling(self, mock_filesystem: MemoryFileSystem) -> None:
        """Test bad values in binary header endian code triggering legacy inference."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=SegyStandard.REV1,
        )

        with mock_filesystem.open(test_config.uri, mode="r+b") as fp:
            fp.seek(3296)
            fp.write(struct.pack(">I", 999))

        file = SegyFile(test_config.uri)

        assert file.spec.endianness == Endianness.BIG
        assert file.spec.segy_standard == SegyStandard.REV1

    def test_endian_infer_failure(self, mock_filesystem: MemoryFileSystem) -> None:
        """Test bad values in binary header triggering spec inference failure."""
        test_config = generate_test_segy(filesystem=mock_filesystem)

        with mock_filesystem.open(test_config.uri, mode="r+b") as fp:
            fp.seek(3224)
            fp.write(struct.pack(">H", 420))  # invalid sample format

        error_message = "Endianness inference failed after attempting all methods."
        with pytest.raises(EndiannessInferenceError, match=error_message):
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
        assert segy_file.binary_header["segy_revision_major"] == 0

    def test_revision_endian_override(self, mock_filesystem: MemoryFileSystem) -> None:
        """Make big-rev0 file and open it as little-rev1 from settings override."""
        test_config = generate_test_segy(
            filesystem=mock_filesystem,
            segy_standard=SegyStandard.REV0,
            endianness=Endianness.BIG,
        )
        # Now ensure overriding endian works. This should raise an
        # error because with little endian the sample format will be
        # interpreted incorrectly from binary header.
        settings_dict_both = {"endianness": "little"}
        settings = SegySettings.model_validate(settings_dict_both)

        with pytest.raises(ValueError, match="is not a valid DataSampleFormatCode"):
            SegyFile(test_config.uri, settings=settings)

    @pytest.mark.parametrize("num_extended_text_headers", [0, 2])
    def test_ext_text_header_override(
        self,
        mock_filesystem: MemoryFileSystem,
        num_extended_text_headers: int,
    ) -> None:
        """Test if settings override for extended header count work for SegyFile."""
        # Create file with zero extended text headers
        test_config = generate_test_segy(
            mock_filesystem,
            segy_standard=SegyStandard.REV1,
            num_extended_text_headers=num_extended_text_headers,
        )

        # Mess up and write 42 headers
        with mock_filesystem.open(test_config.uri, mode="r+b") as fp:
            fp.seek(3504)
            fp.write(struct.pack(">H", 42))  # invalid sample format

        # Ensure failure as is
        error_message = "doesn't match parsed spec"
        with pytest.raises(SegyFileSpecMismatchError, match=error_message):
            SegyFile(test_config.uri)

        # Make it work with override
        bin_override = BinaryHeaderSettings(ext_text_header=num_extended_text_headers)
        settings = SegySettings(binary=bin_override)
        segy_file = SegyFile(test_config.uri, settings=settings)

        assert segy_file.num_ext_text == num_extended_text_headers
