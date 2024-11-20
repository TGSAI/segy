"""Tests for the SegyFactory used in creation."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from segy.factory import SegyFactory
from segy.ibm import ieee2ibm
from segy.schema import Endianness
from segy.schema import HeaderField
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.schema import TextHeaderEncoding
from segy.standards.codes import DataSampleFormatCode
from segy.standards.minimal import minimal_segy


@dataclass
class SegyFactoryTestConfig:
    """Dataclass to configure common test patterns."""

    segy_standard: SegyStandard | None
    endianness: Endianness
    sample_interval: int
    samples_per_trace: int


SEGY_FACTORY_TEST_CONFIGS = [
    SegyFactoryTestConfig(None, Endianness.BIG, 2000, 51),
    SegyFactoryTestConfig(SegyStandard.REV0, Endianness.BIG, 2000, 51),
    SegyFactoryTestConfig(SegyStandard.REV1, Endianness.LITTLE, 3000, 1),
    SegyFactoryTestConfig(SegyStandard.REV0, Endianness.BIG, 5000, 10),
]


@pytest.fixture(params=SEGY_FACTORY_TEST_CONFIGS)
def mock_segy_factory(request: pytest.FixtureRequest) -> SegyFactory:
    """Generates the test cases for SegyFactory.

    We start with a minimal SEG-Y spec and modify its properties to generate various
    files parametrically. `SEGY_FACTORY_TEST_CONFIGS` defines the base configuration.
    """
    test_config = request.param
    spec = minimal_segy

    # Set file wide attributes
    spec.endianness = test_config.endianness
    spec.segy_standard = test_config.segy_standard

    # Shrink trace headers to 16-bytes and add a few fields
    spec.trace.header.item_size = 32
    spec.trace.header.fields = [
        HeaderField(name="field1", format=ScalarType.INT8, byte=3),
        HeaderField(name="field2", format=ScalarType.INT32, byte=5),
        HeaderField(name="field3", format=ScalarType.UINT8, byte=11),
    ]

    return SegyFactory(
        spec,
        sample_interval=test_config.sample_interval,
        samples_per_trace=test_config.samples_per_trace,
    )


@pytest.mark.parametrize(
    "encoding", [TextHeaderEncoding.EBCDIC, TextHeaderEncoding.ASCII]
)
class TestTextualFileHeader:
    """Tests related to the text header writing."""

    def test_text_header_encoding(
        self, encoding: TextHeaderEncoding, default_text: str
    ) -> None:
        """Tests the text header encoding/decoding works correctly."""
        segy_spec = minimal_segy
        segy_spec.text_header.encoding = encoding
        factory = SegyFactory(segy_spec)

        text_bytes = factory.create_textual_header()
        text_actual = segy_spec.text_header.decode(text_bytes)

        # Compare first 5 lines because rest is dynamic.
        assert text_actual[:400] == default_text[:400]

    def test_text_header_custom(self, encoding: TextHeaderEncoding) -> None:
        """Test if custom header is encoded/decoded correctly."""
        segy_spec = minimal_segy
        segy_spec.text_header.encoding = encoding
        factory = SegyFactory(segy_spec)

        chars = string.ascii_uppercase + string.ascii_lowercase + string.ascii_letters
        text_rows = []
        for _ in range(40):
            row_string = "".join([random.choice(chars) for _ in range(80)])  # noqa: S311
            text_rows.append(row_string)
        text_expected = "\n".join(text_rows)

        text_bytes = factory.create_textual_header(text_expected)
        text_actual = segy_spec.text_header.decode(text_bytes)

        assert text_actual == text_expected


class TestBinaryFileHeader:
    """Tests related to the text header writing."""

    @pytest.mark.parametrize(
        "sample_format", [ScalarType.FLOAT32, ScalarType.IBM32, ScalarType.INT16]
    )
    def test_binary_header_default(
        self, mock_segy_factory: SegyFactory, sample_format: ScalarType
    ) -> None:
        """Ensure the binary header is properly encoded and serialized."""
        mock_segy_factory.spec.trace.data.format = sample_format

        binary_bytes = mock_segy_factory.create_binary_header()

        bin_spec = mock_segy_factory.spec.binary_header
        binary_actual = np.frombuffer(binary_bytes, dtype=bin_spec.dtype)
        binary_expected = (
            mock_segy_factory.sample_interval,
            mock_segy_factory.sample_interval,
            mock_segy_factory.samples_per_trace,
            mock_segy_factory.samples_per_trace,
            DataSampleFormatCode[mock_segy_factory.sample_format.name],
            mock_segy_factory.segy_revision.value * 256,
            0,  # fixed length trace flag
            0,  # extended text headers
        )
        assert binary_actual.item() == binary_expected

    def test_binary_header_custom(self, mock_segy_factory: SegyFactory) -> None:
        """Test if binary header encoding field update."""
        mock_segy_factory.spec.trace.data.format = ScalarType.IBM32

        update = {
            "fixed_length_trace_flag": 1,
            "num_extended_text_headers": 2,
        }
        binary_bytes = mock_segy_factory.create_binary_header(update=update)

        bin_spec = mock_segy_factory.spec.binary_header
        binary_actual = np.frombuffer(binary_bytes, dtype=bin_spec.dtype)
        binary_expected = (
            mock_segy_factory.sample_interval,
            mock_segy_factory.sample_interval,
            mock_segy_factory.samples_per_trace,
            mock_segy_factory.samples_per_trace,
            DataSampleFormatCode[mock_segy_factory.sample_format.name],
            mock_segy_factory.segy_revision.value * 256,
            1,  # fixed length trace flag
            2,  # extended text headers
        )
        assert binary_actual.item() == binary_expected


@pytest.mark.parametrize("num_traces", [1, 42])
class TestSegyFactoryTraces:
    """Ensure the trace headers are properly encoded and serialized."""

    def test_trace_header_template(
        self, mock_segy_factory: SegyFactory, num_traces: int
    ) -> None:
        """Test if the trace header template is correct."""
        headers = mock_segy_factory.create_trace_header_template(num_traces)

        header_spec = mock_segy_factory.spec.trace.header
        assert headers.size == num_traces
        assert headers.dtype == header_spec.dtype.newbyteorder("<")

    def test_trace_header_template_with_sample_info(
        self, mock_segy_factory: SegyFactory, num_traces: int
    ) -> None:
        """Test if the trace header template is correct with sample info."""
        mock_segy_factory.spec.trace.header.item_size = 26
        # fmt: off
        mock_segy_factory.spec.trace.header.fields += [
            HeaderField(name="sample_interval", format=ScalarType.INT16, byte=13),
            HeaderField(name="samples_per_trace", format=ScalarType.INT16, byte=25),
        ]
        # fmt: on

        headers = mock_segy_factory.create_trace_header_template(num_traces)

        header_spec = mock_segy_factory.spec.trace.header
        expected_sample_info = mock_segy_factory.sample_interval
        expected_samples_per_trace = mock_segy_factory.samples_per_trace
        assert headers.size == num_traces
        assert headers.dtype == header_spec.dtype.newbyteorder("<")
        assert_array_equal(headers["sample_interval"], expected_sample_info)
        assert_array_equal(headers["samples_per_trace"], expected_samples_per_trace)

    @pytest.mark.parametrize(
        "sample_format", [ScalarType.FLOAT32, ScalarType.IBM32, ScalarType.INT16]
    )
    def test_trace_sample_template(
        self, mock_segy_factory: SegyFactory, num_traces: int, sample_format: ScalarType
    ) -> None:
        """Test if the trace sample template is correct."""
        mock_segy_factory.spec.trace.data.format = sample_format

        samples = mock_segy_factory.create_trace_sample_template(num_traces)

        n_samples = mock_segy_factory.samples_per_trace

        if mock_segy_factory.sample_format == ScalarType.IBM32:
            expected_dtype = np.dtype("float32")
        else:
            expected_dtype = mock_segy_factory.sample_format.dtype

        expected_shape = (num_traces, n_samples)
        assert samples.dtype == expected_dtype
        assert samples.shape == expected_shape

    @pytest.mark.parametrize(
        "sample_format", [ScalarType.FLOAT32, ScalarType.IBM32, ScalarType.INT16]
    )
    def test_trace_serialize(
        self, mock_segy_factory: SegyFactory, num_traces: int, sample_format: ScalarType
    ) -> None:
        """Test if the trace serialization is correct."""
        mock_segy_factory.spec.trace.data.format = sample_format

        # Generate random data
        rng = np.random.default_rng()
        samples_per_trace = mock_segy_factory.samples_per_trace
        shape = (num_traces, samples_per_trace)
        rand_samples = np.float32(255 * rng.random(size=shape))
        rand_fields = {}
        for field in mock_segy_factory.spec.trace.header.fields:
            field_data = rng.integers(-100, 100, dtype="int8", size=num_traces)
            rand_fields[field.name] = field_data

        # Fill in actual and generate bytes
        headers = mock_segy_factory.create_trace_header_template(num_traces)
        samples = mock_segy_factory.create_trace_sample_template(num_traces)
        samples[:] = rand_samples
        for field_name, values in rand_fields.items():
            headers[field_name] = values
        trace_bytes = mock_segy_factory.create_traces(headers, samples)

        # Fill in expected and assertions
        # 1. handle ibm float
        # 2. fill in trace struct
        # 3. handle endianness
        if mock_segy_factory.sample_format == ScalarType.IBM32:
            rand_samples = ieee2ibm(rand_samples)
        trace_dtype_native = mock_segy_factory.spec.trace.dtype.newbyteorder("=")
        expected_traces = np.zeros(shape=num_traces, dtype=trace_dtype_native)
        expected_traces["data"] = rand_samples
        for field_name, values in rand_fields.items():
            expected_traces["header"][field_name] = values
        if mock_segy_factory.spec.endianness == Endianness.BIG:
            expected_traces = expected_traces.byteswap(inplace=True)
            expected_dtype = expected_traces.dtype.newbyteorder(">")
            expected_traces = expected_traces.view(expected_dtype)

        assert trace_bytes == expected_traces.tobytes()


class TestSegyFactoryExceptions:
    """Test the exceptions to SegyFactory creation methods."""

    def test_create_trace_incorrect_ndim(self) -> None:
        """Check if trace dimensions are wrong."""
        spec = minimal_segy
        factory = SegyFactory(spec, sample_interval=2, samples_per_trace=5)

        header_1d = factory.create_trace_header_template(5)
        array_4d = np.empty(shape=(5, 5, 5, 5))

        with pytest.raises(AttributeError, match="Data array must be 2-dimensional"):
            factory.create_traces(header_1d, array_4d)

    def test_create_sample_num_samples_mismatch(self) -> None:
        """Check if trace number of samples are wrong."""
        spec = minimal_segy
        factory = SegyFactory(spec, sample_interval=2, samples_per_trace=5)

        header_1d = factory.create_trace_header_template(size=5)
        array_2d = np.empty(shape=(5, 11))

        with pytest.raises(ValueError, match="Trace length must be"):
            factory.create_traces(header_1d, array_2d)

    def test_create_header_sample_mismatch(self) -> None:
        """Check if headers and traces are different sizes."""
        spec = minimal_segy
        factory = SegyFactory(spec, sample_interval=2, samples_per_trace=11)

        header_1d = factory.create_trace_header_template(size=5)
        array_2d = factory.create_trace_sample_template(size=7)

        with pytest.raises(ValueError, match="same number of rows as data array"):
            factory.create_traces(header_1d, array_2d)
