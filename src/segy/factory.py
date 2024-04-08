"""Factory methods for SEG-Y file creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from segy.schema import Endianness
from segy.schema import ScalarType
from segy.schema import SegyStandard
from segy.standards.mapping import SEGY_FORMAT_MAP
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from segy.schema import SegyDescriptor


DEFAULT_TEXT_HEADER_LINES = [
    "C01 File written by the open-source segy library.",
    "C02",
    "C03 Website: https://segy.readthedocs.io",
    "C04 Source: https://github.com/TGSAI/segy",
]

DEFAULT_TEXT_HEADER_LINES += [f"C{line_no:02}" for line_no in range(5, 40)]
DEFAULT_TEXT_HEADER_LINES += ["C40 END TEXTUAL HEADER"]
DEFAULT_TEXT_HEADER_LINES = [line.ljust(80) for line in DEFAULT_TEXT_HEADER_LINES]
DEFAULT_TEXT_HEADER = "\n".join(DEFAULT_TEXT_HEADER_LINES)


class SegyFactory:
    """Factory class for composing SEG-Y by components.

    Args:
        spec: SEG-Y specification.
        sample_interval: Sample interval to use in new file.
        samples_per_trace: Number of samples per trace.
    """

    def __init__(
        self,
        spec: SegyDescriptor,
        sample_interval: int = 4000,
        samples_per_trace: int = 1500,
    ) -> None:
        self.spec = spec

        self.sample_interval = sample_interval
        self.samples_per_trace = samples_per_trace

        self.spec.trace.sample_descriptor.samples = samples_per_trace

    @property
    def trace_sample_format(self) -> ScalarType:
        """Trace sample format of the SEG-Y file."""
        return self.spec.trace.sample_descriptor.format

    @property
    def segy_revision(self) -> SegyStandard | None:
        """Revision of the SEG-Y file."""
        return self.spec.segy_standard

    def create_textual_header(self, text: str | None = None) -> bytes:
        """Create a textual header for the SEG-Y file.

        The length of the text should match the rows and columns in the spec's
        TextHeaderDescriptor. The newlines must also be in the text to separate
        the rows.

        Args:
            text: String containing text header rows. If left as None, a default
                textual header will be created.

        Returns:
            Bytes containing the encoded text header, ready to write.
        """
        text = DEFAULT_TEXT_HEADER if text is None else text

        text_descriptor = self.spec.text_file_header
        text = text_descriptor._unwrap(text)

        return text_descriptor._encode(text)

    def create_binary_header(self) -> bytes:
        """Create a binary header for the SEG-Y file.

        Returns:
            Bytes containing the encoded binary header, ready to write.
        """
        binary_descriptor = self.spec.binary_file_header
        bin_header = np.zeros(shape=1, dtype=binary_descriptor.dtype)

        rev0 = self.segy_revision == SegyStandard.REV0
        if self.segy_revision is not None and not rev0:
            bin_header["seg_y_revision"] = self.segy_revision.value * 256

        bin_header["sample_interval"] = self.sample_interval
        bin_header["sample_interval_orig"] = self.sample_interval
        bin_header["samples_per_trace"] = self.samples_per_trace
        bin_header["samples_per_trace_orig"] = self.samples_per_trace
        bin_header["data_sample_format"] = SEGY_FORMAT_MAP[self.trace_sample_format]

        return bin_header.tobytes()

    def create_trace_header_template(
        self,
        size: int = 1,
    ) -> NDArray[Any]:
        """Create a trace header template array that conforms to the SEG-Y spec.

        Args:
            size: Number of headers for the template.

        Returns:
            Array containing the trace header template.
        """
        descriptor = self.spec.trace.header_descriptor
        dtype = descriptor.dtype.newbyteorder(Endianness.NATIVE.symbol)

        header_template = np.zeros(shape=size, dtype=dtype)

        # 'names' assumed not None by data structure (type ignores).
        field_names = header_template.dtype.names
        if "sample_interval" in field_names:  # type: ignore[operator]
            header_template["sample_interval"] = self.sample_interval

        if "samples_per_trace" in field_names:  # type: ignore[operator]
            header_template["samples_per_trace"] = self.samples_per_trace

        return header_template

    def create_trace_sample_template(
        self,
        size: int = 1,
    ) -> NDArray[Any]:
        """Create a trace data template array that conforms to the SEG-Y spec.

        Args:
            size: Number of traces for the template.

        Returns:
            Array containing the trace data template.
        """
        descriptor = self.spec.trace.sample_descriptor
        dtype = descriptor.dtype

        if self.trace_sample_format == ScalarType.IBM32:
            dtype = np.dtype(("float32", (self.samples_per_trace,)))

        return np.zeros(shape=size, dtype=dtype)

    def create_traces(self, headers: NDArray[Any], samples: NDArray[Any]) -> bytes:
        """Convert trace data and header to bytes conforming to SEG-Y spec.

        The rows (length) of the headers and traces must match. The headers
        must be a (num_traces,) shape array and data must be a
        (num_traces, num_samples) shape array. They can be created via the
        `create_trace_header_template` and `create_trace_sample_template` methods.

        Args:
            headers: Header array.
            samples: Data array.

        Returns:
            Bytes containing the encoded traces, ready to write.

        Raises:
            AttributeError: if data dimensions are wrong (not 2D trace,samples).
            ValueError: if there is a shape mismatch between headers.
            ValueError: if there is a shape mismatch number of samples.
        """
        trace_descriptor = self.spec.trace
        trace_descriptor.sample_descriptor.samples = self.samples_per_trace

        if samples.ndim != 2:  # noqa: PLR2004
            msg = (
                "Data array must be 2-dimensional with rows as traces "
                "and columns as data samples."
            )
            raise AttributeError(msg)

        if samples.shape[1] != self.samples_per_trace:
            msg = f"Trace length must be {self.samples_per_trace}."
            raise ValueError(msg)

        if len(headers) != len(samples):
            msg = "Header array must have the same number of rows as data array."
            raise ValueError(msg)

        header_pipeline = TransformPipeline()
        data_pipeline = TransformPipeline()

        target_endian = trace_descriptor.endianness
        target_format = trace_descriptor.sample_descriptor.format

        if target_endian == Endianness.BIG:
            byte_swap = TransformFactory.create("byte_swap", target_endian)
            header_pipeline.add_transform(byte_swap)
            data_pipeline.add_transform(byte_swap)

        if target_format == ScalarType.IBM32:
            ibm_float = TransformFactory.create("ibm_float", "to_ibm")
            data_pipeline.add_transform(ibm_float)

        trace = np.zeros(shape=headers.size, dtype=trace_descriptor.dtype)
        trace["header"] = header_pipeline.apply(headers)
        trace["sample"] = data_pipeline.apply(samples)

        return trace.tobytes()
