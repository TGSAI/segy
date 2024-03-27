"""Factory methods for SEG-Y file creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from segy.schema import Endianness
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

        bin_header["seg_y_revision"] = self.spec.segy_standard.value * 256
        bin_header["sample_interval"] = self.sample_interval
        bin_header["sample_interval_orig"] = self.sample_interval
        bin_header["samples_per_trace"] = self.samples_per_trace
        bin_header["samples_per_trace_orig"] = self.samples_per_trace

        return bin_header.tobytes()

    def create_trace_header_template(
        self,
        size: int = 1,
        fill: bool = True,
    ) -> NDArray[Any]:
        """Create a trace header template array that conforms to the SEG-Y spec.

        Args:
            size: Number of headers for the template.
            fill: Optional, fill with zeros. Default is True.

        Returns:
            Array containing the trace header template.
        """
        descriptor = self.spec.trace.header_descriptor
        dtype = descriptor.dtype.newbyteorder(Endianness.NATIVE.symbol)

        header_template = np.empty(shape=size, dtype=dtype)

        if fill is True:
            header_template.fill(0)

        header_template["sample_interval"] = self.sample_interval
        header_template["samples_per_trace"] = self.samples_per_trace

        return header_template

    def create_trace_data_template(
        self,
        size: int = 1,
        fill: bool = True,
    ) -> NDArray[Any]:
        """Create a trace data template array that conforms to the SEG-Y spec.

        Args:
            size: Number of traces for the template.
            fill: Optional, fill with zeros. Default is True.

        Returns:
            Array containing the trace data template.
        """
        descriptor = self.spec.trace.sample_descriptor
        dtype = descriptor.dtype.newbyteorder(Endianness.NATIVE.symbol)

        data_template = np.empty(shape=size, dtype=dtype)

        if fill is True:
            data_template.fill(0)

        return data_template

    def create_traces(self, headers: NDArray[Any], samples: NDArray[Any]) -> bytes:
        """Convert trace data and header to bytes conforming to SEG-Y spec.

        The rows (length) of the headers and traces must match. The headers
        must be a (num_traces,) shape array and data must be a
        (num_traces, num_samples) shape array. They can be created via the
        `create_trace_header_template` and `create_trace_data_template` methods.

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

        target_endian = trace_descriptor.endianness

        header_pipeline = TransformPipeline()
        data_pipeline = TransformPipeline()

        byte_swap = TransformFactory.create("byte_swap", target_endian)
        ibm_float = TransformFactory.create("ibm_float", "to_ibm")

        header_pipeline.add_transform(byte_swap)
        data_pipeline.add_transform(ibm_float)
        data_pipeline.add_transform(byte_swap)

        trace = np.empty(shape=len(samples), dtype=trace_descriptor.dtype)
        trace["header"] = header_pipeline.apply(headers)
        trace["sample"] = data_pipeline.apply(samples)

        return trace.tobytes()
