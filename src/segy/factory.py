"""Factory methods for SEG-Y file creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from segy.schema import Endianness
from segy.transforms import TransformPipeline
from segy.transforms import TransformStrategyFactory

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from segy.schema import SegyDescriptor


DEFAULT_TEXT_HEADER = [
    "C01 File written by the open-source segy library.",
    "C02",
    "C03 Website: https://segy.readthedocs.io",
    "C04 Source: https://github.com/TGSAI/segy",
]

DEFAULT_TEXT_HEADER += [f"C{line_no:02}" for line_no in range(5, 40)]
DEFAULT_TEXT_HEADER += ["C40 END TEXTUAL HEADER"]
DEFAULT_TEXT_HEADER = [line.ljust(80) for line in DEFAULT_TEXT_HEADER]
DEFAULT_TEXT_HEADER = "\n".join(DEFAULT_TEXT_HEADER)


class SegyFactory:
    """Factory class for composing SEG-Y by components."""

    def __init__(
        self,
        spec: SegyDescriptor,
        sample_interval: int = 4000,
        samples_per_trace: int = 1500,
    ) -> None:
        self.spec = spec

        self.sample_interval = sample_interval
        self.samples_per_trace = samples_per_trace

        self.spec.trace.data_descriptor.samples = samples_per_trace

    def create_textual_header(self, text: str = DEFAULT_TEXT_HEADER) -> bytes:
        """Create a textual header for the SEG-Y file.

        This should be 3200 characters long.  If the provided text is
        shorter, it will be padded with spaces.
        """
        text_descriptor = self.spec.text_file_header
        text = text_descriptor._unwrap(text)

        return text_descriptor._encode(text)

    def create_binary_header(self) -> bytes:
        """Create a binary header for the SEG-Y file."""
        binary_descriptor = self.spec.binary_file_header
        bin_header = np.zeros(shape=1, dtype=binary_descriptor.dtype)

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
        """Create a trace header template array that conforms to the SEG-Y spec."""
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
        """Create a trace data template array that conforms to the SEG-Y spec."""
        descriptor = self.spec.trace.data_descriptor
        dtype = descriptor.dtype.newbyteorder(Endianness.NATIVE.symbol)

        data_template = np.empty(shape=size, dtype=dtype)

        if fill is True:
            data_template.fill(0)

        return data_template

    def create_traces(self, headers: NDArray, data: NDArray[Any]) -> bytes:
        """Create traces based on SEG-Y spec."""
        trace_descriptor = self.spec.trace
        trace_descriptor.data_descriptor.samples = self.samples_per_trace

        if data.ndim != 2:  # noqa: PLR2004
            msg = (
                "Data array must be 2-dimensional with rows as traces "
                "and columns as data samples."
            )
            raise AttributeError(msg)

        if data.shape[1] != self.samples_per_trace:
            msg = f"Trace length must be {self.samples_per_trace}."
            raise ValueError(msg)

        if len(headers) != len(data):
            msg = "Header array must have the same number of rows as data array."
            raise ValueError(msg)

        target_endian = trace_descriptor.data_descriptor.endianness

        header_pipeline = TransformPipeline()
        data_pipeline = TransformPipeline()

        convert_to_ibm = TransformStrategyFactory.create_strategy(
            transform_type="ibm_float",
            parameters={"direction": "to_ibm"},
        )

        byte_swap = TransformStrategyFactory.create_strategy(
            transform_type="byte_swap",
            parameters={
                "source_order": Endianness.NATIVE,
                "target_order": target_endian,
            },
        )

        header_pipeline.add_transformation(byte_swap)

        data_pipeline.add_transformation(convert_to_ibm)
        data_pipeline.add_transformation(byte_swap)

        trace = np.empty(shape=len(data), dtype=trace_descriptor.dtype)
        trace["header"] = header_pipeline.transform(headers)
        trace["data"] = data_pipeline.transform(data)

        return trace.tobytes()
