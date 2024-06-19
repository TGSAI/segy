"""Factory methods for SEG-Y file creation."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from segy.arrays import HeaderArray
from segy.arrays import TraceArray
from segy.schema.base import Endianness
from segy.schema.format import ScalarType
from segy.schema.segy import SegyStandard
from segy.standards.mapping import SEGY_FORMAT_MAP
from segy.transforms import TransformFactory
from segy.transforms import TransformPipeline

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    from segy.schema import SegySpec


DEFAULT_TEXT_HEADER_LINES = [
    "C01 File written by the open-source segy library.",
    "C02",
    "C03 Website: https://segy.readthedocs.io",
    "C04 Source: https://github.com/TGSAI/segy",
    "C05",
    "C06 File written: {timestamp}",
    "C07",
    "C08 SEG-Y Revision: {revision}",
    "C09",
    "C10 Sample Interval: {sample_interval} ms",
    "C11 Samples Per Trace: {samples_per_trace}",
]

DEFAULT_TEXT_HEADER_LINES += [f"C{line_no:02}" for line_no in range(12, 40)]
DEFAULT_TEXT_HEADER_LINES += ["C40 END TEXTUAL HEADER"]


def get_default_text(spec: SegySpec) -> str:
    """Dynamically generate default text header based on spec."""
    text_lines = DEFAULT_TEXT_HEADER_LINES.copy()

    # Populate write time
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    text_lines[5] = text_lines[5].format(timestamp=now)

    # Populate revision
    revision = "unknown" if spec.segy_standard is None else spec.segy_standard.value
    text_lines[7] = text_lines[7].format(revision=revision)

    # Populate sample interval and number of samples
    sample_interval = spec.trace.data.interval // 1000  # type: ignore[operator]
    text_lines[9] = text_lines[9].format(sample_interval=sample_interval)
    text_lines[10] = text_lines[10].format(samples_per_trace=spec.trace.data.samples)

    text_lines = [line.ljust(80) for line in text_lines]
    return "\n".join(text_lines)


class SegyFactory:
    """Factory class for composing SEG-Y by components.

    Args:
        spec: SEG-Y specification.
        sample_interval: Sample interval to use in new file.
        samples_per_trace: Number of samples per trace.
    """

    def __init__(
        self,
        spec: SegySpec,
        sample_interval: int = 4000,
        samples_per_trace: int = 1500,
    ) -> None:
        self.spec = spec

        self.spec.trace.data.interval = sample_interval
        self.spec.trace.data.samples = samples_per_trace

    @property
    def sample_format(self) -> ScalarType:
        """Trace sample format of the SEG-Y file."""
        return self.spec.trace.data.format

    @property
    def sample_interval(self) -> int:
        """Return sample interval from spec."""
        # We know its populated at this point and its int (not None)
        return cast(int, self.spec.trace.data.interval)

    @property
    def samples_per_trace(self) -> int:
        """Return number of samples from spec."""
        # We know its populated at this point and its int (not None)
        return cast(int, self.spec.trace.data.samples)

    @property
    def segy_revision(self) -> SegyStandard | None:
        """Revision of the SEG-Y file."""
        return self.spec.segy_standard

    def create_textual_header(self, text: str | None = None) -> bytes:
        """Create a textual header for the SEG-Y file.

        The length of the text should match the rows and columns in the spec's
        TextHeaderSpec. The newlines must also be in the text to separate
        the rows.

        Args:
            text: String containing text header rows. If left as None, a default
                textual header will be created.

        Returns:
            Bytes containing the encoded text header, ready to write.
        """
        if text is None:
            text = get_default_text(self.spec)

        text_spec = self.spec.text_header

        return text_spec.encode(text)

    def create_binary_header(self, update: dict[str, Any] | None = None) -> bytes:
        """Create a binary header for the SEG-Y file.

        This function will create bytes representing the binary header with the
        configuration options provided to SEG-Y factory.

        The `update` parameter is a dictionary that contains binary header fields which
        need to be modified. The function will update these fields with the values
        specified in the update dictionary before returning the encoded bytes.

        Args:
            update: Dictionary containing binary header fields to modify.

        Returns:
            Bytes containing the encoded binary header, ready to write.
        """
        binary_spec = self.spec.binary_header
        bin_header = HeaderArray(np.zeros(shape=1, dtype=binary_spec.dtype))

        rev0 = self.segy_revision == SegyStandard.REV0
        if self.segy_revision is not None and not rev0:
            bin_header["segy_revision"] = self.segy_revision.value * 256

        bin_header["sample_interval"] = self.sample_interval
        bin_header["orig_sample_interval"] = self.sample_interval
        bin_header["samples_per_trace"] = self.samples_per_trace
        bin_header["orig_samples_per_trace"] = self.samples_per_trace
        bin_header["data_sample_format"] = SEGY_FORMAT_MAP[self.sample_format]

        if update is not None:
            for key, value in update.items():
                bin_header[key] = value

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
        trace_header_spec = self.spec.trace.header
        dtype = trace_header_spec.dtype.newbyteorder(Endianness.NATIVE.symbol)

        header_template = HeaderArray(np.zeros(shape=size, dtype=dtype))

        # 'names' assumed not None by data structure (type ignores).
        field_names = header_template.dtype.names
        if "sample_interval" in field_names:
            header_template["sample_interval"] = self.sample_interval

        if "samples_per_trace" in field_names:
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
        trace_data_spec = self.spec.trace.data
        dtype = trace_data_spec.dtype

        if self.sample_format == ScalarType.IBM32:
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
        trace_spec = self.spec.trace
        trace_spec.data.samples = self.samples_per_trace

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

        target_endian = trace_spec.endianness
        target_format = trace_spec.data.format

        if target_endian == Endianness.BIG:
            byte_swap = TransformFactory.create("byte_swap", target_endian)
            header_pipeline.add_transform(byte_swap)
            data_pipeline.add_transform(byte_swap)

        if target_format == ScalarType.IBM32:
            ibm_float = TransformFactory.create("ibm_float", "to_ibm")
            data_pipeline.add_transform(ibm_float)

        trace = TraceArray(np.zeros(shape=headers.size, dtype=trace_spec.dtype))
        trace["header"] = header_pipeline.apply(headers)
        trace["data"] = data_pipeline.apply(samples)

        return trace.tobytes()
