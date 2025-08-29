"""Data model implementations for SEG-Y file spec."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from segy.schema.base import CamelCaseModel

if TYPE_CHECKING:
    from segy.schema.base import Endianness
    from segy.schema.header import HeaderField
    from segy.schema.header import HeaderSpec
    from segy.schema.text_header import ExtendedTextHeaderSpec
    from segy.schema.text_header import TextHeaderSpec
    from segy.schema.trace import TraceDataSpec
    from segy.schema.trace import TraceSpec


class SegyStandard(float, Enum):
    """Allowed values for SEG-Y standards in SegySpec."""

    REV0 = 0.0
    REV1 = 1.0
    REV2 = 2.0
    REV21 = 2.1


class SegySpec(CamelCaseModel):
    """A class defining a SEG-Y file spec."""

    segy_standard: SegyStandard | None = Field(
        ..., description="SEG-Y Revision / Standard. Can also be custom."
    )
    text_header: TextHeaderSpec = Field(..., description="Textual file header spec.")
    binary_header: HeaderSpec = Field(..., description="Binary file header spec.")
    ext_text_header: ExtendedTextHeaderSpec | None = Field(
        default=None, description="Extended textual header spec."
    )
    trace: TraceSpec = Field(..., description="Trace header + data spec.")

    endianness: Endianness | None = Field(
        default=None, description="Endianness of SEG-Y file."
    )

    @model_validator(mode="after")
    def update_submodel_endianness(self) -> SegySpec:
        """Ensure that submodel endianness matches the SEG-Y endianness."""
        self.binary_header.endianness = self.endianness
        self.trace.endianness = self.endianness

        return self

    def update_offsets(self) -> None:
        """Update the offsets of the SEG-Y components."""
        cursor = 0

        if self.text_header.offset is None:
            self.text_header.offset = 0
        cursor += self.text_header.itemsize

        if self.binary_header.offset is None:
            self.binary_header.offset = cursor
        cursor += self.binary_header.itemsize

        if self.ext_text_header is not None:
            self.ext_text_header.offset = cursor
            cursor += self.ext_text_header.itemsize

        if self.trace.offset is None:
            self.trace.offset = cursor

    def _overlap(self, range1: tuple[int, int], range2: tuple[int, int]) -> bool:
        """Checks if two right half-open ranges overlap."""
        return range1[0] < range2[1] and range1[1] > range2[0]

    def _merge_headers_by_name(
        self, existing_fields: list[HeaderField], new_fields: list[HeaderField]
    ) -> list[HeaderField]:
        """Replaces existing headers with new headers that have the same name.

        Args:
            existing_fields: List of existing header fields.
            new_fields: List of new header fields.

        Returns:
            List of header fields with duplicates removed.
        """
        if not existing_fields:
            return new_fields
        if not new_fields:
            return existing_fields

        new_names = {field.name for field in new_fields}

        filtered_fields = list(new_fields)

        for field in existing_fields:
            if field.name not in new_names:
                filtered_fields.append(field)  # noqa: PERF401  Don't use extend here.
        return filtered_fields

    def _merge_headers_by_byte_offset(
        self, existing_fields: list[HeaderField], new_fields: list[HeaderField]
    ) -> list[HeaderField]:
        """Removes existing headers that have bytes that would have been overlapped by new headers.

        Intended to be run AFTER _merge_headers_by_name.
        This algorithm will ensure all neighboring headers are not overlapped.

        An overlap is defined as a

        Args:
            existing_fields: List of existing header fields. State AFTER _merge_headers_by_name.
            new_fields: List of new header fields.

        Returns:
            List of header fields with duplicates removed.
        """
        ranges = [(field.name, field.range) for field in existing_fields]
        ranges.sort(key=lambda range_tuple: range_tuple[1][0])
        indices_to_remove = []
        for i in range(len(ranges) - 1):
            current_key, current_range = ranges[i]
            next_key, next_range = ranges[i + 1]
            if self._overlap(current_range, next_range):
                for field in new_fields:
                    if field.name == current_key:
                        indices_to_remove.append(
                            i + 1
                        )  # Remove next header, not current
                        break
        # Sort indices in reverse order to avoid index shifting when removing elements
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            if 0 <= idx < len(existing_fields):
                header_name, header_range = ranges[idx]
                # Find and remove the header by name
                for i, elem in enumerate(existing_fields):
                    if elem.name == header_name:
                        existing_fields.pop(i)
                        break
        return existing_fields

    def _validate_non_overlapping_headers(self, new_fields: list[HeaderField]) -> None:
        """Validates that a list of new headers have unique names and do not overlap one-another.

        Args:
            new_fields: List of new header fields.

        Raises:
            ValueError: If duplicate header field names are detected.
            ValueError: If header fields overlap.
        """
        if not new_fields:
            return

        names = [field.name for field in new_fields]
        if len(names) != len(set(names)):
            msg = f"Duplicate header field names detected: {names}!"
            raise ValueError(msg)

        ranges = [field.range for field in new_fields]
        ranges.sort(key=lambda range_tuple: range_tuple[0])

        for i in range(len(ranges) - 1):
            if self._overlap(ranges[i], ranges[i + 1]):
                msg = f"Header fields overlap: {ranges[i]} and {ranges[i + 1]}!"
                raise ValueError(msg)

    def customize(  # noqa: PLR0913
        self: SegySpec,
        text_header_spec: TextHeaderSpec | None = None,
        binary_header_fields: list[HeaderField] | None = None,
        ext_text_spec: ExtendedTextHeaderSpec | None = None,
        trace_header_fields: list[HeaderField] | None = None,
        trace_data_spec: TraceDataSpec | None = None,
    ) -> SegySpec:
        """Customize an existing SEG-Y spec.

        Args:
            text_header_spec: New text header specification.
            binary_header_fields: List of custom binary header fields.
            ext_text_spec: New extended text header spec.
            trace_header_fields: List of custom trace header fields.
            trace_data_spec: New trace data specification.

        Returns:
            A modified SEG-Y spec with "custom" segy standard.
        """
        new_spec = self.model_copy(deep=True)
        new_spec.segy_standard = None

        if text_header_spec:
            new_spec.text_header = text_header_spec

        # Update binary header fields if specified; else will revert to default.
        self._validate_non_overlapping_headers(binary_header_fields)
        new_spec.binary_header.fields = self._merge_headers_by_name(
            new_spec.binary_header.fields, binary_header_fields
        )
        new_spec.binary_header.fields = self._merge_headers_by_byte_offset(
            new_spec.binary_header.fields, binary_header_fields
        )

        # Update extended text spec if its specified; else will revert to default.
        if ext_text_spec:
            new_spec.ext_text_header = ext_text_spec

        # Update trace header spec if its specified; else will revert to default.
        self._validate_non_overlapping_headers(trace_header_fields)
        new_spec.trace.header.fields = self._merge_headers_by_name(
            new_spec.trace.header.fields, trace_header_fields
        )
        new_spec.trace.header.fields = self._merge_headers_by_byte_offset(
            new_spec.trace.header.fields, trace_header_fields
        )

        # Update trace data spec if its specified; else will revert to default.
        if trace_data_spec:
            new_spec.trace.data = trace_data_spec

        return new_spec


class SegyInfo(CamelCaseModel):
    """Concise and useful information about SEG-Y files."""

    uri: str = Field(..., description="URI of the SEG-Y file.")

    segy_standard: SegyStandard | None = Field(
        ..., description="SEG-Y Revision / Standard. Can also be custom."
    )

    num_traces: int = Field(..., description="Number of traces.")

    samples_per_trace: int = Field(
        ..., description="Trace length in number of samples."
    )

    sample_interval: int | float = Field(
        ..., description="Sampling rate from binary header."
    )

    file_size: int = Field(..., description="File size in bytes.")
