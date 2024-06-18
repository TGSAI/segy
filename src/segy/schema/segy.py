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
        if binary_header_fields:
            new_spec.binary_header.fields = binary_header_fields

        # Update extended text spec if its specified; else will revert to default.
        if ext_text_spec:
            new_spec.ext_text_header = ext_text_spec

        # Update trace header spec if its specified; else will revert to default.
        if trace_header_fields:
            new_spec.trace.header.fields = trace_header_fields

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
