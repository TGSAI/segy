"""Descriptor data model implementations for SEG-Y file(s)."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from segy.schema.base import CamelCaseModel

if TYPE_CHECKING:
    from segy.schema.data_type import Endianness
    from segy.schema.data_type import StructuredDataTypeDescriptor
    from segy.schema.data_type import StructuredFieldDescriptor
    from segy.schema.header import TextHeaderDescriptor
    from segy.schema.trace import TraceDescriptor
    from segy.schema.trace import TraceSampleDescriptor


class SegyStandard(Enum):
    """Allowed values for SEG-Y standards in SegyDescriptor."""

    REV0 = 0.0
    REV1 = 1.0
    REV2 = 2.0
    REV21 = 2.1
    CUSTOM = "custom"


class SegyDescriptor(CamelCaseModel):
    """A descriptor class for a SEG-Y file."""

    segy_standard: SegyStandard = Field(
        ..., description="SEG-Y Revision / Standard. Can also be custom."
    )
    text_file_header: TextHeaderDescriptor = Field(
        ..., description="Textual file header descriptor."
    )
    binary_file_header: StructuredDataTypeDescriptor = Field(
        ..., description="Binary file header descriptor."
    )
    extended_text_header: TextHeaderDescriptor | None = Field(
        default=None, description="Extended textual header descriptor."
    )
    trace: TraceDescriptor = Field(..., description="Trace header + data descriptor.")

    endianness: Endianness | None = Field(
        default=None, description="Endianness of SEG-Y file."
    )

    @model_validator(mode="after")
    def update_submodel_endianness(self) -> SegyDescriptor:
        """Ensure that submodel endianness matches the SEG-Y endianness."""
        self.binary_file_header.endianness = self.endianness
        self.trace.endianness = self.endianness

        return self

    def customize(  # noqa: PLR0913
        self: SegyDescriptor,
        text_header_spec: TextHeaderDescriptor | None = None,
        binary_header_fields: list[StructuredFieldDescriptor] | None = None,
        extended_text_spec: TextHeaderDescriptor | None = None,
        trace_header_fields: list[StructuredFieldDescriptor] | None = None,
        trace_data_spec: TraceSampleDescriptor | None = None,
    ) -> SegyDescriptor:
        """Customize an existing SEG-Y descriptor.

        Args:
            text_header_spec: New text header specification.
            binary_header_fields: List of custom binary header fields.
            extended_text_spec: New extended text header specification.
            trace_header_fields: List of custom trace header fields.
            trace_data_spec: New trace data specification.

        Returns:
            A modified SEG-Y descriptor with "custom" segy standard.
        """
        new_descr = self.model_copy(deep=True)
        new_descr.segy_standard = SegyStandard.CUSTOM

        if text_header_spec:
            new_descr.text_file_header = text_header_spec

        # Update binary header fields if specified; else will revert to default.
        if binary_header_fields:
            new_descr.binary_file_header.fields = binary_header_fields

        # Update extended text spec if its specified; else will revert to default.
        if extended_text_spec:
            new_descr.extended_text_header = extended_text_spec

        # Update trace header spec if its specified; else will revert to default.
        if trace_header_fields:
            new_descr.trace.header_descriptor.fields = trace_header_fields

        # Update trace data spec if its specified; else will revert to default.
        if trace_data_spec:
            new_descr.trace.sample_descriptor = trace_data_spec

        return new_descr


class SegyInfo(CamelCaseModel):
    """Concise and useful information about SEG-Y files."""

    uri: str = Field(..., description="URI of the SEG-Y file.")

    segy_standard: SegyStandard = Field(
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
