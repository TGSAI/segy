"""Descriptor data model implementations for SEG-Y file(s)."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import Field

from segy.schema.base import CamelCaseModel

if TYPE_CHECKING:
    from segy.schema.header import BinaryHeaderDescriptor
    from segy.schema.header import HeaderFieldDescriptor
    from segy.schema.header import TextHeaderDescriptor
    from segy.schema.trace import TraceDataDescriptor
    from segy.schema.trace import TraceDescriptor


class SegyStandard(Enum):
    """Allowed values for SegyStandard in SegyDescriptor."""

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
    binary_file_header: BinaryHeaderDescriptor = Field(
        ..., description="Binary file header descriptor."
    )
    extended_text_header: TextHeaderDescriptor | None = Field(
        default=None, description="Extended textual header descriptor."
    )
    trace: TraceDescriptor = Field(..., description="Trace header + data descriptor.")

    def customize(  # noqa: PLR0913
        self: SegyDescriptor,
        text_header_spec: TextHeaderDescriptor = None,
        binary_header_fields: list[HeaderFieldDescriptor] = None,
        extended_text_spec: TextHeaderDescriptor = None,
        trace_header_fields: list[HeaderFieldDescriptor] = None,
        trace_data_spec: TraceDataDescriptor = None,
    ) -> SegyDescriptor:
        """Customize an existing SEG-Y descriptor."""
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
            new_descr.trace.data_descriptor = trace_data_spec

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

    sample_interval: int = Field(..., description="Sampling rate from binary header.")

    file_size: float = Field(..., description="File size in GB.")
