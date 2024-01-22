"""Descriptor data model implementations for SEG-Y file(s)."""
from __future__ import annotations

from enum import Enum

from pydantic import Field
from pydantic import create_model

from segy.schema.base import CamelCaseModel
from segy.schema.header import BinaryHeaderDescriptor
from segy.schema.header import HeaderFieldDescriptor
from segy.schema.header import TextHeaderDescriptor
from segy.schema.trace import TraceDataDescriptor
from segy.schema.trace import TraceDescriptor


class SegyStandard(Enum):
    """Allowed values for SegyStandard in SegyDescriptor."""

    REV0 = 0
    REV1 = 1
    REV2 = 2
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

    @classmethod
    def customize(  # noqa: PLR0913
        cls: type[SegyDescriptor],
        text_header_spec: TextHeaderDescriptor = None,
        binary_header_fields: list[HeaderFieldDescriptor] = None,
        extended_text_spec: TextHeaderDescriptor = None,
        trace_header_fields: list[HeaderFieldDescriptor] = None,
        trace_data_spec: TraceDataDescriptor = None,
    ) -> type[SegyDescriptor]:
        """Customize an existing SEG-Y descriptor."""
        old_fields = cls.model_fields

        new_fields = {"segy_standard": (SegyStandard, SegyStandard.CUSTOM)}

        if text_header_spec:
            new_fields["text_file_header"] = (TextHeaderDescriptor, text_header_spec)

        if binary_header_fields:
            # Update binary header fields if specified; else will revert to default.
            bin_hdr_spec = old_fields["binary_file_header"].default.model_copy()
            bin_hdr_spec.fields = binary_header_fields
            new_fields["binary_file_header"] = (BinaryHeaderDescriptor, bin_hdr_spec)

        if extended_text_spec:
            # Update extended text spec if its specified; else will revert to default.
            new_fields["extended_text_header"] = (
                TextHeaderDescriptor | None,
                extended_text_spec,
            )

        # Handling trace spec.
        trc_spec = old_fields["trace"].default.model_copy()

        if trace_header_fields:
            # Update trace header spec if its specified; else will revert to default.
            trc_spec.header_descriptor.fields = trace_header_fields
            new_fields["trace"] = (TraceDescriptor, trc_spec)

        if trace_data_spec:
            # Update trace data spec if its specified; else will revert to default.
            trc_spec.data_descriptor = trace_data_spec
            new_fields["trace"] = (TraceDescriptor, trc_spec)

        return create_model(
            "CustomSegySpec",
            **new_fields,
            __doc__=f"Custom SEG-Y descriptor derived from {cls.__name__}",
            __base__=cls,
        )
