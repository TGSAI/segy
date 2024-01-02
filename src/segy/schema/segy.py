"""Descriptor data model implementations for SEG-Y file(s)."""


from enum import Enum
from typing import Optional

from pydantic import Field

from segy.schema.base import CamelCaseModel
from segy.schema.header import BinaryHeaderDescriptor
from segy.schema.header import TextHeaderDescriptor
from segy.schema.trace import TraceDescriptor


class SegyStandard(Enum):
    """Supported (to-be) SEG-Y standards."""

    REV0 = 0
    REV1 = 1
    REV2 = 2
    REV21 = 2.1
    CUSTOM = "custom"


class SegyDescriptor(CamelCaseModel):
    """A descriptor class for a SEG-Y file."""

    segy_standard: Optional[SegyStandard] = Field(
        default=None, description="SEG-Y Revision"
    )
    text_file_header: TextHeaderDescriptor = Field(...)
    binary_file_header: BinaryHeaderDescriptor = Field(...)
    extended_text_headers: Optional[list[TextHeaderDescriptor]] = Field(default=None)
    traces: TraceDescriptor = Field(...)
