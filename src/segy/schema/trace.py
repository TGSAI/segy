"""Descriptor data model implementations for traces."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pydantic import Field
from pydantic import model_validator

from segy.schema.base import BaseTypeDescriptor

if TYPE_CHECKING:
    from segy.schema.data_type import Endianness
    from segy.schema.data_type import ScalarType
    from segy.schema.data_type import StructuredDataTypeDescriptor


class TraceSampleDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace Samples."""

    format: ScalarType = Field(..., description="Format of trace samples.")  # noqa: A003
    samples: int | None = Field(
        default=None,
        description=(
            "Number of samples in trace. It can be variable, "
            "then it must be read from each trace header."
        ),
    )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        format_char = self.format.char
        dtype_str = f"{self.samples}{format_char}"
        return np.dtype(dtype_str)


class TraceDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace (Header + Data)."""

    header_descriptor: StructuredDataTypeDescriptor = Field(
        ..., description="Trace header descriptor."
    )
    extended_header_descriptor: StructuredDataTypeDescriptor | None = Field(
        default=None, description="Extended trace header descriptor."
    )
    sample_descriptor: TraceSampleDescriptor = Field(
        ..., description="Trace data descriptor."
    )
    offset: int | None = Field(
        default=None, description="Starting offset of the trace."
    )
    endianness: Endianness | None = Field(
        default=None, description="Endianness of traces and headers."
    )

    @model_validator(mode="after")
    def update_submodel_endianness(self) -> TraceDescriptor:
        """Ensure that submodel endianness matches the trace endianness."""
        self.header_descriptor.endianness = self.endianness

        if self.extended_header_descriptor is not None:
            self.extended_header_descriptor.endianness = self.endianness

        return self

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        header_dtype = self.header_descriptor.dtype
        data_dtype = self.sample_descriptor.dtype

        trace_dtype = np.dtype([("header", header_dtype), ("sample", data_dtype)])

        if self.endianness is not None:
            trace_dtype = trace_dtype.newbyteorder(self.endianness.symbol)

        return trace_dtype
