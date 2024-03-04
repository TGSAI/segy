"""Descriptor data model implementations for traces."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pydantic import Field

from segy.schema.base import BaseTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import StructuredDataTypeDescriptor

if TYPE_CHECKING:
    from segy.schema.data_type import ScalarType


class TraceDataDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace Data (samples)."""

    format: ScalarType = Field(..., description="Format of trace samples.")  # noqa: A003
    endianness: Endianness = Field(
        default=Endianness.BIG, description="Endianness of trace samples."
    )
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
        dtype_str = "".join([self.endianness.symbol, str(self.samples), format_char])
        return np.dtype(dtype_str)


class TraceDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace (Header + Data)."""

    header_descriptor: StructuredDataTypeDescriptor = Field(
        ..., description="Trace header descriptor."
    )
    extended_header_descriptor: StructuredDataTypeDescriptor | None = Field(
        default=None, description="Extended trace header descriptor."
    )
    data_descriptor: TraceDataDescriptor = Field(
        ..., description="Trace data descriptor."
    )
    offset: int | None = Field(
        default=None, description="Starting offset of the trace."
    )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        header_dtype = self.header_descriptor.dtype
        data_dtype = self.data_descriptor.dtype

        return np.dtype([("header", header_dtype), ("data", data_dtype)])
