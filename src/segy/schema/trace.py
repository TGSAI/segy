"""Descriptor data model implementations for traces."""


from typing import Optional

import numpy as np
from pydantic import Field

from segy.schema.base import BaseTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.header import TraceHeaderDescriptor


class TraceDataDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace Data (samples)."""

    format: ScalarType = Field(...)  # noqa: A003
    endianness: Endianness = Field(default=Endianness.BIG)
    samples: Optional[int] = Field(
        default=None,
        description=(
            "Number of samples in trace. It can be variable, "
            "then it must be read from each trace header."
        ),
    )

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype."""
        format_char = self.format.char
        dtype_str = "".join([self.endianness.symbol, str(self.samples), format_char])
        return np.dtype(dtype_str)


class TraceDescriptor(BaseTypeDescriptor):
    """A descriptor class for a Trace (Header + Data)."""

    header_descriptor: TraceHeaderDescriptor = Field(...)
    extended_header_descriptor: Optional[TraceHeaderDescriptor] = Field(
        default=None, description="Extended trace header descriptor."
    )
    data_descriptor: TraceDataDescriptor = Field(...)
    offset: Optional[int] = Field(default=None)

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype."""
        header_dtype = self.header_descriptor.dtype
        data_dtype = self.data_descriptor.dtype

        return np.dtype([("header", header_dtype), ("data", data_dtype)])
