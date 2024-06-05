"""Data model implementations for trace specification."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pydantic import Field
from pydantic import model_validator

from segy.schema.base import BaseDataType

if TYPE_CHECKING:
    from segy.schema.base import Endianness
    from segy.schema.format import ScalarType
    from segy.schema.header import HeaderSpec


class TraceDataSpec(BaseDataType):
    """A spec class for trace data (samples)."""

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
        dtype = (self.format.char, (self.samples,))
        return np.dtype(dtype)


class TraceSpec(BaseDataType):
    """A spec class for a trace (header + data)."""

    header_spec: HeaderSpec = Field(..., description="Trace header spec.")
    ext_header_spec: HeaderSpec | None = Field(
        default=None, description="Extended trace header spec."
    )
    data_spec: TraceDataSpec = Field(..., description="Trace data spec.")
    offset: int | None = Field(
        default=None, description="Starting offset of the trace."
    )
    endianness: Endianness | None = Field(
        default=None, description="Endianness of traces and headers."
    )

    @model_validator(mode="after")
    def update_submodel_endianness(self) -> TraceSpec:
        """Ensure that submodel endianness matches the trace endianness."""
        self.header_spec.endianness = self.endianness

        if self.ext_header_spec is not None:
            self.ext_header_spec.endianness = self.endianness

        return self

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        header_dtype = self.header_spec.dtype
        data_dtype = self.data_spec.dtype

        trace_dtype = np.dtype([("header", header_dtype), ("data", data_dtype)])

        if self.endianness is not None:
            trace_dtype = trace_dtype.newbyteorder(self.endianness.symbol)

        return trace_dtype
