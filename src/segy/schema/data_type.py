"""Data models representing data types."""
from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import Field

from segy.schema.base import BaseTypeDescriptor


class Endianness(StrEnum):
    """Enum class representing endianness."""

    BIG = "big"
    LITTLE = "little"

    @property
    def symbol(self) -> str:
        """Get the numpy symbol for the endianness."""
        return ">" if self == Endianness.BIG else "<"


class ScalarType(StrEnum):
    """A class representing different data types used for data formatting."""

    IBM32 = "ibm32"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT64 = "uint64"
    UINT32 = "uint32"
    UINT16 = "uint16"
    UINT8 = "uint8"
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"

    @property
    def char(self) -> str:
        """Returns the numpy character code for a given data type string."""
        if self.value == "ibm32":
            return np.sctype2char("uint32")

        return np.sctype2char(str(self.value))


class DataTypeDescriptor(BaseTypeDescriptor):
    """A descriptor class for a scalar data type with endianness support."""

    format: ScalarType = Field(..., description="The data type of the field.")  # noqa: A003
    endianness: Endianness = Field(
        default=Endianness.BIG, description="The byte order of the field."
    )

    @property
    def dtype(self) -> np.dtype:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        symbol = self.endianness.symbol
        char = self.format.char

        return np.dtype(symbol + char)


class StructuredFieldDescriptor(DataTypeDescriptor):
    """A descriptor class for a structured data-type field."""

    name: str = Field(..., description="The short name of the field.")
    offset: int = Field(..., ge=0, description="Starting byte offset.")


class StructuredDataTypeDescriptor(BaseTypeDescriptor):
    """A descriptor class for a structured array data-type."""

    fields: list[StructuredFieldDescriptor] = Field(
        ..., description="Fields of the structured data type."
    )
    item_size: int | None = Field(
        default=None, description="Expected size of the struct."
    )
    offset: int | None = Field(default=None, ge=0, description="Starting byte offset.")

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype."""
        names = [field.name for field in self.fields]
        offsets = [field.offset for field in self.fields]
        formats = [field.dtype for field in self.fields]

        dtype_conf = {
            "names": names,
            "formats": formats,
            "offsets": offsets,
        }

        if self.item_size is not None:
            dtype_conf["itemsize"] = self.item_size

        return np.dtype(dtype_conf)
