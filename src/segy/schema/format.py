"""Data format specification representing scalar and trace data types."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from segy.compat import StrEnum
from segy.schema.base import BaseDataType


class ScalarType(StrEnum):
    """A class representing scalar data types."""

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
            return np.sctype2char("uint32")  # noqa: NPY201

        return np.sctype2char(str(self.value))  # noqa: NPY201


class DataFormat(BaseDataType):
    """A class representing a data format spec.

    Examples:
        A float32:

        >>> data_format = DataFormat(format="float32")
        >>> data_format.dtype
        dtype('float32')

        A 16-bit unsigned integer:

        >>> data_format = DataFormat(format="uint16")
        >>> data_format.dtype
        dtype('uint16')
    """

    format: ScalarType = Field(..., description="The data type of the field.")  # noqa: A003

    @property
    def dtype(self) -> np.dtype[Any]:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        return np.dtype(self.format.char)
