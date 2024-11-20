"""Data format specification representing scalar and trace data types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from segy.compat import StrEnum

if TYPE_CHECKING:
    from typing import Any


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
    STRING8 = "S8"

    def __repr__(self) -> str:
        """Nicer representation for users."""
        return f"{self.__class__.__name__}.{self._name_}"

    @property
    def dtype(self) -> np.dtype[Any]:
        """Return numpy dtype of the format."""
        # Special case for IBM 32-bit float
        if self.value == "ibm32":
            return np.dtype("uint32")

        return np.dtype(self.value)


class TextHeaderEncoding(StrEnum):
    """Supported textual header encodings."""

    ASCII = "ascii"
    EBCDIC = "ebcdic"

    @property
    def dtype(self) -> ScalarType:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        return ScalarType.UINT8
