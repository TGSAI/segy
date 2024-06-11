"""Data format specification representing scalar and trace data types."""

from __future__ import annotations

import numpy as np

from segy.compat import StrEnum


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
    def char(self) -> str:
        """Returns the numpy character code for a given data type string."""
        if self.value == "ibm32":
            return np.sctype2char("uint32")  # noqa: NPY201

        return np.sctype2char(str(self.value))  # noqa: NPY201


class TextHeaderEncoding(StrEnum):
    """Supported textual header encodings."""

    ASCII = "ascii"
    EBCDIC = "ebcdic"

    @property
    def dtype(self) -> ScalarType:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        return ScalarType.UINT8
