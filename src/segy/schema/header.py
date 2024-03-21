"""Classes for managing headers and header groups."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from pydantic import Field

from segy.compat import StrEnum
from segy.ebcdic import ASCII_TO_EBCDIC
from segy.ebcdic import EBCDIC_TO_ASCII
from segy.schema.base import BaseTypeDescriptor

if TYPE_CHECKING:
    from segy.schema.data_type import ScalarType


class TextHeaderEncoding(StrEnum):
    """Supported textual header encodings."""

    ASCII = "ascii"
    EBCDIC = "ebcdic"


class TextHeaderDescriptor(BaseTypeDescriptor):
    """A descriptor class for SEG-Y textual headers."""

    rows: int = Field(..., description="Number of rows in text header.")
    cols: int = Field(..., description="Number of columns in text header.")
    encoding: TextHeaderEncoding = Field(..., description="String encoding.")
    format: ScalarType = Field(..., description="Type of string.")  # noqa: A003
    offset: int | None = Field(default=None, ge=0, description="Starting byte offset.")

    def __len__(self) -> int:
        """Get length of the textual header (number of characters)."""
        return self.rows * self.cols

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        return np.dtype((self.format, len(self)))

    def _decode(self, buffer: bytes) -> str:
        """Decode EBCDIC or ASCII bytes into string."""
        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer_int = np.frombuffer(buffer, dtype=self.dtype)
            buffer = EBCDIC_TO_ASCII[buffer_int].tobytes()

        return buffer.decode("ascii", errors="ignore")

    def _encode(self, text_header: str) -> bytes:
        """Encode string to EBCDIC or ASCII bytes."""
        if len(text_header) != len(self):
            msg = "Text length must be equal to rows x cols."
            raise ValueError(msg)

        buffer = text_header.encode("ascii")

        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer_int = np.frombuffer(buffer, dtype=self.dtype)
            buffer = ASCII_TO_EBCDIC[buffer_int].tobytes()

        return buffer

    def _wrap(self, string: str) -> str:
        """Wrap text header string to be multi-line with 80 character columns."""
        if len(string) != len(self):
            msg = "rows x cols must be equal wrapped text length."
            raise ValueError(msg)

        rows = []
        for row_idx in range(self.rows):
            start = row_idx * self.cols
            stop = start + self.cols
            rows.append(string[start:stop])

        return "\n".join(rows)

    @staticmethod
    def _unwrap(text_header: str) -> str:
        """Unwrap a multi-line string to a single line."""
        return text_header.replace("\n", "")
