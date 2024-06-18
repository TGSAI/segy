"""Classes for managing headers and header groups."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from pydantic import Field

from segy.ebcdic import ASCII_TO_EBCDIC
from segy.ebcdic import EBCDIC_TO_ASCII
from segy.schema.base import BaseDataType
from segy.schema.format import TextHeaderEncoding  # noqa: TCH001


class TextProcessor:
    """Processor to transcode and (un)wrap text.

    Args:
        rows: Number of rows in text. Used for wrap/unwrap.
        cols: Number of columns in text. Used for wrap/unwrap.
        encoding: Text encoding used in transcoding.
    """

    def __init__(self, rows: int, cols: int, encoding: TextHeaderEncoding):
        self.rows = rows
        self.cols = cols
        self.encoding = encoding
        self.dtype = encoding.dtype

    def decode(self, buffer: bytes) -> str:
        """Decode bytes into a string given encoding."""
        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer_int = np.frombuffer(buffer, dtype=self.dtype)
            buffer = EBCDIC_TO_ASCII[buffer_int].tobytes()

        return buffer.decode("ascii", errors="replace")

    def encode(self, text: str) -> bytes:
        """Encode string into bytes given encoding."""
        buffer = text.encode("ascii")

        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer_int = np.frombuffer(buffer, dtype=self.dtype)
            buffer = ASCII_TO_EBCDIC[buffer_int].tobytes()

        return buffer

    def wrap(self, string: str) -> str:
        """Add line breaks after each row (n-cols)."""
        lines = []
        for row_idx in range(self.rows):
            start = row_idx * self.cols
            stop = start + self.cols
            lines.append(string[start:stop])

        return "\n".join(lines)

    @staticmethod
    def unwrap(text: str) -> str:
        """Remove all line breaks."""
        return text.replace("\n", "")


class TextHeaderSpec(BaseDataType):
    """A class representing spec for SEG-Y textual headers (8-bit)."""

    rows: int = Field(default=40, description="Number of rows in text header.")
    cols: int = Field(default=80, description="Number of columns in text header.")
    encoding: TextHeaderEncoding = Field(
        default=TextHeaderEncoding.EBCDIC, description="String encoding."
    )
    offset: int | None = Field(default=None, ge=0, description="Starting byte offset.")

    @cached_property
    def processor(self) -> TextProcessor:
        """Prepare transforms for encoding / decoding."""
        return TextProcessor(self.rows, self.cols, self.encoding)

    def __len__(self) -> int:
        """Get length of the textual header (number of characters)."""
        return self.rows * self.cols

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        return np.dtype((self.encoding.dtype, len(self)))

    def decode(self, buffer: bytes) -> str:
        """Decode EBCDIC or ASCII bytes into string."""
        string = self.processor.decode(buffer)
        return self.processor.wrap(string)

    def encode(self, string: str) -> bytes:
        """Encode string to EBCDIC or ASCII bytes."""
        string = self.processor.unwrap(string)
        return self.processor.encode(string)


class ExtendedTextHeaderSpec(BaseDataType):
    """A class representing spec for SEG-Y extended textual headers."""

    spec: TextHeaderSpec = Field(..., description="Extended text header spec.")
    count: int = Field(default=0, ge=0, description="Number of extended text headers.")
    offset: int | None = Field(default=None, ge=0, description="Starting byte offset.")

    def __len__(self) -> int:
        """Get length of the textual header (number of characters)."""
        return len(self.spec) * self.count

    @property
    def dtype(self) -> np.dtype[Any]:
        """Get numpy dtype."""
        return np.dtype((self.spec.encoding.dtype, len(self)))

    def decode(self, buffer: bytes) -> list[str]:
        """Decode EBCDIC or ASCII bytes into string."""
        string = self.spec.processor.decode(buffer)

        chunk_size = len(self.spec)
        strings = []
        for start in range(0, len(self), chunk_size):
            stop = start + chunk_size
            text = self.spec.processor.wrap(string[start:stop])
            strings.append(text)

        return strings

    def encode(self, strings: list[str]) -> bytes:
        """Encode string to EBCDIC or ASCII bytes."""
        string = "".join(strings)
        string = self.spec.processor.unwrap(string)
        return self.spec.processor.encode(string)
