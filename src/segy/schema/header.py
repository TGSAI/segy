"""Classes for managing headers and header groups."""
import textwrap
from enum import StrEnum
from io import BufferedReader
from typing import Optional
from typing import TypeAlias

import numpy as np
from pydantic import Field

from segy.ebcdic import ASCII_TO_EBCDIC
from segy.ebcdic import EBCDIC_TO_ASCII
from segy.schema.base import BaseTypeDescriptor
from segy.schema.data_type import DataTypeDescriptor
from segy.schema.data_type import ScalarType


class StructuredFieldDescriptor(DataTypeDescriptor):
    """A descriptor class for a structured data-type field."""

    name: str = Field(..., description="The short name of the field.")
    offset: int = Field(..., ge=0, description="Starting byte offset.")


class StructuredDataTypeDescriptor(BaseTypeDescriptor):
    """A descriptor class for a structured array data-type."""

    fields: list[StructuredFieldDescriptor] = Field(
        ..., description="Fields of the structured data type."
    )
    item_size: Optional[int] = Field(
        default=None, description="Expected size of the struct."
    )
    offset: Optional[int] = Field(
        default=None, ge=0, description="Starting byte offset."
    )

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
    offset: Optional[int] = Field(
        default=None, ge=0, description="Starting byte offset."
    )

    def __len__(self) -> int:
        """Get length of the textual header (number of characters)."""
        return self.rows * self.cols

    @property
    def dtype(self) -> np.dtype:
        """Get numpy dtype."""
        return np.dtype((self.format, len(self)))

    def _decode(self, buffer: bytes) -> str:
        """Decode EBCDIC or ASCII bytes into string."""
        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer = np.frombuffer(buffer, dtype=self.dtype)
            buffer = EBCDIC_TO_ASCII[buffer].tobytes()

        return buffer.decode("ascii")

    def _encode(self, text_header: str) -> bytes:
        """Encode string to EBCDIC or ASCII bytes."""
        if len(text_header) != len(self):
            msg = "Text length must be equal to rows x cols."
            raise ValueError(msg)

        buffer = text_header.encode("ascii")

        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer = np.frombuffer(buffer, dtype=self.dtype)
            buffer = ASCII_TO_EBCDIC[buffer].tobytes()

        return buffer

    def _wrap(self, string: str) -> str:
        """Wrap text header string to be multi-line with 80 character columns."""
        if len(string) != len(self):
            msg = "rows x cols must be equal wrapped text length."
            raise ValueError(msg)

        return textwrap.fill(string, width=80, drop_whitespace=False)

    @staticmethod
    def _unwrap(text_header: str) -> str:
        """Unwrap a multi-line string to a single line."""
        return text_header.replace("\n", "")

    def read(self, file_pointer: BufferedReader) -> str:
        """Read and decode textual header from a file."""
        file_pointer.seek(self.offset)
        buffer = file_pointer.read(self.dtype.itemsize)
        text_header = self._decode(buffer)
        return self._wrap(text_header)

    def write(self, text_header: str, file_pointer: BufferedReader) -> None:
        """Encode and write the textual header to a file."""
        text_header = self._unwrap(text_header)
        buffer = self._encode(text_header)

        file_pointer.seek(self.offset)
        file_pointer.write(buffer)


TraceHeaderDescriptor: TypeAlias = StructuredDataTypeDescriptor
BinaryHeaderDescriptor: TypeAlias = StructuredDataTypeDescriptor
HeaderFieldDescriptor: TypeAlias = StructuredFieldDescriptor
