"""Classes for managing headers and header groups."""
from enum import StrEnum
from io import BufferedReader
from typing import Optional
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from segy_ninja.ebcdic import ASCII_TO_EBCDIC
from segy_ninja.ebcdic import EBCDIC_TO_ASCII
from segy_ninja.schema.base import BaseTypeDescriptor
from segy_ninja.schema.data_type import DataTypeDescriptor
from segy_ninja.schema.data_type import ScalarType


class StructuredFieldDescriptor(DataTypeDescriptor):
    name: str = Field(..., description="The short name of the field.")
    offset: int = Field(..., ge=0, description="Starting byte offset.")


class StructuredDataTypeDescriptor(BaseTypeDescriptor):
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
        """The NumPy structured dtype object corresponding to the header fields."""
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

    def read(self, file_pointer: BufferedReader) -> NDArray:
        file_pointer.seek(self.offset)

        buffer_size = self.itemsize
        buffer = bytearray(buffer_size)
        file_pointer.readinto(buffer)

        array = np.frombuffer(buffer, dtype=self.dtype)

        # TODO: Add little-endian support. Currently assume big-endian.
        # TODO: Add IBM32 parsing support.
        array = array.byteswap(inplace=True).newbyteorder()

        return array


class TextHeaderEncoding(StrEnum):
    ASCII = "ascii"
    EBCDIC = "ebcdic"


class TextHeaderDescriptor(BaseTypeDescriptor):
    rows: int = Field(..., description="Number of rows in text header.")
    cols: int = Field(..., description="Number of columns in text header.")
    encoding: TextHeaderEncoding = Field(..., description="String encoding.")
    format: ScalarType = Field(..., description="Type of string.")
    offset: Optional[int] = Field(
        default=None, ge=0, description="Starting byte offset."
    )

    def __len__(self) -> int:
        return self.rows * self.cols

    @property
    def dtype(self) -> np.dtype:
        return np.dtype((self.format, len(self)))

    def _decode(self, buffer: bytes) -> str:
        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer = np.frombuffer(buffer, dtype=self.dtype)
            buffer = EBCDIC_TO_ASCII[buffer].tobytes()

        return buffer.decode("ascii")

    def _encode(self, text_header) -> bytes:
        if len(text_header) != len(self):
            raise ValueError("Text length must be equal to rows x cols.")

        buffer = text_header.encode("ascii")

        if self.encoding == TextHeaderEncoding.EBCDIC:
            buffer = np.frombuffer(buffer, dtype=self.dtype)
            buffer = ASCII_TO_EBCDIC[buffer].tobytes()

        return buffer

    def _unwrap(self, wrapped_text: str, line_sep: str = "\n") -> str:
        if len(wrapped_text) != len(self):
            raise ValueError("rows x cols must be equal wrapped text length.")

        rows_text = []
        for idx in range(self.rows):
            start = idx * self.cols
            stop = start + self.cols
            rows_text.append(wrapped_text[start:stop])

        unwrapped = line_sep.join(rows_text)

        return unwrapped

    @staticmethod
    def _wrap(text_header: str, line_sep: str = "\n") -> str:
        return text_header.replace(line_sep, "")

    def read(self, file_pointer: BufferedReader) -> str:
        file_pointer.seek(self.offset)
        buffer = file_pointer.read(self.dtype.itemsize)
        text_header = self._decode(buffer)
        return self._unwrap(text_header)

    def write(self, text_header: str, file_pointer: BufferedReader) -> None:
        text_header = self._wrap(text_header)
        buffer = self._encode(text_header)

        file_pointer.seek(self.offset)
        file_pointer.write(buffer)


TraceHeaderDescriptor: TypeAlias = StructuredDataTypeDescriptor
BinaryHeaderDescriptor: TypeAlias = StructuredDataTypeDescriptor
HeaderFieldDescriptor: TypeAlias = StructuredFieldDescriptor
