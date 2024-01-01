from io import BufferedReader
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from segy.ibm import ibm2ieee
from segy.schema.base import BaseTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.header import TraceHeaderDescriptor


class TraceDataDescriptor(BaseTypeDescriptor):
    format: ScalarType = Field(...)
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
        format_char = self.format.char
        dtype_str = "".join([self.endianness.symbol, str(self.samples), format_char])
        return np.dtype(dtype_str)


class TraceDescriptor(BaseTypeDescriptor):
    header_descriptor: TraceHeaderDescriptor = Field(...)
    data_descriptor: TraceDataDescriptor = Field(...)
    offset: Optional[int] = Field(default=None)

    @property
    def dtype(self):
        header_dtype = self.header_descriptor.dtype
        data_dtype = self.data_descriptor.dtype

        return np.dtype([("header", header_dtype), ("data", data_dtype)])

    def read(self, file_pointer: BufferedReader, n_trc=1000) -> NDArray:
        file_pointer.seek(self.offset)

        buffer_size = n_trc * self.itemsize
        buffer = bytearray(buffer_size)
        file_pointer.readinto(buffer)

        array = np.frombuffer(buffer, dtype=self.dtype)

        # TODO: Add little-endian support. Currently assume big-endian.
        array = array.byteswap(inplace=True).newbyteorder()

        if self.data_descriptor.format == ScalarType.IBM32:
            header_dtype = array.dtype["header"]
            data_dtype = np.dtype(("float32", self.data_descriptor.samples))

            dtype_new = np.dtype([("header", header_dtype), ("data", data_dtype)])
            array["data"] = ibm2ieee(array["data"]).view("uint32")

            array = array.view(dtype_new)

        return array
