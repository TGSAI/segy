"""Data models representing data types."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import cast

import numpy as np
from pydantic import Field

from segy.compat import StrEnum
from segy.schema.base import BaseTypeDescriptor


class Endianness(StrEnum):
    """Enumeration class with three possible endianness values.

    Examples:
        >>> endian = Endianness.BIG
        >>> print(endian.symbol)
        >
    """

    BIG = "big"
    LITTLE = "little"
    NATIVE = "native"

    def __init__(self, _: str) -> None:
        self._symbol_map = {
            "big": ">",
            "little": "<",
            "native": "=",
        }

    @property
    def symbol(self) -> Literal["<", ">", "="]:
        """Get the numpy symbol for the endianness from mapping."""
        return cast(Literal["<", ">", "="], self._symbol_map[self.value])


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


class DataTypeDescriptor(BaseTypeDescriptor):
    """A class representing a descriptor for a data type.

    Examples:
        A float32:

        >>> data_type = DataTypeDescriptor(format="float32")
        >>> data_type.dtype
        dtype('float32')

        A 16-bit unsigned integer:

        >>> data_type = DataTypeDescriptor(format="uint16")
        >>> data_type.dtype
        dtype('uint16')
    """

    format: ScalarType = Field(..., description="The data type of the field.")  # noqa: A003

    @property
    def dtype(self) -> np.dtype[Any]:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        return np.dtype(self.format.char)


class StructuredFieldDescriptor(DataTypeDescriptor):
    """A class representing a descriptor for a structured data-type field.

    Examples:
        A named float at offset 8-bytes:

        >>> data_type = StructuredFieldDescriptor(
        >>>     name="my_var",
        >>>     format="float32",
        >>>     offset=8,
        >>> )

        The name and offset fields will only be used if the structured
        field is used within the context of a :class:`StructuredDataTypeDescriptor`.

        >>> data_type.name
        my_var
        >>> data_type.offset
        8

        The `dtype` property is inherited from :class:`DataTypeDescriptor`.

        >>> data_type.dtype
        dtype('float32')
    """

    name: str = Field(..., description="The short name of the field.")
    offset: int = Field(..., ge=0, description="Starting byte offset.")


class StructuredDataTypeDescriptor(BaseTypeDescriptor):
    """A class representing a descriptor for a structured data-type.

    Examples:
        Let's build a structured data type from scratch!

        We will define three fields with different names, data-types, and
        starting offsets.

        >>> field1 = StructuredFieldDescriptor(
        >>>     name="foo",
        >>>     format="int32",
        >>>     offset=0,
        >>> )
        >>> field2 = StructuredFieldDescriptor(
        >>>     name="bar",
        >>>     format="int16",
        >>>     offset=4,
        >>> )
        >>> field3 = StructuredFieldDescriptor(
        >>>     name="fizz",
        >>>     format="int32",
        >>>     offset=16,
        >>> )

        Note that the fields span the following byte ranges:

        * `field1` between bytes `[0, 4)`
        * `field2` between bytes `[4, 6)`
        * `field3` between bytes `[16, 20)`

        The gap between `field2` and `field3` will be padded with `void`. In
        this case we expect to see an item size of 20-bytes (total length of
        the struct).

        >>> struct_dtype = StructuredDataTypeDescriptor(
        >>>     fields=[field1, field2, field3],
        >>> )

        Now let's look at its data type:

        >>> struct_dtype.dtype
        dtype({'names': ['foo', 'bar', 'fizz'], 'formats': ['<i4', '<i2', '<i4'], 'offsets': [0, 4, 16], 'itemsize': 20})

        If we wanted to pad the end of the struct (to fit a specific byte range),
        we would provide the item_size in the descriptor. If we set it to 30,
        this means that we padded the struct by 10 bytes at the end.

        >>> struct_dtype = StructuredDataTypeDescriptor(
        >>>     fields=[field1, field2, field3],
        >>>     item_size=30,
        >>> )

        Now let's look at its data type:

        >>> struct_dtype.dtype
        dtype({'names': ['foo', 'bar', 'fizz'], 'formats': ['<i4', '<i2', '<i4'], 'offsets': [0, 4, 16], 'itemsize': 30})

        To see what's going under the hood, we can look at a lower level numpy
        description of the `dtype`. Here we observe all the gaps (void types).

        >>> struct_dtype.dtype.descr
        [('foo', '<i4'), ('bar', '<i2'), ('', '|V10'), ('fizz', '<i4'), ('', '|V10')]
    """  # noqa: E501

    fields: list[StructuredFieldDescriptor] = Field(
        ..., description="A list of descriptors for a structured data-type."
    )
    item_size: int | None = Field(
        default=None, description="Expected size of the struct."
    )
    offset: int | None = Field(default=None, ge=0, description="Starting byte offset.")

    endianness: Endianness | None = Field(
        default=None, description="Endianness of structured data type."
    )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Converts the names, data types, and offsets of the object into a NumPy dtype."""
        dtype_conf = {
            "names": [field.name for field in self.fields],
            "formats": [field.dtype for field in self.fields],
            "offsets": [field.offset for field in self.fields],
        }

        if self.item_size is not None:
            dtype_conf["itemsize"] = self.item_size

        struct_dtype = np.dtype(dtype_conf)  # type: ignore[call-overload]

        if self.endianness is not None:
            struct_dtype = struct_dtype.newbyteorder(self.endianness.symbol)

        return struct_dtype  # type: ignore[no-any-return]
