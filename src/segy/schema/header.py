"""Specification representing header fields and headers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from segy.schema.base import BaseDataType
from segy.schema.base import Endianness
from segy.schema.format import ScalarType  # noqa: TCH001


class HeaderField(BaseDataType):
    """A class representing header field spec.

    Examples:
        A named float starting at byte location 9:

        >>> field = HeaderField(
        >>>     name="my_var",
        >>>     format="float32",
        >>>     byte=9,
        >>> )

        The name, byte, and offset fields will only be used if the structured
        field is used within the context of a :class:`HeaderSpec`. Offset is
        calculated automatically from byte location.

        >>> field.name
        my_var
        >>> field.byte
        9
        >>> field.offset
        8

        The `dtype` property is inherited from :class:`DataFormat`.

        >>> field.dtype
        dtype('float32')
    """

    name: str = Field(..., description="The short name of the field.")
    byte: int = Field(..., ge=1, description="Field's start byte location.")
    format: ScalarType = Field(..., description="The data type of the field.")  # noqa: A003

    @property
    def offset(self) -> int:
        """Return zero based offset from one based byte location."""
        return self.byte - 1

    @property
    def dtype(self) -> np.dtype[Any]:
        """Converts the byte order and data type of the object into a NumPy dtype."""
        return np.dtype(self.format.char)


class HeaderSpec(BaseDataType):
    """A class representing a header specification.

    Examples:
        Let's build a header from scratch!

        We will define three fields with different names, data-types, and
        start byte locations.

        >>> field1 = HeaderField(
        >>>     name="foo",
        >>>     format="int32",
        >>>     byte=1,
        >>> )
        >>> field2 = HeaderField(
        >>>     name="bar",
        >>>     format="int16",
        >>>     byte=5,
        >>> )
        >>> field3 = HeaderField(
        >>>     name="fizz",
        >>>     format="int32",
        >>>     byte=17,
        >>> )

        Note that the fields span the following byte ranges:

        * `field1` between bytes `[0, 4)`
        * `field2` between bytes `[4, 6)`
        * `field3` between bytes `[16, 20)`

        The gap between `field2` and `field3` will be padded with `void`. In
        this case we expect to see an item size of 20-bytes (total length of
        the header struct).

        >>> header = HeaderSpec(
        >>>     fields=[field1, field2, field3],
        >>> )

        Now let's look at its data type:

        >>> header.dtype
        dtype({'names': ['foo', 'bar', 'fizz'], 'formats': ['<i4', '<i2', '<i4'], 'offsets': [0, 4, 16], 'itemsize': 20})

        If we wanted to pad the end of the struct (to fit a specific byte range),
        we would provide the item_size in the spec. If we set it to 30, this means
        that we padded the struct by 10 bytes at the end.

        >>> header = HeaderSpec(
        >>>     fields=[field1, field2, field3],
        >>>     item_size=30,
        >>> )

        Now let's look at its data type:

        >>> header.dtype
        dtype({'names': ['foo', 'bar', 'fizz'], 'formats': ['<i4', '<i2', '<i4'], 'offsets': [0, 4, 16], 'itemsize': 30})

        To see what's going under the hood, we can look at a lower level numpy
        description of the `dtype`. Here we observe all the gaps (void types).

        >>> header.dtype.descr
        [('foo', '<i4'), ('bar', '<i2'), ('', '|V10'), ('fizz', '<i4'), ('', '|V10')]
    """  # noqa: E501

    fields: list[HeaderField] = Field(
        ..., description="List containing multiple header field spec instances."
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
            "names": self.names,
            "formats": self.formats,
            "offsets": self.offsets,
        }

        if self.item_size is not None:
            dtype_conf["itemsize"] = self.item_size

        struct_dtype = np.dtype(dtype_conf)  # type: ignore[call-overload]

        if self.endianness is not None:
            struct_dtype = struct_dtype.newbyteorder(self.endianness.symbol)

        return struct_dtype  # type: ignore[no-any-return]

    @property
    def names(self) -> list[str]:
        """Get the names of the fields."""
        return [field.name for field in self.fields]

    @property
    def formats(self) -> list[np.dtype[Any]]:
        """Get the formats of the fields."""
        return [field.dtype for field in self.fields]

    @property
    def offsets(self) -> list[int]:
        """Get the offsets the fields."""
        return [field.offset for field in self.fields]

    @field_validator("fields")
    @classmethod
    def ensure_no_duplicate_fields(
        cls: type[HeaderSpec], fields: list[HeaderField]
    ) -> list[HeaderField]:
        """Check if fields are unique and error out if not."""
        name_counter = Counter(field.name for field in fields)
        duplicates = [name for name, count in name_counter.items() if count > 1]

        if duplicates:
            msg = f"Duplicate header fields detected: {', '.join(duplicates)}."
            raise ValueError(msg)

        return fields

    @model_validator(mode="after")
    def ensure_offsets_in_itemsize_bounds(self) -> HeaderSpec:
        """Ensure fields don't go above the allowed itemsize."""
        if len(self.fields) == 0:
            return self

        max_field = max(self.fields, key=lambda field: field.offset)

        if self.item_size is None:
            return self

        if max_field.offset + max_field.dtype.itemsize > self.item_size:
            msg = "Offsets exceed allowed header size."
            raise ValueError(msg)

        return self

    def add_field(self, field: HeaderField, overwrite: bool = False) -> None:
        """Add a field to the structured data type."""
        if field.name in self.names and overwrite is False:
            msg = (
                f"Field named {field.name} already exists. If you wish to "
                "overwrite an existing field, pass `overwrite=True`."
            )
            raise KeyError(msg)

        if field.name not in self.names:
            self.fields.append(field)
        else:
            field_idx = self.names.index(field.name)
            self.fields[field_idx] = field

        # Trigger validation
        self.fields = self.fields

    def remove_field(self, name: str) -> None:
        """Remove a field from the structured data type by name."""
        try:
            field_idx = self.names.index(name)
        except ValueError as err:
            msg = f"Field named {name} does not exist."
            raise KeyError(msg) from err

        del self.fields[field_idx]
