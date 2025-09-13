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


def _merge_headers(spec: HeaderSpec, new_fields: list[HeaderField]) -> None:
    """Merges existing headers with new headers.

    Args:
        spec: HeaderSpec to modify.
        new_fields: List of new header fields.
    """
    _validate_non_overlapping_headers(new_fields)
    _merge_headers_by_name(spec, new_fields)
    _merge_headers_by_byte_offset(spec, new_fields)


def _merge_headers_by_name(spec: HeaderSpec, new_fields: list[HeaderField]) -> None:
    """Replaces existing headers with new headers that have the same name.

    Args:
        spec: HeaderSpec to modify.
        new_fields: List of new header fields.
    """
    for field in new_fields:
        spec.add_field(field, overwrite=True)


def _merge_headers_by_byte_offset(
    spec: HeaderSpec, new_fields: list[HeaderField]
) -> None:
    """Removes existing headers that have bytes that would have been overlapped by new headers.

    Intended to be run AFTER _merge_headers_by_name.
    This algorithm will ensure all neighboring headers are not overlapped.

    An overlap is anywhere that an `existing_spec` header byte width would have ANY intersection
    with a `new_field` byte width.

    Args:
        spec: HeaderSpec to modify.
        new_fields: List of new header fields.

    Returns:
        List of header fields with duplicates removed.
    """
    ranges = [(field.name, field.range) for field in spec.fields]
    ranges.sort(key=lambda range_tuple: range_tuple[1][0])
    fields_to_remove = []
    processed_new_fields = set()  # Track which new fields have already caused a removal

    for i in range(len(ranges) - 1):
        current_key, current_range = ranges[i]
        next_key, next_range = ranges[i + 1]
        if ranges_overlap(current_range, next_range):
            # Only remove existing fields that overlap with new fields
            # Each new field should cause at most one removal
            current_is_new = any(field.name == current_key for field in new_fields)
            next_is_new = any(field.name == next_key for field in new_fields)

            if (
                current_is_new
                and not next_is_new
                and current_key not in processed_new_fields
            ):
                # Current is new, next is existing - remove next (existing)
                fields_to_remove.append(next_key)
                processed_new_fields.add(current_key)
            elif (
                not current_is_new
                and next_is_new
                and next_key not in processed_new_fields
            ):
                # Current is existing, next is new - remove current (existing)
                fields_to_remove.append(current_key)
                processed_new_fields.add(next_key)

    # Remove fields using the HeaderSpec's remove_field method
    for field_name in fields_to_remove:
        spec.remove_field(field_name)


def _validate_non_overlapping_headers(new_fields: list[HeaderField]) -> None:
    """Validates that a list of new headers have unique names and do not overlap one-another.

    Args:
        new_fields: List of new header fields.

    Raises:
        ValueError: If duplicate header field names are detected.
        ValueError: If header fields overlap.
    """
    names = [field.name for field in new_fields]
    if len(names) != len(set(names)):
        msg = f"Duplicate header field names detected: {names}!"
        raise ValueError(msg)

    ranges = [field.range for field in new_fields]
    ranges.sort(key=lambda range_tuple: range_tuple[0])

    for i in range(len(ranges) - 1):
        if ranges_overlap(ranges[i], ranges[i + 1]):
            msg = f"Header fields overlap: {ranges[i]} and {ranges[i + 1]}!"
            raise ValueError(msg)


def ranges_overlap(range1: tuple[int, int], range2: tuple[int, int]) -> bool:
    """Checks if two right half-open ranges overlap."""
    return range1[0] < range2[1] and range1[1] > range2[0]


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
        """Converts the data type of the object into a NumPy dtype."""
        return self.format.dtype

    @property
    def range(self) -> tuple[int, int]:
        """Return the start and stop byte location of the field.

        Note: This return is Fortran-style and right half-open. [start, stop)
        """
        return self.byte, self.byte + self.dtype.itemsize


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

    def customize(self, fields: list[HeaderField]) -> None:
        """Customize a HeaderField with user defined fields."""
        _merge_headers(self, fields)
