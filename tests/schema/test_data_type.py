"""Tests for components that define fields and data types of a SEGY Schema."""


from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from segy.schema.data_type import DataTypeDescriptor
from segy.schema.data_type import Endianness
from segy.schema.data_type import ScalarType
from segy.schema.data_type import StructuredDataTypeDescriptor
from segy.schema.data_type import StructuredFieldDescriptor

if TYPE_CHECKING:
    from segy.schema.trace import TraceDataDescriptor


@pytest.mark.parametrize(
    "format_",
    [
        ("ibm32", "u4"),
        ("int64", "i8"),
        ("int32", "i4"),
        ("int16", "i2"),
        ("int8", "i1"),
        ("uint64", "u8"),
        ("uint32", "u4"),
        ("uint16", "u2"),
        ("uint8", "u1"),
        ("float64", "f8"),
        ("float32", "f4"),
        ("float16", "f2"),
    ],
)
@pytest.mark.parametrize("endianness", ["big", "little"])
@pytest.mark.parametrize("description", [None, "this is a data type description"])
def test_data_type_descriptor(
    format_: tuple[str, str], endianness: str, description: str | None
) -> None:
    """Test creating data type descriptors for `ScalarType`s."""
    format_str, format_char = format_
    new_dt_descr = DataTypeDescriptor(
        format=ScalarType(format_str),
        endianness=Endianness(endianness),
        description=description,
    )
    assert new_dt_descr.format == format_str
    assert new_dt_descr.endianness == endianness
    assert new_dt_descr.description == description
    assert new_dt_descr.format.char == np.sctype2char(format_char)  # noqa: NPY201
    assert new_dt_descr.itemsize == new_dt_descr.dtype.itemsize
    assert new_dt_descr.dtype == np.dtype(
        f"{new_dt_descr.endianness.symbol}{format_char}"
    )


@pytest.mark.parametrize("name", ["varA"])
@pytest.mark.parametrize("offset", [0, 2, 5, -2])
def test_structured_field_descriptor(
    data_types: DataTypeDescriptor, name: str, offset: int
) -> None:
    """Test StructuredFieldDescriptors with ScalarTypes and offsets."""
    if offset < 0:
        with pytest.raises(ValidationError):
            StructuredFieldDescriptor(
                name=name, offset=offset, **data_types.model_dump()
            )
    else:
        new_sfd = StructuredFieldDescriptor(
            name=name, offset=offset, **data_types.model_dump()
        )
        assert new_sfd.name == name
        assert new_sfd.offset == offset


def build_sfd_helper(
    format_: str,
    endianness: str,
    name: str,
    offset: int,
    description: str | None = None,
) -> StructuredFieldDescriptor:
    """Convenience helper for creating the StrucuredFieldDescriptors."""
    return StructuredFieldDescriptor(
        format=ScalarType(format_),
        endianness=Endianness(endianness),
        name=name,
        offset=offset,
        description=description,
    )


def build_sdt_fields(*params: dict[str, Any]) -> list[StructuredFieldDescriptor]:
    """Convenience for creating a list of StructuredFieldDescriptors."""
    return [build_sfd_helper(*p) for p in params]


@pytest.mark.parametrize(
    ("fields", "item_size", "offset"),
    [
        (
            build_sdt_fields(
                ("int32", "little", "varA", 2),
                ("int16", "little", "varB", 0),
                ("int32", "little", "varC", 6),
            ),
            10,
            0,
        ),
        (
            build_sdt_fields(
                ("float32", "big", "varA", 0), ("float32", "big", "varB", 4)
            ),
            8,
            12,
        ),
    ],
)
def test_structured_data_type_descriptor(
    fields: list[StructuredFieldDescriptor], item_size: int, offset: int
) -> None:
    """This tests for creatin a StructuredDataTypeDescriptor for different component data types."""
    new_sdtd = StructuredDataTypeDescriptor(
        fields=fields, item_size=item_size, offset=offset
    )
    assert new_sdtd.dtype.names == tuple([f.name for f in fields])
    assert new_sdtd.item_size == new_sdtd.dtype.itemsize


def test_trace_data_descriptors(trace_data_descriptors: TraceDataDescriptor) -> None:
    """Test for reading trace data descriptors.

    Tested on a buffer of random values and compares descriptor
    dtype results to a standard numpy struct to parse the same values.
    """
    samples = trace_data_descriptors.samples
    endianness = trace_data_descriptors.endianness.symbol
    format_ = trace_data_descriptors.format.char

    expected = np.dtype(f"({samples},){endianness}{format_}")

    assert trace_data_descriptors.dtype == expected