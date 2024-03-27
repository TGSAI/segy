"""Tests for components that define fields and data types of a SEGY Schema."""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

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
@pytest.mark.parametrize("description", [None, "this is a data type description"])
def test_data_type_descriptor(
    format_: tuple[str, str], description: str | None
) -> None:
    """Test creating data type descriptors for `ScalarType`s."""
    format_str, format_char = format_
    new_dt_descr = DataTypeDescriptor(
        format=ScalarType(format_str),
        description=description,
    )
    assert new_dt_descr.format == format_str
    assert new_dt_descr.description == description
    assert new_dt_descr.format.char == np.sctype2char(format_char)  # noqa: NPY201
    assert new_dt_descr.itemsize == new_dt_descr.dtype.itemsize
    assert new_dt_descr.dtype == np.dtype(format_char)


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
    name: str,
    offset: int,
    description: str | None = None,
) -> StructuredFieldDescriptor:
    """Convenience helper for creating the StrucuredFieldDescriptors."""
    return StructuredFieldDescriptor(
        format=ScalarType(format_),
        name=name,
        offset=offset,
        description=description,
    )


def build_sdt_fields(
    *params: tuple[str, str, int],
) -> tuple[StructuredFieldDescriptor, ...]:
    """Convenience for creating a list of StructuredFieldDescriptors."""
    return tuple(build_sfd_helper(*p) for p in params)


@pytest.mark.parametrize(
    ("fields", "item_size", "offset"),
    [
        (
            build_sdt_fields(
                ("int32", "varA", 2), ("int16", "varB", 0), ("int32", "varC", 6)
            ),
            10,
            0,
        ),
        (
            build_sdt_fields(
                ("float32", "varA", 0),
                ("float32", "varB", 4),
            ),
            8,
            12,
        ),
        (
            build_sdt_fields(
                ("ibm32", "varA", 0),
                ("float32", "varB", 4),
            ),
            None,
            12,
        ),
    ],
)
def test_structured_data_type_descriptor(
    fields: tuple[StructuredFieldDescriptor, ...], item_size: int | None, offset: int
) -> None:
    """This tests for creatin a StructuredDataTypeDescriptor for different component data types."""
    new_sdtd = StructuredDataTypeDescriptor(
        fields=list(fields), item_size=item_size, offset=offset
    )
    assert new_sdtd.dtype.names == tuple([f.name for f in fields])
    if item_size is not None:
        assert new_sdtd.item_size == new_sdtd.dtype.itemsize
    else:
        assert new_sdtd.item_size == item_size


def test_trace_data_descriptors(trace_data_descriptors: TraceDataDescriptor) -> None:
    """Test for reading trace data descriptors.

    Tested on a buffer of random values and compares descriptor
    dtype results to a standard numpy struct to parse the same values.
    """
    samples = trace_data_descriptors.samples
    format_ = trace_data_descriptors.format.char

    expected = np.dtype(f"({samples},){format_}")

    assert trace_data_descriptors.dtype == expected


def test_validate_json_structured_data_type_descriptor() -> None:
    """Test for validating recreating a StrucrutedDataTypeDescriptor from a JSON string."""
    struct_json = """
    {
      "description": "dummy description",
      "fields": [
        {"description": "field1", "format": "int32", "name": "f1", "offset": 0},
        {"description": "field2", "format": "ibm32", "name": "f2", "offset": 4}
      ],
      "itemSize": 8,
      "offset": 200,
      "endianness": "big"
    }
    """

    expected_model = StructuredDataTypeDescriptor(
        description="dummy description",
        fields=[
            StructuredFieldDescriptor(
                description="field1", format=ScalarType.INT32, name="f1", offset=0
            ),
            StructuredFieldDescriptor(
                description="field2", format=ScalarType.IBM32, name="f2", offset=4
            ),
        ],
        item_size=8,
        offset=200,
        endianness=Endianness.BIG,
    )

    actual_model = StructuredDataTypeDescriptor.model_validate_json(struct_json)
    assert actual_model == expected_model


def _compare_json_strings(s1: str, s2: str) -> bool:
    """Helper function for clearing whitespace to compare json strings."""
    remove = string.whitespace
    mapping = {ord(c): None for c in remove}
    return s1.translate(mapping) == s2.translate(mapping)
